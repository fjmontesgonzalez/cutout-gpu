"""Prueba de humo contra un servidor de recorte v3 en marcha (sólo stdlib + Pillow + numpy).

    python smoke_test.py http://127.0.0.1:9000 <token> [foto.jpg]

Comprueba: /health, /initialize (carga de modelos base + SAM 3), 401 sin token válido,
/categories, y que POST / responde con el contrato: 200 + PNG RGBA del mismo tamaño con
veredicto ok, o 422 JSON "rejected" con motivo (en CPU/560 px SAM 3 puede no ver el libro
sintético: ambas son respuestas válidas del servidor). Con CUTOUT_ALLOW_DEBUG=1 pide además
debug=1 y verifica la estructura (alfas por modelo, máscaras de SAM 3, veredictos).
"""
from __future__ import annotations

import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image, ImageDraw


def _req(url, *, data=None, headers=None, method=None, timeout=60):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers), resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), exc.read()


def _multipart(jpeg: bytes, fields: dict):
    boundary = "----cutout" + uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append('--{}\r\nContent-Disposition: form-data; name="{}"\r\n\r\n{}\r\n'
                     .format(boundary, k, v).encode())
    parts.append('--{}\r\nContent-Disposition: form-data; name="file"; filename="photo.jpg"\r\n'
                 'Content-Type: image/jpeg\r\n\r\n'.format(boundary).encode() + jpeg + b"\r\n")
    parts.append("--{}--\r\n".format(boundary).encode())
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


def _photo(w=720, h=960) -> bytes:
    rng = np.random.default_rng(3)
    base = np.full((h, w, 3), (196, 178, 150), np.uint8)
    base = np.clip(base.astype(int) + rng.integers(-18, 18, (h, w, 1)), 0, 255).astype(np.uint8)
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    d.rectangle([120, 100, w - 120, h - 90], fill=(28, 52, 110))
    d.rectangle([170, 160, w - 170, 330], fill=(230, 200, 60))
    for i in range(10):
        d.rectangle([170, 400 + 28 * i, w - 170, 410 + 28 * i], fill=(235, 235, 235))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _cut_headers(headers: dict) -> dict:
    return {k: v for k, v in headers.items() if k.lower().startswith("x-cutout")}


def main() -> int:
    base, token = sys.argv[1].rstrip("/"), sys.argv[2]
    jpeg = open(sys.argv[3], "rb").read() if len(sys.argv) > 3 else _photo()
    size = Image.open(io.BytesIO(jpeg)).size

    for _ in range(120):
        try:
            status, _, body = _req(base + "/health", timeout=5)
            if status == 200:
                print("health:", body.decode())
                break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1)
    else:
        print("FAIL: el servidor no arrancó")
        return 1

    t = time.time()
    status, _, body = _req(base + "/initialize", data=b"", method="POST", timeout=3600)
    print("initialize: {} en {:.1f}s {}".format(status, time.time() - t, body.decode()[:400]))
    if status != 200:
        return 1
    init = json.loads(body.decode())
    if not init.get("bases") or not init.get("sam3_dtype"):
        print("FAIL: initialize sin modelos base o sin SAM 3")
        return 1

    status, _, body = _req(base + "/categories", timeout=30)
    cats = json.loads(body.decode())
    print("categories:", sorted(cats["categories"]), "hazards:", cats["hazards"])
    if status != 200 or "book" not in cats["categories"] or "hand" not in cats["hazards"]:
        print("FAIL: /categories incompleto")
        return 1

    data, ctype = _multipart(jpeg, {})
    status, _, body = _req(base + "/", data=data, method="POST", timeout=600,
                           headers={"Content-Type": ctype, "Authorization": "Bearer nope"})
    print("token malo:", status, body[:80])
    if status != 401:
        print("FAIL: esperaba 401")
        return 1

    failures = 0
    for fields in ({}, {"category": "bag"}, {"category": "book", "kind": "cara_plana"}):
        data, ctype = _multipart(jpeg, fields)
        t = time.time()
        status, headers, body = _req(base + "/", data=data, method="POST", timeout=1800,
                                     headers={"Content-Type": ctype, "Authorization": "Bearer " + token})
        cut = _cut_headers(headers)
        print("recorte {}: {} en {:.1f}s {}".format(fields or "normal", status, time.time() - t,
                                                    json.dumps(cut, ensure_ascii=False)))
        if "X-Cutout-Verdict" not in cut and "x-cutout-verdict" not in {k.lower() for k in cut}:
            print("FAIL: sin cabecera X-Cutout-Verdict")
            failures += 1
            continue
        if status == 422:
            try:
                err = json.loads(body.decode())
            except Exception:  # noqa: BLE001
                err = {}
            if err.get("error") != "rejected" or not err.get("reason"):
                print("FAIL: 422 sin motivo:", body[:200])
                failures += 1
            else:
                print("   rechazo legítimo:", err["reason"])
            continue
        if status != 200 or not body.startswith(b"\x89PNG"):
            print("FAIL:", body[:300])
            failures += 1
            continue
        img = Image.open(io.BytesIO(body))
        a = np.asarray(img.getchannel("A")) if img.mode == "RGBA" else None
        problems = []
        if img.mode != "RGBA":
            problems.append("modo {}".format(img.mode))
        if img.size != size:
            problems.append("tamaño {} != {}".format(img.size, size))
        if a is not None and ((a <= 8).mean() < 0.05 or (a >= 192).mean() < 0.05):
            problems.append("alfa sin fondo/producto: fondo={:.3f} producto={:.3f}".format(
                (a <= 8).mean(), (a >= 192).mean()))
        if problems:
            print("FAIL:", "; ".join(problems))
            failures += 1

    data, ctype = _multipart(jpeg, {"debug": "1"})
    status, headers, body = _req(base + "/", data=data, method="POST", timeout=1800,
                                 headers={"Content-Type": ctype, "Authorization": "Bearer " + token})
    if status == 200 and body[:1] == b"{":
        dbg = json.loads(body.decode())
        models = dbg.get("models", {})
        ok = bool(models) and all("base_png" in m and "final_png" in m and "verdict" in m for m in models.values()) \
            and "union_png" in dbg and "instances" in dbg.get("sam3", {})
        print("debug: modelos={} veredictos={} sam3={}".format(
            sorted(models), {k: v["verdict"] + "/" + v["reason"] for k, v in models.items()},
            {k: len(v) for k, v in dbg["sam3"]["instances"].items()}))
        if ok:
            for name, m in models.items():
                png = base64.b64decode(m["base_png"])
                if Image.open(io.BytesIO(png)).size != size:
                    print("FAIL: alfa de", name, "con otro tamaño")
                    failures += 1
        else:
            print("FAIL: estructura de debug incompleta")
            failures += 1
    else:
        print("debug no disponible (CUTOUT_ALLOW_DEBUG no está a 1):", status)
    print("OK" if not failures else "FALLOS: {}".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
