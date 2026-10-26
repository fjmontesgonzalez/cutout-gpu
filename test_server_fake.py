"""Pruebas del contrato HTTP del servidor v3 con modelos FALSOS (sin GPU ni pesos).

    python deploy/alibaba-cutout/test_server_fake.py

Sustituye `_base_alpha` y `_sam3_scene` por funciones sintéticas y comprueba rutas,
cabeceras, el 422 de rechazo, el modo debug, la selección de modelo base y la
configuración de categorías por variable de entorno.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ["CUTOUT_TOKEN_SHA256"] = hashlib.sha256(b"tok").hexdigest()
os.environ["CUTOUT_ALLOW_DEBUG"] = "1"
os.environ["CUTOUT_CATEGORIES"] = json.dumps({"shoe": {"prompts": ["shoe", "sneaker"]},
                                              "book": {"params": {"bite_deficit": 0.05}}})
os.environ["CUTOUT_HAZARDS"] = json.dumps({"support": ["book stand", "easel", "shelf"]})

import postproc as pp  # noqa: E402
import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

H, W = 1200, 900
BOOK = (150, 150, 750, 1050)
seen: dict = {}


def _brect(box):
    m = np.zeros((H, W), bool)
    x1, y1, x2, y2 = box
    m[y1:y2, x1:x2] = True
    return m


def _book_box(shape):
    """Caja del libro falso en proporción a la foto (900x1200 -> 150,150,750,1050)."""
    h, w = shape[:2]
    return int(w * 0.1667), int(h * 0.125), int(w * 0.8333), int(h * 0.875)


def fake_base(name, rgb):
    seen.setdefault("bases", []).append(name)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = _book_box(rgb.shape)
    a = np.zeros((h, w), np.uint8)
    a[y1:y2, x1:x2] = 255
    if name == "rmbg2":
        a[y2:int(h * 0.925), int(w * 0.111):int(w * 0.889)] = 255   # rmbg2 "ve" también una repisa
    return a, {"work": "{}x{}".format(*pp.work_size(w, h))}


def fake_scene(rgb, prompts, hazards):
    seen["prompts"] = list(prompts)
    seen["hazards"] = dict(hazards)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = _book_box(rgb.shape)
    book = np.zeros((h, w), bool)
    book[y1:y2, x1:x2] = True
    products = [pp.Instance(book, 0.92, prompts[0])]
    hands = []
    if seen.get("with_hand"):
        hand = np.zeros((h, w), bool)
        hand[int(h * 0.467):int(h * 0.633), 0:int(w * 0.367)] = True
        hands = [pp.Instance(hand, 0.8, "hand")]
    return pp.Scene(products=products, hands=hands), {"t_sam3": 0.1, "t_vision": 0.05, "dtype": "fp16",
                                                        "instances": {prompts[0]: [{"score": 0.92, "box": [x1, y1, x2, y2]}]}}


server._base_alpha = fake_base
server._sam3_scene = fake_scene
server._M.update(loaded=True, device="cpu", gpu="cpu", base_default="birefnet",
                 bases={"birefnet": {"mode": "dynamic"}, "rmbg2": {"mode": "fixed1024"}}, sam3_dtype=None)

client = TestClient(server.app)
AUTH = {"Authorization": "Bearer tok"}


def _jpeg():
    img = np.full((H, W, 3), (190, 180, 160), np.uint8)
    img[150:1050, 150:750] = (40, 60, 120)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=90)
    return buf.getvalue()


JPEG = _jpeg()


def post(fields=None, token="tok"):
    return client.post("/", files={"file": ("p.jpg", JPEG, "image/jpeg")}, data=fields or {},
                       headers={"Authorization": "Bearer " + token})


def test_health_and_categories():
    h = client.get("/health").json()
    assert h["ok"] and h["version"] == "v3" and h["bases"] == ["birefnet", "rmbg2"]
    c = client.get("/categories").json()
    assert c["default"] == "book"
    assert {"book", "bag", "backpack", "clothing", "shoe"} <= set(c["categories"])
    assert c["categories"]["shoe"]["prompts"] == ["shoe", "sneaker"]
    assert c["categories"]["book"]["params"]["bite_deficit"] == 0.05
    assert c["hazards"]["support"] == ["book stand", "easel", "shelf"] and c["hazards"]["hand"] == ["hand"]


def test_auth():
    assert post(token="nope").status_code == 401
    r = client.post("/", files={"file": ("p.jpg", JPEG, "image/jpeg")})
    assert r.status_code == 401


def test_ok_png_with_headers():
    seen.clear()
    r = post({"kind": "cara_plana"})
    assert r.status_code == 200 and r.content.startswith(b"\x89PNG"), r.text[:200]
    img = Image.open(io.BytesIO(r.content))
    assert img.mode == "RGBA" and img.size == (W, H)
    h = r.headers
    assert h["x-cutout-verdict"] == "ok" and h["x-cutout-model"] == "birefnet"
    assert h["x-cutout-category"] == "book" and h["x-cutout-kind"] == "cara_plana"
    assert h["x-cutout-products"] == "1" and h["x-cutout-matched"] == "1"
    assert "x-cutout-ms" in h and h["x-cutout-sam3-dtype"] == "fp16"
    assert seen["prompts"] == ["book"] and seen["bases"] == ["birefnet"]
    a = np.asarray(img.getchannel("A"))
    assert a[600, 450] == 255 and a[50, 50] == 0


def test_model_selection_and_clip():
    seen.clear()
    r = post({"model": "rmbg2"})
    assert r.status_code == 200 and r.headers["x-cutout-model"] == "rmbg2"
    a = np.asarray(Image.open(io.BytesIO(r.content)).getchannel("A"))
    assert a[1100, 450] == 0, "la repisa fuera del libro se recorta"
    assert "clipped" in r.headers["x-cutout-fixes"]
    seen.clear()
    r = post({"model": "inexistente"})
    assert r.status_code == 200 and r.headers["x-cutout-model"] == "birefnet"


def test_category_prompts():
    for cat, first in (("bag", "handbag"), ("backpack", "backpack"), ("shoe", "shoe"), ("clothing", "garment")):
        seen.clear()
        r = post({"category": cat})
        assert r.status_code == 200 and r.headers["x-cutout-category"] == cat, (cat, r.status_code)
        assert seen["prompts"][0] == first, seen["prompts"]
    seen.clear()
    r = post({"category": "desconocida"})
    assert r.headers["x-cutout-category"] == "book"


def test_reject_422_with_reason():
    seen.clear()
    seen["with_hand"] = True
    try:
        r = post({})
    finally:
        seen["with_hand"] = False
    assert r.status_code == 422, r.text[:200]
    body = r.json()
    assert body["error"] == "rejected" and body["reason"] == "hand_occlusion", body
    assert r.headers["x-cutout-verdict"] == "reject" and r.headers["x-cutout-reason"] == "hand_occlusion"
    assert r.headers["x-cutout-hands"] == "1"


def test_debug_returns_all_models():
    seen.clear()
    r = post({"debug": "1"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    d = r.json()
    assert sorted(d["models"]) == ["birefnet", "rmbg2"] and d["model_default"] == "birefnet"
    assert d["size"] == [W, H] and "instances" in d["sam3"]
    for name, m in d["models"].items():
        assert m["verdict"] == "ok", (name, m)
        for key in ("base_png", "final_png"):
            png = Image.open(io.BytesIO(base64.b64decode(m[key])))
            assert png.size == (W, H), (name, key, png.size)
    union = Image.open(io.BytesIO(base64.b64decode(d["union_png"])))
    assert union.size == (W, H)
    assert sorted(seen["bases"]) == ["birefnet", "rmbg2"]


def test_warmup_runs_with_fakes():
    # El calentamiento usa los mismos caminos que una petición real (falló en CI por una tupla).
    seen.clear()
    import torch
    server._M["torch"] = torch
    server._warmup()
    assert server._M.get("warmup_products") == 1 and sorted(seen["bases"]) == ["birefnet", "rmbg2"]


def test_sam3_off_returns_base_alpha():
    # CUTOUT_SAM3=0: sustituto exacto de Bria: alfa base tal cual, veredicto ok, sin análisis.
    saved = server._M.get("sam3")
    server._M["sam3"] = None
    seen.clear()
    seen["with_hand"] = True                               # con SAM 3 se rechazaría; apagado, no
    try:
        r = post({"model": "rmbg2"})
    finally:
        server._M["sam3"] = saved
        seen["with_hand"] = False
    assert r.status_code == 200 and r.headers["x-cutout-verdict"] == "ok"
    assert r.headers["x-cutout-sam3-dtype"] == "off"
    a = np.asarray(Image.open(io.BytesIO(r.content)).getchannel("A"))
    assert a[1100, 450] == 255, "sin SAM 3 la repisa NO se recorta (como Bria)"
    assert "prompts" not in seen, "no se llama a SAM 3"


def test_debug_disabled_without_flag():
    server.ALLOW_DEBUG = False
    try:
        r = post({"debug": "1"})
        assert r.status_code == 200 and r.content.startswith(b"\x89PNG")
    finally:
        server.ALLOW_DEBUG = True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print("ok  ", t.__name__)
        except AssertionError as exc:
            failed += 1
            print("FAIL", t.__name__, exc)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print("ERROR", t.__name__, type(exc).__name__, exc)
    print("{}/{} ok".format(len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
