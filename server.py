"""Servidor de recorte GPU para Alibaba Function Compute (custom container, puerto 9000).

Contrato (lo consume core/imggate/cutout_http.py, sin cambios en el core):
    POST /             multipart `file` = JPEG (opcional `kind`)
                       Authorization: Bearer <token>
        -> 200 image/png  RGBA del mismo tamaño, RGB intacto (sólo se calcula el alfa)
           cabeceras X-Cutout-* con métricas y lo que se hizo
        -> 4xx/5xx JSON {"error": "..."}
    POST /initialize   hook de inicialización de FC: carga modelos y calienta la GPU
    GET  /health       estado (sin token)

Pipeline por foto:
    1. BiRefNet_dynamic (MIT) a resolución nativa (lado largo <= 1536, múltiplos de 32).
    2. Grounding DINO tiny (Apache-2.0): caja del producto. Lo que la máscara deje
       fuera de la caja (atril, franja de mesa) se pone a 0 si el detector está seguro.
    3. Si la máscara sale mordida (déficit de casco > 0,035) o despiezada (>2 trozos),
       SAM 2.1 large (Apache-2.0) con la caja rellena lo que falta DENTRO del casco del
       producto, y sólo si SAM ve una forma convexa y el resultado mejora.
El token no vive en la imagen ni en la config de FC: FC sólo guarda su SHA-256
(CUTOUT_TOKEN_SHA256). Sin esa variable, POST / responde 503 (fail-closed).
"""
from __future__ import annotations

import hashlib
import hmac
import io
import logging
import os
import sys
import threading
import time
import types
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import postproc as pp  # noqa: E402

from fastapi import FastAPI, File, Form, Header, UploadFile  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("cutout")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
BIREFNET_MAX_SIDE = int(_env_float("BIREFNET_MAX_SIDE", 1536))
DINO_PROMPT = os.environ.get("DINO_PROMPT", "book. clothing.")
DINO_THRESHOLD = _env_float("DINO_THRESHOLD", 0.30)
DINO_TEXT_THRESHOLD = _env_float("DINO_TEXT_THRESHOLD", 0.25)
CLIP_MIN_COVERAGE = _env_float("CLIP_MIN_COVERAGE", 0.85)
CLIP_MIN_SCORE = _env_float("CLIP_MIN_SCORE", 0.35)
RESCUE_DEFICIT = _env_float("RESCUE_DEFICIT", 0.035)
SAM_MAX_DEFICIT = _env_float("SAM_MAX_DEFICIT", 0.03)
ALLOW_DEBUG = os.environ.get("CUTOUT_ALLOW_DEBUG", "") == "1"
MAX_UPLOAD = 20 * 1024 * 1024

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_M: Dict[str, object] = {"loaded": False}
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


# --------------------------------------------------------------------------- carga
def _load_birefnet(path: str, device: str, dtype):
    """AutoModel con trust_remote_code; si la versión de transformers lo rompe, carga
    el módulo a mano y el state_dict de safetensors (mismo resultado)."""
    import torch  # noqa: F401
    try:
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True)
        how = "automodel"
    except Exception as exc:  # noqa: BLE001
        log.warning("BiRefNet AutoModel falló (%s: %s); carga manual", type(exc).__name__, exc)
        import importlib
        from safetensors.torch import load_file
        pkg = types.ModuleType("birefnet_pkg")
        pkg.__path__ = [path]
        sys.modules["birefnet_pkg"] = pkg
        cfg_mod = importlib.import_module("birefnet_pkg.BiRefNet_config")
        mod = importlib.import_module("birefnet_pkg.birefnet")
        model = mod.BiRefNet(config=cfg_mod.BiRefNetConfig(bb_pretrained=False))
        missing, unexpected = model.load_state_dict(
            load_file(os.path.join(path, "model.safetensors")), strict=False)
        if missing:
            raise RuntimeError("BiRefNet: faltan {} pesos (p.ej. {})".format(len(missing), missing[:3]))
        how = "manual"
    model.to(device).eval()
    if dtype is not None:
        model.to(dtype)
    return model, how


def load_models() -> None:
    if _M["loaded"]:
        return
    with _load_lock:
        if _M["loaded"]:
            return
        t0 = time.time()
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  Sam2Model, Sam2Processor)
        torch.backends.cudnn.benchmark = False   # tamaños variables: no re-tunear cada foto
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        half = torch.float16 if cuda else None
        birefnet, how = _load_birefnet(os.path.join(MODELS_DIR, "birefnet"), device, half)
        dino_proc = AutoProcessor.from_pretrained(os.path.join(MODELS_DIR, "gdino"))
        dino = AutoModelForZeroShotObjectDetection.from_pretrained(
            os.path.join(MODELS_DIR, "gdino")).to(device).eval()
        sam_proc = Sam2Processor.from_pretrained(os.path.join(MODELS_DIR, "sam2"))
        sam = Sam2Model.from_pretrained(os.path.join(MODELS_DIR, "sam2")).to(device).eval()
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        _M.update(torch=torch, device=device, half=half, birefnet=birefnet,
                  birefnet_loader=how, dino_proc=dino_proc, dino=dino,
                  sam_proc=sam_proc, sam=sam, mean=mean, std=std,
                  gpu=(torch.cuda.get_device_name(0) if cuda else "cpu"))
        t1 = time.time()
        _warmup()
        _M["load_s"] = round(t1 - t0, 2)
        _M["warmup_s"] = round(time.time() - t1, 2)
        _M["loaded"] = True
        log.info("modelos listos en %s (%s): carga %.1fs, calentamiento %.1fs, birefnet=%s",
                 device, _M["gpu"], t1 - t0, _M["warmup_s"], how)


def _synthetic_photo(w: int = 960, h: int = 1280) -> np.ndarray:
    """Foto falsa: contraportada pálida con código de barras sobre una mesa clara."""
    rng = np.random.default_rng(7)
    img = np.full((h, w, 3), 200, np.uint8) + rng.integers(0, 25, (h, w, 1), dtype=np.uint8)
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    d.rectangle([140, 120, w - 140, h - 60], fill=(236, 230, 214))
    for i in range(14):
        d.rectangle([200, 220 + 34 * i, w - 200, 232 + 34 * i], fill=(60, 60, 60))
    for i in range(40):
        x = w // 2 - 120 + 6 * i
        d.rectangle([x, h - 260, x + (2 if i % 3 else 4), h - 120], fill=(10, 10, 10))
    return np.asarray(pil)


def _warmup() -> None:
    rgb = _synthetic_photo()
    _process(rgb, force_rescue=True)


# --------------------------------------------------------------------------- modelos
def _birefnet_alpha(rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    torch = _M["torch"]
    import torch.nn.functional as F
    h, w = rgb.shape[:2]
    tw, th = pp.work_size(w, h, max_side=BIREFNET_MAX_SIDE)
    small = np.array(Image.fromarray(rgb).resize((tw, th), Image.Resampling.BICUBIC))
    x = torch.from_numpy(small).to(_M["device"]).permute(2, 0, 1).float().div_(255.0)
    x = ((x - _M["mean"]) / _M["std"]).unsqueeze(0)
    if _M["half"] is not None:
        x = x.to(_M["half"])
    with torch.inference_mode():
        pred = _M["birefnet"](x)[-1].float().sigmoid()
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        alpha = pred[0, 0].clamp_(0, 1).mul_(255.0).round_().to(torch.uint8).cpu().numpy()
    return alpha, (tw, th)


def _dino(rgb: np.ndarray) -> List[pp.Detection]:
    torch = _M["torch"]
    h, w = rgb.shape[:2]
    proc = _M["dino_proc"]
    inputs = proc(images=Image.fromarray(rgb), text=DINO_PROMPT, return_tensors="pt").to(_M["device"])
    with torch.inference_mode():
        out = _M["dino"](**inputs)
    kwargs = dict(text_threshold=DINO_TEXT_THRESHOLD, target_sizes=[(h, w)])
    try:
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=DINO_THRESHOLD, **kwargs)[0]
    except TypeError:   # transformers antiguos: box_threshold
        res = proc.post_process_grounded_object_detection(
            out, inputs.input_ids, box_threshold=DINO_THRESHOLD, **kwargs)[0]
    labels = res.get("text_labels", res.get("labels")) or []
    dets = []
    for i, (score, box) in enumerate(zip(res["scores"].tolist(), res["boxes"].tolist())):
        x1, y1, x2, y2 = (float(v) for v in box)
        lab = labels[i] if i < len(labels) and isinstance(labels[i], str) else ""
        dets.append(pp.Detection(float(score), (max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)), lab))
    return dets


def _sam(rgb: np.ndarray, box: pp.Box) -> np.ndarray:
    torch = _M["torch"]
    proc = _M["sam_proc"]
    inputs = proc(images=Image.fromarray(rgb), input_boxes=[[[float(v) for v in box]]],
                  return_tensors="pt").to(_M["device"])
    with torch.inference_mode():
        if _M["half"] is not None:
            with torch.autocast("cuda", dtype=torch.float16):
                out = _M["sam"](**inputs, multimask_output=False)
        else:
            out = _M["sam"](**inputs, multimask_output=False)
    masks = proc.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"].cpu())[0]
    m = masks
    while m.ndim > 2:
        m = m[0]
    return m.numpy().astype(bool)


# --------------------------------------------------------------------------- pipeline
def _process(rgb: np.ndarray, *, kind: Optional[str] = None,
             force_rescue: bool = False) -> Tuple[np.ndarray, Dict[str, object]]:
    info: Dict[str, object] = {}
    t = time.time()
    alpha, work = _birefnet_alpha(rgb)
    info["work"] = "{}x{}".format(*work)
    info["t_birefnet"] = round(time.time() - t, 3)

    t = time.time()
    dets = _dino(rgb)
    plan = pp.choose_boxes(alpha, dets, min_score=DINO_THRESHOLD)
    info["t_dino"] = round(time.time() - t, 3)
    if plan.main is not None:
        b = plan.main.box
        info["box"] = "{:.0f},{:.0f},{:.0f},{:.0f}@{:.2f}".format(b[0], b[1], b[2], b[3], plan.main.score)
    alpha, clip = pp.clip_outside(alpha, plan, min_coverage=CLIP_MIN_COVERAGE, min_main_score=CLIP_MIN_SCORE)
    info["clip"] = "removed={}".format(clip["removed"]) if clip.get("clipped") else str(clip.get("reason"))

    need, rinfo = pp.needs_rescue(alpha, kind=kind, deficit_trigger=RESCUE_DEFICIT)
    info["deficit_initial"] = rinfo["deficit"]
    info["components_initial"] = rinfo["components"]
    info["rescue"] = "none"
    if need or force_rescue:
        t = time.time()
        box = pp.rescue_box(alpha, plan.main)
        if box is None:
            info["rescue"] = "no_box"
        else:
            merged, minfo = pp.merge_rescue(alpha, _sam(rgb, box), box, sam_max_deficit=SAM_MAX_DEFICIT)
            if merged is not None:
                alpha = merged
                info["rescue"] = "sam"
            else:
                info["rescue"] = "rejected:{}".format(minfo.get("reject"))
        info["t_sam"] = round(time.time() - t, 3)
    info["deficit_final"] = pp.hull_deficit(alpha)
    info["components_final"] = len(pp.significant_components(alpha))
    return alpha, info


def _encode_png(rgb: np.ndarray, alpha: np.ndarray) -> bytes:
    import cv2
    bgra = np.dstack([rgb[:, :, ::-1], alpha])
    ok, buf = cv2.imencode(".png", bgra, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise RuntimeError("png_encode_failed")
    return buf.tobytes()


def _auth_ok(authorization: Optional[str]) -> Optional[JSONResponse]:
    expected = (os.environ.get("CUTOUT_TOKEN_SHA256") or "").strip().lower()
    if not expected:
        return JSONResponse({"error": "server_token_not_configured"}, status_code=503)
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse({"error": "missing_bearer"}, status_code=401)
    got = hashlib.sha256(authorization[7:].strip().encode("utf-8")).hexdigest()
    if not hmac.compare_digest(got, expected):
        return JSONResponse({"error": "invalid_token"}, status_code=401)
    return None


# --------------------------------------------------------------------------- rutas
@app.post("/initialize")
def initialize():
    load_models()
    return {"status": "initialized", "device": _M.get("device"), "gpu": _M.get("gpu"),
            "load_s": _M.get("load_s"), "warmup_s": _M.get("warmup_s"),
            "birefnet_loader": _M.get("birefnet_loader")}


@app.get("/health")
def health():
    return {"ok": True, "loaded": bool(_M.get("loaded")), "device": _M.get("device"),
            "gpu": _M.get("gpu")}


@app.get("/")
def root():
    return health()


def _cutout(file_bytes: bytes, kind: Optional[str], force_rescue: bool) -> Response:
    t0 = time.time()
    if not file_bytes:
        return JSONResponse({"error": "empty_file"}, status_code=400)
    if len(file_bytes) > MAX_UPLOAD:
        return JSONResponse({"error": "file_too_large"}, status_code=413)
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.load()
        rgb = np.asarray(img.convert("RGB"))
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": "invalid_image: {}".format(type(exc).__name__)}, status_code=400)
    if min(rgb.shape[:2]) < 16:
        return JSONResponse({"error": "image_too_small"}, status_code=400)
    try:
        load_models()
        with _infer_lock:
            alpha, info = _process(rgb, kind=kind, force_rescue=force_rescue)
        png = _encode_png(rgb, alpha)
    except Exception as exc:  # noqa: BLE001
        log.exception("fallo procesando la foto")
        return JSONResponse({"error": "processing_failed: {}".format(type(exc).__name__)}, status_code=500)
    info["ms"] = int((time.time() - t0) * 1000)
    headers = {"X-Cutout-" + k.replace("_", "-").title(): str(v) for k, v in info.items()}
    log.info("recorte %dx%d %s", rgb.shape[1], rgb.shape[0], info)
    return Response(content=png, media_type="image/png", headers=headers)


@app.post("/")
def cutout(file: UploadFile = File(...), kind: Optional[str] = Form(None),
           force_rescue: Optional[str] = Form(None), authorization: Optional[str] = Header(None)):
    denied = _auth_ok(authorization)
    if denied is not None:
        return denied
    force = ALLOW_DEBUG and (force_rescue or "") == "1"
    return _cutout(file.file.read(), (kind or "").strip() or None, force)
