"""Servidor de recorte GPU v3 para Alibaba Function Compute (custom container, puerto 9000).

Contrato (lo consume core/imggate/cutout_http.py):
    POST /             multipart `file` = JPEG; campos opcionales `category` (book|bag|
                       backpack|clothing|...), `kind` (cara_plana|objeto_sobre_fondo, lo que
                       dice el core), `model` (birefnet|rmbg2|ben2) y `debug=1`.
                       Authorization: Bearer <token>
        -> 200 image/png  RGBA del mismo tamaño, RGB intacto (sólo se calcula el alfa);
                          cabeceras X-Cutout-* con el veredicto y lo que se hizo.
        -> 422 JSON {"error":"rejected","reason":...}  la foto NO sirve (producto cortado
                          por el encuadre, mano que lo tapa, sin producto, mordida). Las
                          mismas cabeceras X-Cutout-*. Un core antiguo lo trata como fallo
                          permanente y no publica nada: fail-closed.
        -> 200 JSON (debug=1, sólo con CUTOUT_ALLOW_DEBUG=1): alfas intermedios de todos los
                          modelos base cargados, máscaras de SAM 3 y veredicto por modelo.
        -> 4xx/5xx JSON {"error": "..."}
    POST /initialize   hook de FC: carga modelos y calienta la GPU
    GET  /health       estado (sin token); GET /categories  configuración activa

Pipeline por foto (ver postproc.analyze):
    1. Modelo BASE (BiRefNet_dynamic por defecto; RMBG-2.0 y BEN2 opcionales para comparar):
       alfa fino del primer plano.
    2. SAM 3 (texto): instancias del producto ("book", "handbag"...), manos y soportes, con
       puntuación de presencia. Un solo paso del codificador visual, un prompt por concepto.
    3. Cruce: rechazo (producto cortado, mano, sin producto, mordido) o arreglo (fondo/atril
       fuera del producto, huecos y zonas translúcidas en caras planas, mordiscos que SAM 3 ve
       enteros, herrajes que la base perdió), con bordes añadidos refinados por el RGB.
El token no vive en la imagen ni en la config de FC: FC sólo guarda su SHA-256
(CUTOUT_TOKEN_SHA256). Sin esa variable, POST / responde 503 (fail-closed).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
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


def _env_json(name: str) -> Optional[dict]:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001
        log.error("%s no es un JSON válido (%s); se ignora", name, exc)
        return None


MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
BIREFNET_MAX_SIDE = int(_env_float("BIREFNET_MAX_SIDE", 1536))
SAM3_THRESHOLD = _env_float("SAM3_THRESHOLD", 0.2)       # umbral de recogida; cada papel aplica el suyo
SAM3_IMAGE_SIZE = int(_env_float("SAM3_IMAGE_SIZE", 1008))  # 1008 = nativo; menos sólo para CI en CPU
SAM3_DTYPE = (os.environ.get("SAM3_DTYPE", "auto") or "auto").strip().lower()  # auto|fp16|fp32
ALLOW_DEBUG = os.environ.get("CUTOUT_ALLOW_DEBUG", "") == "1"
MAX_UPLOAD = 20 * 1024 * 1024

# Modelos base disponibles (se cargan los que existan en MODELS_DIR y estén en
# CUTOUT_BASE_MODELS; por defecto todos los presentes). El de producción se elige con
# CUTOUT_BASE_MODEL; cada petición puede pedir otro con `model=`.
BASE_SPECS: Dict[str, Dict[str, str]] = {
    "birefnet": {"dir": "birefnet", "mode": "dynamic"},     # ZhengPeng7/BiRefNet_dynamic, resolución nativa
    "rmbg2": {"dir": "rmbg2", "mode": "fixed1024"},         # briaai/RMBG-2.0 (misma red), 1024x1024
    "ben2": {"dir": "ben2", "mode": "ben2"},                # PramaLLC/BEN2 base, 1024x1024
}
BASE_MODEL_DEFAULT = (os.environ.get("CUTOUT_BASE_MODEL", "birefnet") or "birefnet").strip().lower()
BASE_MODELS_WANTED = [m.strip().lower() for m in (os.environ.get("CUTOUT_BASE_MODELS", "") or "").split(",") if m.strip()]

# Categorías: prompts de SAM 3 para el producto, si es una cara plana (libros) y umbrales
# (`params`, cualquier campo de postproc.Params). Se amplían/cambian sin tocar código con
# CUTOUT_CATEGORIES (JSON), p.ej.
#   {"shoe": {"prompts": ["shoe", "sneaker"]},
#    "book": {"params": {"bite_deficit": 0.05}}}
DEFAULT_CATEGORIES: Dict[str, Dict[str, object]] = {
    "book": {"prompts": ["book"], "flat": True, "params": {}},
    "bag": {"prompts": ["handbag", "bag", "strap"], "flat": False, "params": {}},
    "backpack": {"prompts": ["backpack", "strap"], "flat": False, "params": {}},
    "clothing": {"prompts": ["garment", "clothing"], "flat": False, "params": {}},
}
# Peligros que SAM 3 busca en todas las fotos: una mano que sujeta el producto (rechazo) y
# soportes (se quitan si quedan fuera del producto). CUTOUT_HAZARDS (JSON) los cambia.
DEFAULT_HAZARDS: Dict[str, List[str]] = {
    "hand": ["hand"],
    "support": ["book stand", "easel", "display stand"],
}


def _load_categories() -> Dict[str, Dict[str, object]]:
    cats = {k: {"prompts": list(v["prompts"]), "flat": bool(v["flat"]), "params": dict(v.get("params") or {})}
            for k, v in DEFAULT_CATEGORIES.items()}
    extra = _env_json("CUTOUT_CATEGORIES") or {}
    for name, conf in extra.items():
        key = str(name).strip().lower()
        merged = cats.get(key, {"prompts": [key], "flat": False, "params": {}})
        conf = conf or {}
        if "prompts" in conf:
            prompts = conf["prompts"]
            merged["prompts"] = [str(p) for p in (prompts if isinstance(prompts, list) else [prompts])]
        if "flat" in conf:
            merged["flat"] = bool(conf["flat"])
        if isinstance(conf.get("params"), dict):
            merged["params"] = {**merged.get("params", {}), **conf["params"]}
        cats[key] = merged
    return cats


def _load_hazards() -> Dict[str, List[str]]:
    haz = {k: list(v) for k, v in DEFAULT_HAZARDS.items()}
    extra = _env_json("CUTOUT_HAZARDS") or {}
    for role, prompts in extra.items():
        if role in haz:
            haz[role] = [str(p) for p in (prompts if isinstance(prompts, list) else [prompts])]
    return haz


CATEGORIES = _load_categories()
HAZARDS = _load_hazards()
DEFAULT_CATEGORY = (os.environ.get("CUTOUT_CATEGORY", "book") or "book").strip().lower()

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
_M: Dict[str, object] = {"loaded": False, "bases": {}}
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


# --------------------------------------------------------------------------- carga
def _import_dir_module(path: str, pkg_name: str, module: str):
    import importlib
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [path]
    sys.modules[pkg_name] = pkg
    return importlib.import_module("{}.{}".format(pkg_name, module))


def _load_birefnet(path: str, device: str, dtype):
    """AutoModel con trust_remote_code; si la versión de transformers lo rompe, carga
    el módulo a mano y el state_dict de safetensors (mismo resultado)."""
    import torch
    try:
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(path, trust_remote_code=True)
        how = "automodel"
    except Exception as exc:  # noqa: BLE001
        log.warning("BiRefNet AutoModel falló (%s: %s); carga manual", type(exc).__name__, exc)
        from safetensors.torch import load_file
        pkg = "birefnet_pkg_" + hashlib.md5(path.encode()).hexdigest()[:8]
        cfg_mod = _import_dir_module(path, pkg, "BiRefNet_config")
        mod = _import_dir_module(path, pkg, "birefnet")
        model = mod.BiRefNet(config=cfg_mod.BiRefNetConfig(bb_pretrained=False))
        missing, _unexpected = model.load_state_dict(load_file(os.path.join(path, "model.safetensors")), strict=False)
        if missing:
            raise RuntimeError("BiRefNet: faltan {} pesos (p.ej. {})".format(len(missing), missing[:3]))
        how = "manual"
    model.to(device).eval()
    model.to(dtype if dtype is not None else torch.float32)
    return model, how


def _load_ben2(path: str, device: str, dtype):
    import torch
    from safetensors.torch import load_file
    mod = _import_dir_module(path, "ben2_pkg", "BEN2")
    model = mod.BEN_Base()
    state = load_file(os.path.join(path, "model.safetensors"))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model.to(dtype if dtype is not None else torch.float32)
    return model, "manual"


def _load_sam3(path: str, device: str):
    import torch
    from transformers import Sam3Config, Sam3Model, Sam3Processor
    kwargs = {}
    if SAM3_IMAGE_SIZE != 1008:
        cfg = Sam3Config.from_pretrained(path)
        cfg.image_size = SAM3_IMAGE_SIZE
        kwargs["config"] = cfg
        proc = Sam3Processor.from_pretrained(path, size={"height": SAM3_IMAGE_SIZE, "width": SAM3_IMAGE_SIZE})
    else:
        proc = Sam3Processor.from_pretrained(path)
    model = Sam3Model.from_pretrained(path, **kwargs)
    model.to(device).eval()
    if device != "cuda" or SAM3_DTYPE == "fp32":
        model.float()
        dtype = torch.float32
    else:
        model.half()
        dtype = torch.float16
    return model, proc, dtype


def load_models() -> None:
    if _M["loaded"]:
        return
    with _load_lock:
        if _M["loaded"]:
            return
        t0 = time.time()
        import torch
        torch.backends.cudnn.benchmark = False   # tamaños variables: no re-tunear cada foto
        cuda = torch.cuda.is_available()
        device = "cuda" if cuda else "cpu"
        half = torch.float16 if cuda else None
        bases: Dict[str, Dict[str, object]] = {}
        for name, spec in BASE_SPECS.items():
            if BASE_MODELS_WANTED and name not in BASE_MODELS_WANTED:
                continue
            path = os.path.join(MODELS_DIR, spec["dir"])
            if not os.path.isdir(path):
                continue
            loader = _load_ben2 if spec["mode"] == "ben2" else _load_birefnet
            model, how = loader(path, device, half)
            bases[name] = {"model": model, "mode": spec["mode"], "loader": how}
            log.info("modelo base %s cargado (%s, %s)", name, spec["mode"], how)
        if not bases:
            raise RuntimeError("ningún modelo base en {} ({})".format(MODELS_DIR, sorted(BASE_SPECS)))
        default = BASE_MODEL_DEFAULT if BASE_MODEL_DEFAULT in bases else sorted(bases)[0]
        sam3_dir = os.path.join(MODELS_DIR, "sam3")
        if not os.path.isdir(sam3_dir):
            raise RuntimeError("falta SAM 3 en {}: el servidor v3 no funciona sin él".format(sam3_dir))
        sam3, sam3_proc, sam3_dtype = _load_sam3(sam3_dir, device)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        _M.update(torch=torch, device=device, half=half, bases=bases, base_default=default,
                  sam3=sam3, sam3_proc=sam3_proc, sam3_dtype=sam3_dtype, mean=mean, std=std,
                  gpu=(torch.cuda.get_device_name(0) if cuda else "cpu"))
        t1 = time.time()
        _warmup()
        _M["load_s"] = round(t1 - t0, 2)
        _M["warmup_s"] = round(time.time() - t1, 2)
        _M["loaded"] = True
        log.info("modelos listos en %s (%s): bases=%s por defecto=%s sam3=%s; carga %.1fs, calentamiento %.1fs",
                 device, _M["gpu"], sorted(bases), default, _sam3_dtype_name(), t1 - t0, _M["warmup_s"])


def _sam3_dtype_name() -> str:
    dt = _M.get("sam3_dtype")
    return "fp16" if dt is not None and str(dt).endswith("float16") else "fp32"


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
    """Calienta la GPU y decide la precisión de SAM 3: si en fp16 devuelve valores no finitos
    o no ve el libro sintético que sí ve en fp32, se queda en fp32 (la T4 no tiene bf16)."""
    rgb = _synthetic_photo()
    torch = _M["torch"]
    try:
        scene, sinfo = _sam3_scene(rgb, ["book"], HAZARDS)
        finite = all(np.isfinite(p.score) for p in scene.products)
        reliable = finite and bool(scene.products)
        why = "finito={} libros={}".format(finite, len(scene.products))
    except Exception as exc:  # noqa: BLE001
        reliable = False
        why = "{}: {}".format(type(exc).__name__, str(exc)[:160])
        scene = pp.Scene()
    if _M["device"] == "cuda" and _sam3_dtype_name() == "fp16" and not reliable:
        if SAM3_DTYPE == "fp16":
            raise RuntimeError("SAM 3 en fp16 no es fiable ({}) y SAM3_DTYPE lo fuerza".format(why))
        log.warning("SAM 3 en fp16 no es fiable (%s); se pasa a fp32", why)
        _M["sam3"].float()
        _M["sam3_dtype"] = torch.float32
        scene, sinfo = _sam3_scene(rgb, ["book"], HAZARDS)
    _M["warmup_products"] = len(scene.products)
    for name in _M["bases"]:
        alpha = _base_alpha(name, rgb)
        pp.analyze(alpha, rgb, scene, flat=True)


# --------------------------------------------------------------------------- modelos
def _base_alpha(name: str, rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    torch = _M["torch"]
    import torch.nn.functional as F
    entry = _M["bases"][name]
    model, mode = entry["model"], entry["mode"]
    h, w = rgb.shape[:2]
    if mode == "dynamic":
        tw, th = pp.work_size(w, h, max_side=BIREFNET_MAX_SIDE)
    else:
        tw, th = 1024, 1024
    small = np.array(Image.fromarray(rgb).resize((tw, th), Image.Resampling.BICUBIC))
    x = torch.from_numpy(small).to(_M["device"]).permute(2, 0, 1).float().div_(255.0)
    x = ((x - _M["mean"]) / _M["std"]).unsqueeze(0)
    if _M["half"] is not None:
        x = x.to(_M["half"])
    with torch.inference_mode():
        if mode == "ben2":
            pred = model(x).float()                       # [1,1,1024,1024] ya con sigmoide
            lo, hi = pred.min(), pred.max()
            pred = (pred - lo) / torch.clamp(hi - lo, min=1e-6)   # como BEN2.postprocess_image
        else:
            pred = model(x)[-1].float().sigmoid()
        pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        alpha = pred[0, 0].clamp_(0, 1).mul_(255.0).round_().to(torch.uint8).cpu().numpy()
    return alpha, {"work": "{}x{}".format(tw, th)}


def _dedupe(instances: List[pp.Instance], iou_min: float = 0.8) -> List[pp.Instance]:
    kept: List[pp.Instance] = []
    for inst in sorted(instances, key=lambda i: -i.score):
        dup = False
        for k in kept:
            inter = float((inst.mask & k.mask).sum())
            union = float((inst.mask | k.mask).sum())
            if union > 0 and inter / union >= iou_min:
                dup = True
                break
        if not dup:
            kept.append(inst)
    return kept


def _sam3_scene(rgb: np.ndarray, product_prompts: List[str],
                hazards: Dict[str, List[str]]) -> Tuple[pp.Scene, Dict[str, object]]:
    """Un paso del codificador visual de SAM 3 y un prompt por concepto (producto, mano,
    soporte). Devuelve las instancias a resolución de la foto."""
    torch = _M["torch"]
    model, proc = _M["sam3"], _M["sam3_proc"]
    h, w = rgb.shape[:2]
    device = _M["device"]
    use_fp16 = device == "cuda" and _sam3_dtype_name() == "fp16"
    roles: List[Tuple[str, str]] = [("product", p) for p in product_prompts]
    roles += [(role, p) for role, prompts in hazards.items() for p in prompts]
    info: Dict[str, object] = {"instances": {}}
    scene = pp.Scene()
    t = time.time()
    with torch.inference_mode():
        img_inputs = proc(images=Image.fromarray(rgb), return_tensors="pt").to(device)
        pixel_values = img_inputs["pixel_values"]
        if use_fp16:
            pixel_values = pixel_values.half()
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
            vision = model.get_vision_features(pixel_values=pixel_values)
            info["t_vision"] = round(time.time() - t, 3)
            for role, prompt in roles:
                text_inputs = proc(text=prompt, return_tensors="pt").to(device)
                outputs = model(vision_embeds=vision, **text_inputs)
                res = proc.post_process_instance_segmentation(
                    outputs, threshold=SAM3_THRESHOLD, mask_threshold=0.5, target_sizes=[(h, w)])[0]
                masks = res["masks"]
                scores = res["scores"]
                found = []
                for i in range(len(scores)):
                    m = masks[i]
                    while m.ndim > 2:
                        m = m[0]
                    mask = m.detach().to("cpu").numpy().astype(bool)
                    score = float(scores[i])
                    if mask.shape != (h, w):
                        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.Resampling.NEAREST)) > 127
                    inst = pp.Instance(mask=mask, score=score, label=prompt)
                    found.append(inst)
                    bb = inst.box
                    info["instances"].setdefault(prompt, []).append(
                        {"score": round(score, 3), "box": [round(v) for v in bb] if bb else None})
                target = {"product": scene.products, "hand": scene.hands, "support": scene.supports}.get(role)
                if target is not None:
                    target.extend(found)
    scene.products = _dedupe(scene.products)
    scene.hands = _dedupe(scene.hands)
    scene.supports = _dedupe(scene.supports)
    info["t_sam3"] = round(time.time() - t, 3)
    info["dtype"] = _sam3_dtype_name()
    return scene, info


# --------------------------------------------------------------------------- pipeline
def _category(raw: Optional[str]) -> str:
    value = (raw or "").strip().lower()
    return value if value in CATEGORIES else (
        DEFAULT_CATEGORY if DEFAULT_CATEGORY in CATEGORIES else "book")


def _mask_png_b64(mask: np.ndarray) -> str:
    import cv2
    arr = mask if mask.dtype == np.uint8 else (mask.astype(np.uint8) * 255)
    ok, buf = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _process(rgb: np.ndarray, *, category: Optional[str], kind: Optional[str],
             model: Optional[str], all_models: bool = False) -> Dict[str, object]:
    cat = _category(category)
    conf = CATEGORIES[cat]
    params = pp.Params.from_dict(conf.get("params"))
    flat = bool(conf.get("flat"))
    kind = (kind or "").strip().lower() or None
    want = (model or "").strip().lower()
    chosen = want if want in _M["bases"] else str(_M["base_default"])
    names = sorted(_M["bases"]) if all_models else [chosen]

    scene, sinfo = _sam3_scene(rgb, list(conf["prompts"]), HAZARDS)
    out: Dict[str, object] = {"category": cat, "kind": kind, "flat": flat, "model": chosen,
                              "sam3": sinfo, "models": {}}
    for name in names:
        t = time.time()
        alpha, binfo = _base_alpha(name, rgb)
        t_base = round(time.time() - t, 3)
        t = time.time()
        res = pp.analyze(alpha, rgb, scene, flat=flat, kind=kind, params=params)
        out["models"][name] = {"base_alpha": alpha, "result": res, "t_base": t_base,
                               "t_analyze": round(time.time() - t, 3), "work": binfo["work"]}
    out["scene"] = scene
    return out


def _headers(proc: Dict[str, object], name: str) -> Dict[str, str]:
    entry = proc["models"][name]
    res: pp.Result = entry["result"]
    info = res.info
    fixes = [k for k in ("holes_filled", "solidified", "bite_filled", "snapped", "support_removed",
                         "recovered_parts", "dropped_components", "clipped", "rescued") if info.get(k)]
    frame = info.get("frame") or {}
    h = {
        "X-Cutout-Verdict": "ok" if res.ok else "reject",
        "X-Cutout-Reason": res.reason,
        "X-Cutout-Model": name,
        "X-Cutout-Category": str(proc["category"]),
        "X-Cutout-Kind": str(proc.get("kind") or ""),
        "X-Cutout-Products": str(info.get("products", 0)),
        "X-Cutout-Matched": str(info.get("matched", 0)),
        "X-Cutout-Product-Score": str(info.get("product_score", 0)),
        "X-Cutout-Hands": str(info.get("hands", 0)),
        "X-Cutout-Supports": str(info.get("supports", 0)),
        "X-Cutout-Frame": ",".join(frame.get("sides", [])) if frame else "",
        "X-Cutout-Cover": str(info.get("cover", "")),
        "X-Cutout-Fill": str(info.get("fill", "")),
        "X-Cutout-Deficit": str(info.get("deficit", "")),
        "X-Cutout-Sam-Deficit": str(info.get("sam_deficit", "")),
        "X-Cutout-Fixes": ",".join(fixes),
        "X-Cutout-Nonconvex": "1" if info.get("nonconvex_accepted") else "",
        "X-Cutout-Work": str(entry["work"]),
        "X-Cutout-T-Base": str(entry["t_base"]),
        "X-Cutout-T-Sam3": str(proc["sam3"].get("t_sam3", "")),
        "X-Cutout-Sam3-Dtype": str(proc["sam3"].get("dtype", "")),
    }
    return {k: v for k, v in h.items() if v != ""}


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
            "bases": sorted(_M.get("bases", {})), "base_default": _M.get("base_default"),
            "sam3_dtype": _sam3_dtype_name(), "warmup_products": _M.get("warmup_products")}


@app.get("/health")
def health():
    return {"ok": True, "loaded": bool(_M.get("loaded")), "device": _M.get("device"), "gpu": _M.get("gpu"),
            "bases": sorted(_M.get("bases", {})), "base_default": _M.get("base_default"),
            "sam3_dtype": _sam3_dtype_name() if _M.get("loaded") else None, "version": "v3"}


@app.get("/")
def root():
    return health()


@app.get("/categories")
def categories():
    """Categorías activas, peligros y umbrales por defecto (sin secretos)."""
    return {"default": _category(None), "categories": CATEGORIES, "hazards": HAZARDS,
            "params_default": pp.Params().__dict__, "base_models": sorted(_M.get("bases", {})) or sorted(BASE_SPECS)}


def _cutout(file_bytes: bytes, *, category: Optional[str], kind: Optional[str], model: Optional[str],
            debug: bool) -> Response:
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
            proc = _process(rgb, category=category, kind=kind, model=model, all_models=debug)
    except Exception as exc:  # noqa: BLE001
        log.exception("fallo procesando la foto")
        return JSONResponse({"error": "processing_failed: {}".format(type(exc).__name__)}, status_code=500)
    ms = int((time.time() - t0) * 1000)
    chosen = str(proc["model"])
    headers = _headers(proc, chosen)
    headers["X-Cutout-Ms"] = str(ms)
    res: pp.Result = proc["models"][chosen]["result"]
    log.info("recorte %dx%d cat=%s model=%s verdict=%s/%s %s", rgb.shape[1], rgb.shape[0], proc["category"],
             chosen, "ok" if res.ok else "reject", res.reason, {k: v for k, v in res.info.items() if k != "frame"})
    if debug:
        scene: pp.Scene = proc["scene"]
        h, w = rgb.shape[:2]
        body = {
            "category": proc["category"], "kind": proc.get("kind"), "flat": proc["flat"], "size": [w, h],
            "model_default": chosen, "ms": ms,
            "sam3": {k: v for k, v in proc["sam3"].items()},
            "union_png": _mask_png_b64(pp.union_of(scene.products, (h, w))),
            "hands_png": _mask_png_b64(pp.union_of(scene.hands, (h, w))),
            "supports_png": _mask_png_b64(pp.union_of(scene.supports, (h, w))),
            "models": {},
        }
        for name, entry in proc["models"].items():
            r: pp.Result = entry["result"]
            body["models"][name] = {
                "verdict": "ok" if r.ok else "reject", "reason": r.reason, "info": r.info,
                "t_base": entry["t_base"], "t_analyze": entry["t_analyze"], "work": entry["work"],
                "base_png": _mask_png_b64(entry["base_alpha"]), "final_png": _mask_png_b64(r.alpha),
            }
        return JSONResponse(body, headers=headers)
    if not res.ok:
        return JSONResponse({"error": "rejected", "reason": res.reason,
                             "info": {k: v for k, v in res.info.items()}}, status_code=422, headers=headers)
    try:
        png = _encode_png(rgb, res.alpha)
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"error": "processing_failed: {}".format(type(exc).__name__)}, status_code=500)
    return Response(content=png, media_type="image/png", headers=headers)


@app.post("/")
def cutout(file: UploadFile = File(...), kind: Optional[str] = Form(None),
           category: Optional[str] = Form(None), model: Optional[str] = Form(None),
           debug: Optional[str] = Form(None), authorization: Optional[str] = Header(None)):
    denied = _auth_ok(authorization)
    if denied is not None:
        return denied
    want_debug = ALLOW_DEBUG and (debug or "") == "1"
    return _cutout(file.file.read(), category=category, kind=kind, model=model, debug=want_debug)
