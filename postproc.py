"""Lógica de máscara del servidor de recorte (numpy + OpenCV, sin torch).

Separada de `server.py` para poder probarla en local sin GPU ni modelos
(`python test_postproc.py`). Todas las máscaras son arrays HxW: `alpha` uint8
(0..255) o booleanos. Las cajas son (x1, y1, x2, y2) en píxeles de la imagen.

Medidas con las mismas reglas que la auditoría del 2026-10-25 (scratchpad
alpha/hull.py): déficit de casco = 1 - píxeles(alfa>=128) / píxeles(casco convexo),
calculado sobre TODOS los píxeles visibles con la máscara reducida a 400 px de lado.
Allí, portadas planas buenas <= 0,012; la contraportada mordida de Picoult 0,127;
libros abiertos/pilas 0,2-0,4 (no convexos por naturaleza).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SOLID = 128     # alfa >= SOLID cuenta como producto (mismo umbral que el déficit medido)
VISIBLE = 32    # alfa >= VISIBLE cuenta como visible (mismo umbral que inspect_cutout)

Box = Tuple[float, float, float, float]


# --------------------------------------------------------------------------- tamaños
def work_size(w: int, h: int, *, max_side: int = 1536, multiple: int = 32,
              min_side: int = 256) -> Tuple[int, int]:
    """Tamaño de trabajo para BiRefNet: lado largo <= max_side y múltiplos de 32.

    BiRefNet trocea las features en parches con `rearrange`, así que alto y ancho
    tienen que ser múltiplos de 32. Nunca agranda por encima del original salvo para
    llegar al mínimo de 256 con el que se entrenó.
    """
    scale = min(1.0, float(max_side) / float(max(w, h)))
    tw = int(round(w * scale / multiple)) * multiple
    th = int(round(h * scale / multiple)) * multiple
    return max(min_side, tw), max(min_side, th)


def _downscale(mask: np.ndarray, side: int) -> np.ndarray:
    h, w = mask.shape[:2]
    scale = min(1.0, float(side) / float(max(w, h)))
    if scale >= 1.0:
        return mask
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(mask, size, interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- métricas
def hull_deficit(alpha: np.ndarray, *, thr: int = SOLID, side: int = 400) -> float:
    """1 - área(alfa>=thr) / área(casco convexo). 0 = forma convexa sin huecos."""
    small = _downscale(alpha.astype(np.uint8), side)
    m = small >= thr
    ys, xs = np.nonzero(m)
    if len(xs) < 10:
        return 0.0
    pts = np.column_stack([xs, ys]).astype(np.int32)
    hull = cv2.convexHull(pts)
    canvas = np.zeros(m.shape, np.uint8)
    cv2.fillConvexPoly(canvas, hull, 1)
    hull_area = int(canvas.sum())
    if hull_area <= 0:
        return 0.0
    return round(max(0.0, 1.0 - float(m.sum()) / float(hull_area)), 4)


def significant_components(alpha: np.ndarray, *, thr: int = VISIBLE, side: int = 512,
                           min_ratio: float = 0.0025) -> List[Tuple[int, int, int, int, int]]:
    """Réplica de `bria._significant_components`: trozos 8-conexos >= 0,25 % del área.

    Devuelve (x0, y0, x1, y1, area) en coordenadas de la máscara reducida.
    """
    m = (alpha >= thr).astype(np.uint8) * 255
    h, w = m.shape
    scale = min(1.0, float(side) / float(max(w, h)))
    if scale < 1.0:
        # BOX + umbral >0, como el core: aproxima max-pooling.
        m = cv2.resize(m, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                       interpolation=cv2.INTER_AREA)
    m = (m > 0).astype(np.uint8)
    sh, sw = m.shape
    min_area = max(4, int(math.ceil(sw * sh * min_ratio)))
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area >= min_area:
            out.append((x, y, x + bw, y + bh, area))
    return out


def solid_bbox(alpha: np.ndarray, thr: int = SOLID) -> Optional[Box]:
    ys, xs = np.nonzero(alpha >= thr)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def expand_box(box: Box, frac: float, w: int, h: int, min_px: float = 0.0) -> Box:
    x1, y1, x2, y2 = box
    m = max(min_px, frac * max(x2 - x1, y2 - y1))
    return max(0.0, x1 - m), max(0.0, y1 - m), min(float(w), x2 + m), min(float(h), y2 + m)


def union_box(a: Box, b: Box) -> Box:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def box_iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _box_slices(box: Box, w: int, h: int) -> Tuple[slice, slice]:
    x1 = int(max(0, math.floor(box[0]))); y1 = int(max(0, math.floor(box[1])))
    x2 = int(min(w, math.ceil(box[2]))); y2 = int(min(h, math.ceil(box[3])))
    return slice(y1, max(y1, y2)), slice(x1, max(x1, x2))


def region_from_boxes(boxes: Sequence[Box], w: int, h: int) -> np.ndarray:
    region = np.zeros((h, w), bool)
    for b in boxes:
        sy, sx = _box_slices(b, w, h)
        region[sy, sx] = True
    return region


# --------------------------------------------------------------------------- decisiones
@dataclass
class Detection:
    score: float
    box: Box
    label: str = ""


@dataclass
class BoxPlan:
    main: Optional[Detection] = None
    keep: List[Detection] = field(default_factory=list)
    info: Dict[str, object] = field(default_factory=dict)


def choose_boxes(alpha: np.ndarray, detections: Sequence[Detection], *,
                 min_score: float = 0.30, min_fill: float = 0.45) -> BoxPlan:
    """Cajas del detector que corresponden al producto que ve BiRefNet.

    Se quedan las cajas con puntuación suficiente cuyo interior esté ocupado por la
    máscara (fill >= min_fill). La principal es la que contiene más máscara.
    """
    h, w = alpha.shape
    solid = alpha >= SOLID
    total = float(solid.sum())
    plan = BoxPlan()
    if total <= 0:
        plan.info["reason"] = "empty_mask"
        return plan
    best = -1.0
    for det in detections:
        if det.score < min_score:
            continue
        sy, sx = _box_slices(det.box, w, h)
        sub = solid[sy, sx]
        if sub.size == 0:
            continue
        fill = float(sub.mean())
        inside = float(sub.sum()) / total
        if fill < min_fill:
            continue
        plan.keep.append(det)
        if inside > best:
            best = inside
            plan.main = det
    plan.info["candidates"] = len(detections)
    plan.info["kept"] = len(plan.keep)
    return plan


def clip_outside(alpha: np.ndarray, plan: BoxPlan, *, margin: float = 0.025,
                 min_margin_px: float = 6.0, min_coverage: float = 0.85,
                 min_main_score: float = 0.35) -> Tuple[np.ndarray, Dict[str, object]]:
    """Pone a 0 lo que BiRefNet dejó fuera de las cajas del producto (atril, mesa).

    Sólo actúa si el detector está seguro (caja principal >= min_main_score) y si al
    menos `min_coverage` de la máscara cae dentro: si la caja deja fuera más que eso,
    es más probable que la caja esté mal que la máscara, y no se toca nada.
    """
    info: Dict[str, object] = {"clipped": False}
    if plan.main is None or plan.main.score < min_main_score or not plan.keep:
        info["reason"] = "no_confident_box"
        return alpha, info
    h, w = alpha.shape
    boxes = [expand_box(d.box, margin, w, h, min_margin_px) for d in plan.keep]
    region = region_from_boxes(boxes, w, h)
    solid = alpha >= SOLID
    total = float(solid.sum())
    if total <= 0:
        info["reason"] = "empty_mask"
        return alpha, info
    coverage = float((solid & region).sum()) / total
    info["coverage"] = round(coverage, 4)
    if coverage >= 0.999:
        info["reason"] = "nothing_outside"
        return alpha, info
    if coverage < min_coverage:
        info["reason"] = "box_disagrees"
        return alpha, info
    out = alpha.copy()
    out[~region] = 0
    info["clipped"] = True
    info["removed"] = round(1.0 - coverage, 4)
    return out, info


def needs_rescue(alpha: np.ndarray, *, kind: Optional[str] = None,
                 deficit_trigger: float = 0.035, max_components: int = 2) -> Tuple[bool, Dict[str, object]]:
    """¿La máscara parece mordida o despiezada?

    - Déficit de casco > trigger (sólo si el producto puede ser una cara plana).
    - Más de `max_components` trozos significativos: es la señal con la que el core
      descarta la foto (`foreground_fragmented`); el rescate puede salvarla.
    """
    deficit = hull_deficit(alpha)
    comps = len(significant_components(alpha))
    info: Dict[str, object] = {"deficit": deficit, "components": comps}
    reasons = []
    if kind != "objeto_sobre_fondo" and deficit > deficit_trigger:
        reasons.append("deficit")
    if comps > max_components:
        reasons.append("fragmented")
    info["reasons"] = reasons
    return bool(reasons), info


def rescue_box(alpha: np.ndarray, main: Optional[Detection], *, pad: float = 0.01) -> Optional[Box]:
    """Caja para SAM: la de la máscara (todos los trozos), unida a la del detector si coinciden."""
    h, w = alpha.shape
    mb = solid_bbox(alpha, VISIBLE)
    if mb is None:
        return main.box if main is not None else None
    box = mb
    if main is not None and box_iou(mb, main.box) >= 0.5:
        box = union_box(mb, main.box)
    return expand_box(box, pad, w, h, 2.0)


def merge_rescue(alpha: np.ndarray, sam_mask: np.ndarray, box: Box, *,
                 sam_max_deficit: float = 0.03, hull_dilate: float = 0.01,
                 erode_px: int = 2, feather_sigma: float = 0.8,
                 min_gain: float = 0.01) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """Rellena con la máscara de SAM sólo lo que falta DENTRO del casco del producto.

    Reglas, todas necesarias:
    1. La máscara de SAM es convexa (déficit <= sam_max_deficit): ve una cara plana.
       Un libro abierto, una pila o un jarrón la tumban, y se queda BiRefNet.
    2. Sólo se añade dentro del casco convexo de lo que BiRefNet ya vio (todos sus
       trozos), dilatado un 1 %: nunca se añaden esquinas de mesa fuera del producto.
    3. El resultado mejora de verdad: menos déficit o menos trozos.
    Los píxeles RGB no se tocan; sólo cambia el alfa.
    """
    h, w = alpha.shape
    info: Dict[str, object] = {}
    sam = sam_mask.astype(bool)
    sy, sx = _box_slices(expand_box(box, 0.01, w, h, 2.0), w, h)
    inbox = np.zeros_like(sam)
    inbox[sy, sx] = True
    sam &= inbox
    solid = alpha >= SOLID
    if sam.sum() < 0.5 * max(1, solid.sum()):
        info["reject"] = "sam_small"
        return None, info
    d_sam = hull_deficit(sam.astype(np.uint8) * 255)
    info["sam_deficit"] = d_sam
    if d_sam > sam_max_deficit:
        info["reject"] = "sam_not_convex"
        return None, info

    visible = alpha >= VISIBLE
    ys, xs = np.nonzero(visible)
    if len(xs) < 10:
        info["reject"] = "empty_mask"
        return None, info
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    allowed = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(allowed, hull, 1)
    grow = max(1, int(round(hull_dilate * math.hypot(w, h))))
    allowed = cv2.dilate(allowed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    add = sam & allowed.astype(bool)
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        add = cv2.erode(add.astype(np.uint8), k).astype(bool)
    soft = add.astype(np.float32) * 255.0
    if feather_sigma > 0:
        soft = cv2.GaussianBlur(soft, (0, 0), feather_sigma)
    merged = np.maximum(alpha, np.clip(soft, 0, 255).astype(np.uint8))

    d_before = hull_deficit(alpha)
    c_before = len(significant_components(alpha))
    d_after = hull_deficit(merged)
    c_after = len(significant_components(merged))
    info.update({"deficit_before": d_before, "deficit_after": d_after,
                 "components_before": c_before, "components_after": c_after,
                 "added": round(float((merged >= SOLID).sum() - solid.sum()) / max(1.0, float(solid.sum())), 4)})
    if d_after <= d_before - min_gain or c_after < c_before:
        return merged, info
    info["reject"] = "no_gain"
    return None, info
