"""Lógica de máscara del servidor de recorte v3 (numpy + OpenCV, sin torch).

Separada de `server.py` para poder probarla sin GPU ni modelos
(`python test_postproc.py`). Todas las máscaras son arrays HxW: `alpha` uint8
(0..255) o booleanos. Las cajas son (x1, y1, x2, y2) en píxeles de la imagen.

Idea central (v3): dos modelos con papeles distintos.
  * El modelo BASE (BiRefNet / RMBG-2.0 / BEN2) dibuja el borde fino del producto,
    pero no sabe qué ES producto: mete el atril, la caja o la mano, y cuando el
    producto llena el encuadre recorta lo impreso en vez del objeto.
  * SAM 3 (`Scene`) sabe qué es producto y qué no: instancias de "book"/"handbag"...,
    manos y soportes, con una puntuación de presencia. Sus bordes son bastos.
`analyze()` cruza los dos: rechaza lo que no tiene arreglo (producto cortado por
el encuadre, mano que lo tapa, nada de producto) y arregla lo que sí (fondo/atril
fuera del producto, huecos y zonas translúcidas en una cara plana, mordiscos que
SAM 3 sí ve enteros), refinando los bordes añadidos con un filtro guiado por el RGB.

Medidas heredadas de la auditoría del 2026-10-25 (déficit de casco convexo =
1 - área(alfa>=128) / área(casco), con la máscara a 400 px de lado): portadas
planas buenas <= 0,012; contraportada mordida de Picoult 0,127; pulgar borrado
0,048; libros abiertos/pilas 0,2-0,4 (no convexos por naturaleza).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
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
    if alpha.dtype == bool:
        alpha = alpha.astype(np.uint8) * 255
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


def mask_bbox(mask: np.ndarray) -> Optional[Box]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


# --------------------------------------------------------------------------- morfología
def _kernel(px: int) -> np.ndarray:
    px = max(1, int(px))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), _kernel(px)).astype(bool)


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    return cv2.erode(mask.astype(np.uint8), _kernel(px)).astype(bool)


def hull_mask(mask: np.ndarray) -> np.ndarray:
    """Casco convexo (relleno) de una máscara booleana, a resolución completa."""
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = [c.reshape(-1, 2) for c in contours if len(c) > 0]
    if not pts:
        return np.zeros(mask.shape, bool)
    hull = cv2.convexHull(np.concatenate(pts).astype(np.int32))
    canvas = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(canvas, hull, 1)
    return canvas.astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Rellena los huecos cerrados de una máscara booleana (lo no conectado al borde)."""
    m = mask.astype(np.uint8)
    h, w = m.shape
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1:-1, 1:-1] = m
    flood = padded.copy()
    ff_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 1)
    holes = (flood == 0)[1:-1, 1:-1]
    return mask.astype(bool) | holes


def drop_components(mask: np.ndarray, keep_touching: np.ndarray) -> Tuple[np.ndarray, int]:
    """Quita los trozos 8-conexos de `mask` que no tocan `keep_touching`."""
    n, labels, _stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.astype(bool), 0
    touching = np.unique(labels[keep_touching & mask.astype(bool)])
    keep = np.isin(labels, touching[touching > 0])
    dropped = int(n - 1 - len(touching[touching > 0]))
    return keep & mask.astype(bool), dropped


# --------------------------------------------------------------------------- filtro guiado
def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    """Filtro guiado (He et al.) con box filters de OpenCV. `guide` y `src` float32 en 0..1."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, k)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, k)
    corr_i = cv2.boxFilter(guide * guide, cv2.CV_32F, k)
    corr_ip = cv2.boxFilter(guide * src, cv2.CV_32F, k)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return cv2.boxFilter(a, cv2.CV_32F, k) * guide + cv2.boxFilter(b, cv2.CV_32F, k)


def refine_band(alpha: np.ndarray, rgb: Optional[np.ndarray], region: np.ndarray, *,
                radius: int = 16, eps: float = 1e-4, clamp: Tuple[float, float] = (0.3, 0.7)) -> np.ndarray:
    """Alinea con el RGB el borde exterior de `region` (píxeles añadidos desde SAM 3).

    SAM 3 devuelve máscaras a baja resolución (escalones al ampliarlas); el filtro
    guiado por la luminancia mueve ese borde al borde real del objeto. Medido en
    sintético: con radio ~1/100 del lado largo (16 px a 1500) y recorte 0,3-0,7
    desaparecen dientes de 20 px; con radio 4 apenas cambian. Sólo se toca la banda
    alrededor del contorno exterior; el interior y el alfa del modelo base quedan intactos.
    """
    if rgb is None or not region.any():
        return alpha
    solid = alpha >= SOLID
    band = dilate(region, radius) & ~erode(solid, radius)
    if not band.any():
        return alpha
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    src = alpha.astype(np.float32) / 255.0
    q = guided_filter(gray, src, radius, eps)
    lo, hi = clamp
    q = np.clip((q - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    out = alpha.copy()
    out[band] = np.round(q[band] * 255.0).astype(np.uint8)
    return out


def _lab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    lab[..., 1:] -= 128.0
    return lab


def color_snap(alpha: np.ndarray, rgb: Optional[np.ndarray], ring: np.ndarray, interior: np.ndarray,
               outside: np.ndarray, *, radius: int, delta_e: float = 18.0, margin: float = 6.0) -> np.ndarray:
    """Dentro del anillo de tolerancia alrededor del producto SAM 3, quita lo que no es del
    mismo color que el producto adyacente y sí del color de lo que hay fuera (la repisa
    o peana de un atril, la base de metacrilato, una franja de mesa).

    El filtro guiado no sirve aquí (medido: deja la repisa casi entera, porque dentro de
    la ventana hay tres poblaciones y el modelo lineal no las separa); comparar colores
    Lab locales sí: el canto real del libro es continuo con la portada; la repisa no.
    """
    if rgb is None or not ring.any():
        return alpha
    lab = _lab(rgb)
    k = (2 * radius + 1, 2 * radius + 1)

    def local_mean(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m = mask.astype(np.float32)
        num = cv2.boxFilter(lab * m[..., None], cv2.CV_32F, k)
        den = cv2.boxFilter(m, cv2.CV_32F, k)
        return num / np.maximum(den, 1e-6)[..., None], den

    ref_in, den_in = local_mean(interior)
    ref_out, den_out = local_mean(outside)
    d_in = np.linalg.norm(lab - ref_in, axis=2)
    d_out = np.linalg.norm(lab - ref_out, axis=2)
    tiny = 1e-4
    foreign = ring & (den_in > tiny) & (d_in > delta_e) & ((den_out <= tiny) | (d_in > d_out + margin))
    if not foreign.any():
        return alpha
    out = alpha.copy()
    out[foreign] = 0
    return out


# --------------------------------------------------------------------------- escena SAM 3
@dataclass
class Instance:
    mask: np.ndarray            # bool HxW
    score: float
    label: str = ""

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def box(self) -> Optional[Box]:
        return mask_bbox(self.mask)


@dataclass
class Scene:
    """Lo que SAM 3 ve en la foto: instancias del producto, manos y soportes."""
    products: List[Instance] = field(default_factory=list)
    hands: List[Instance] = field(default_factory=list)
    supports: List[Instance] = field(default_factory=list)


@dataclass
class Params:
    """Umbrales de `analyze`; cualquiera se puede cambiar por categoría sin tocar código."""
    product_score: float = 0.5        # puntuación mínima (clasificación x presencia) de una instancia de producto
    hand_score: float = 0.4
    support_score: float = 0.4
    match_overlap: float = 0.25       # solape mínimo instancia<->máscara base para "ser el mismo producto"
    frame_margin_frac: float = 0.006  # banda de borde que cuenta como contacto (mín. 2 px)
    bottom_contact_max: float = 0.10  # anchura de contacto inferior tolerada (fracción del ancho del producto)
    hand_outside_min: float = 0.30    # parte de la mano fuera del producto para ser oclusión (no una mano impresa)
    hand_touch_frac: float = 0.01
    mismatch_cover_min: float = 0.50  # si menos de esto de la máscara base es producto, no nos fiamos
    rescue_fill_max: float = 0.60     # base cubre menos de esto del producto => recortó lo impreso: rescate
    clip_dilate_frac: float = 0.012   # margen alrededor del producto SAM 3 que se respeta del borde base
    support_dilate_frac: float = 0.002
    outside_component_min: float = 0.01   # no planos: sólo se quitan trozos ajenos >= 1 % del área base
    snap_delta_e: float = 18.0        # planos: color del anillo distinto del producto => ajeno (repisa, peana)
    snap_margin: float = 6.0
    solidify_erode_frac: float = 0.015
    solidify_max_deficit: float = 0.08
    bite_deficit: float = 0.035
    sam_convex_deficit: float = 0.03
    recover_part_max_frac: float = 0.03   # no planos: trozos de producto que SAM 3 ve y la base no (herrajes)
    refine_radius_frac: float = 0.01  # radio del filtro guiado (fracción del lado largo, mín. 8 px)
    guided_eps: float = 1e-4

    @classmethod
    def from_dict(cls, conf: Optional[Dict[str, object]]) -> "Params":
        p = cls()
        if not conf:
            return p
        names = {f.name for f in fields(cls)}
        for k, v in conf.items():
            if k not in names:
                continue
            current = getattr(p, k)
            try:
                setattr(p, k, int(v) if isinstance(current, int) else float(v))
            except (TypeError, ValueError):
                pass
        return p


@dataclass
class Result:
    alpha: np.ndarray
    ok: bool
    reason: str
    info: Dict[str, object]


# --------------------------------------------------------------------------- piezas
def frame_contact(mask: np.ndarray, margin_px: int) -> Dict[str, object]:
    """Qué bordes de la imagen toca la máscara y qué anchura tiene el contacto inferior."""
    h, w = mask.shape
    m = max(1, int(margin_px))
    sides = []
    if mask[:m, :].any():
        sides.append("top")
    if mask[-m:, :].any():
        sides.append("bottom")
    if mask[:, :m].any():
        sides.append("left")
    if mask[:, -m:].any():
        sides.append("right")
    bottom_span = 0.0
    if "bottom" in sides:
        bb = mask_bbox(mask)
        width = max(1.0, bb[2] - bb[0]) if bb else float(w)
        bottom_span = float(mask[-m:, :].any(axis=0).sum()) / width
    return {"sides": sides, "bottom_span": round(bottom_span, 3)}


def match_products(base_solid: np.ndarray, products: Sequence[Instance], *,
                   min_overlap: float = 0.25) -> List[Instance]:
    """Instancias de producto que se solapan con lo que vio el modelo base."""
    base_area = float(base_solid.sum())
    out = []
    for p in products:
        area = float(p.mask.sum())
        if area <= 0:
            continue
        inter = float((p.mask & base_solid).sum())
        if inter / area >= min_overlap or (base_area > 0 and inter / base_area >= min_overlap):
            out.append(p)
    return out


def hand_occlusion(hands: Sequence[Instance], union: np.ndarray, *, touch_px: int,
                   outside_min: float) -> Optional[Dict[str, object]]:
    """Una mano que TOCA el producto y queda en buena parte fuera de él lo está sujetando
    (oclusión real). Una mano impresa en la portada queda dentro y se ignora."""
    near = dilate(union, touch_px)
    for hnd in hands:
        area = float(hnd.mask.sum())
        if area <= 0:
            continue
        inside = float((hnd.mask & near).sum())
        if inside <= 0:
            continue                       # no toca el producto: mano en el fondo
        outside = 1.0 - inside / area
        if outside >= outside_min:
            return {"score": round(hnd.score, 3), "outside": round(outside, 3)}
    return None


def union_of(instances: Sequence[Instance], shape: Tuple[int, int]) -> np.ndarray:
    u = np.zeros(shape, bool)
    for inst in instances:
        u |= inst.mask
    return u


# --------------------------------------------------------------------------- análisis
def _reject(alpha: np.ndarray, reason: str, info: Dict[str, object]) -> Result:
    info["verdict"] = "reject"
    info["reason"] = reason
    return Result(alpha, False, reason, info)


def analyze(alpha: np.ndarray, rgb: Optional[np.ndarray], scene: Scene, *, flat: bool,
            kind: Optional[str] = None, params: Optional[Params] = None) -> Result:
    """Cruza la máscara base con la escena de SAM 3 y decide: alfa final + veredicto.

    `flat` = categoría de cara plana (libros): activa el relleno de huecos, la
    solidificación del interior y la regla de mordisco. `kind` es la clase que da el
    core (cara_plana / objeto_sobre_fondo) y sólo se usa para desempatar el caso
    ambiguo "ni la base ni SAM 3 ven una forma convexa" (libro abierto vs. mordido).
    """
    P = params or Params()
    info: Dict[str, object] = {}
    h, w = alpha.shape
    long_side = max(w, h)
    base_solid = alpha >= SOLID
    base_area = float(base_solid.sum())
    if base_area < 0.002 * h * w:
        return _reject(alpha, "empty_mask", info)

    products = [p for p in scene.products if p.score >= P.product_score]
    info["products"] = len(products)
    info["product_score"] = round(max((p.score for p in scene.products), default=0.0), 3)
    if not products:
        return _reject(alpha, "no_product", info)
    matched = match_products(base_solid, products, min_overlap=P.match_overlap)
    info["matched"] = len(matched)
    if not matched:
        return _reject(alpha, "product_mismatch", info)
    union = union_of(matched, (h, w))
    if flat:
        # La instancia "book" de SAM 3 deja huecos donde hay figuras impresas (medido en
        # "Hindi-Bindi": excluía a las dos mujeres y sus manos dibujadas caían "fuera" del
        # libro => rechazo por mano). Una cara plana no tiene agujeros: se rellenan.
        filled = fill_holes(union)
        if filled.sum() > union.sum():
            info["union_holes_filled"] = round(float(filled.sum() - union.sum()) / max(1.0, float(filled.sum())), 4)
            union = filled
    union_area = float(union.sum())

    # 1) Producto cortado por el encuadre: no tiene arreglo.
    fc = frame_contact(union, max(2, int(round(P.frame_margin_frac * long_side))))
    info["frame"] = fc
    unsafe = [s for s in fc["sides"] if s != "bottom"]
    if unsafe:
        return _reject(alpha, "product_cut:" + ",".join(unsafe), info)
    if "bottom" in fc["sides"] and fc["bottom_span"] >= P.bottom_contact_max:
        return _reject(alpha, "product_cut:bottom", info)

    # 2) Mano que sujeta/tapa el producto: nada puede inventar lo que hay debajo.
    hands = [x for x in scene.hands if x.score >= P.hand_score]
    info["hands"] = len(hands)
    # En una cara plana se mide contra el casco convexo del producto: las figuras impresas
    # dejan "bahías" abiertas en la máscara de SAM 3 (medido en Hindi-Bindi: la copa y la mano
    # dibujadas tocan el canto inferior) y una mano impresa caería medio "fuera". Una mano que
    # sujeta el libro de verdad sale del casco. En bolsos no: la mano en el hueco del asa
    # debe seguir contando como fuera.
    hand_region = hull_mask(union) if flat else union
    occl = hand_occlusion(hands, hand_region, touch_px=max(2, int(round(P.hand_touch_frac * long_side))),
                          outside_min=P.hand_outside_min)
    if occl:
        info["hand"] = occl
        return _reject(alpha, "hand_occlusion", info)

    # 3) Fusión base <-> producto.
    inter = float((base_solid & union).sum())
    cover = inter / base_area                 # cuánta máscara base es producto
    fill = inter / max(1.0, union_area)       # cuánto producto cubre la base
    info["cover"] = round(cover, 3)
    info["fill"] = round(fill, 3)
    if cover < P.mismatch_cover_min:
        return _reject(alpha, "product_mismatch", info)

    out = alpha.copy()
    added = np.zeros((h, w), bool)
    if fill < P.rescue_fill_max:
        # La base recortó lo impreso (ilustración, título): el producto es lo que ve SAM 3.
        add = union & ~base_solid
        out[add] = 255
        added |= add
        info["rescued"] = round(1.0 - fill, 3)

    r_out = max(2, int(round(P.clip_dilate_frac * long_side)))
    sam_region = hull_mask(union) if flat else union
    keep_region = dilate(sam_region, r_out)
    visible = out >= VISIBLE
    outside = visible & ~keep_region
    if outside.any():
        if flat:
            out[outside] = 0
            info["clipped"] = round(float(outside.sum()) / base_area, 4)
        else:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(outside.astype(np.uint8), connectivity=8)
            removed = np.zeros((h, w), bool)
            for i in range(1, n):
                if stats[i, cv2.CC_STAT_AREA] >= P.outside_component_min * base_area:
                    removed |= labels == i
            if removed.any():
                out[removed] = 0
                info["clipped"] = round(float(removed.sum()) / base_area, 4)

    supports = [s for s in scene.supports if s.score >= P.support_score]
    info["supports"] = len(supports)
    if supports:
        r_small = max(1, int(round(P.support_dilate_frac * long_side)))
        near = dilate(union, r_small)
        region = union_of(supports, (h, w)) & ~near & (out >= VISIBLE)
        if region.any():
            out[region] = 0
            info["support_removed"] = round(float(region.sum()) / base_area, 4)

    if flat:
        # Anillo de tolerancia: lo que la base conserva pegado al producto pero fuera del
        # contorno de SAM 3. Si su color no es el del producto adyacente, es ajeno.
        ring = keep_region & ~sam_region & (out >= VISIBLE)
        if ring.any():
            before = int(ring.sum())
            out = color_snap(out, rgb, ring, erode(union, 2) & (out >= SOLID), ~keep_region,
                             radius=3 * r_out, delta_e=P.snap_delta_e, margin=P.snap_margin)
            snapped = before - int((ring & (out >= VISIBLE)).sum())
            if snapped > 0:
                info["snapped"] = round(float(snapped) / base_area, 4)

    # Trozos sueltos que no tocan el producto (restos de caja, sombras lejanas).
    vis = out >= VISIBLE
    kept, dropped = drop_components(vis, dilate(union, r_out))
    if dropped:
        out[vis & ~kept] = 0
        info["dropped_components"] = dropped

    single_flat = flat and len(matched) == 1 and kind != "objeto_sobre_fondo"
    d_sam = hull_deficit(union)
    info["sam_deficit"] = d_sam
    if single_flat:
        # 4) Integridad de la cara plana: un libro es opaco y no tiene agujeros.
        vis = out >= VISIBLE
        filled = fill_holes(vis)
        holes = filled & ~vis
        if holes.any():
            out[holes] = 255
            added |= holes
            info["holes_filled"] = round(float(holes.sum()) / base_area, 4)
        d_now = hull_deficit(out)
        if d_now <= P.solidify_max_deficit:
            k = max(2, int(round(P.solidify_erode_frac * long_side)))
            core = erode(hull_mask(out >= SOLID), k) & dilate(union, max(1, k // 3))
            soft = core & (out < 255)
            if soft.any():
                info["solidified"] = round(float(soft.sum()) / base_area, 4)
                out[soft] = 255
        d_now = hull_deficit(out)
        if d_now > P.bite_deficit and d_sam <= P.sam_convex_deficit:
            # SAM 3 ve la cara entera y la base la dejó mordida: se rellena lo que falta.
            fillr = union & ~(out >= SOLID)
            if fillr.any():
                out[fillr] = 255
                added |= fillr
                info["bite_filled"] = round(float(fillr.sum()) / base_area, 4)
                d_now = hull_deficit(out)
        info["deficit"] = d_now
        if d_now > P.bite_deficit:
            if d_sam > P.sam_convex_deficit and kind != "cara_plana":
                # Ni la base ni SAM 3 ven una forma convexa y el core no asegura que sea una
                # portada: puede ser un libro abierto. Se acepta y se deja constancia.
                info["nonconvex_accepted"] = True
            else:
                return _reject(out, "bitten", info)
    else:
        info["deficit"] = hull_deficit(out)
        if not flat and P.recover_part_max_frac > 0:
            # Herrajes y detalles pequeños que SAM 3 ve como producto y la base perdió.
            missing = union & ~(out >= VISIBLE)
            if missing.any():
                n, labels, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
                near = dilate(out >= SOLID, 3)
                recovered = np.zeros((h, w), bool)
                for i in range(1, n):
                    part = labels == i
                    if stats[i, cv2.CC_STAT_AREA] <= P.recover_part_max_frac * union_area and (part & near).any():
                        recovered |= part
                if recovered.any():
                    out[recovered] = 255
                    added |= recovered
                    info["recovered_parts"] = round(float(recovered.sum()) / base_area, 4)

    if added.any():
        radius = max(8, int(round(P.refine_radius_frac * long_side)))
        out = refine_band(out, rgb, added, radius=radius, eps=P.guided_eps)
        info["added"] = round(float(added.sum()) / base_area, 4)

    info["components"] = len(significant_components(out))
    info["verdict"] = "ok"
    info["reason"] = "ok"
    return Result(out, True, "ok", info)


__all__ = [
    "SOLID", "VISIBLE", "Box", "Instance", "Scene", "Params", "Result",
    "work_size", "hull_deficit", "significant_components", "mask_bbox",
    "dilate", "erode", "hull_mask", "fill_holes", "drop_components",
    "guided_filter", "refine_band", "color_snap", "frame_contact", "match_products", "hand_occlusion",
    "union_of", "analyze",
]
