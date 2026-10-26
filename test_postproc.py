"""Pruebas de la lógica de máscara con máscaras sintéticas (sin GPU ni modelos).

    python deploy/alibaba-cutout/test_postproc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import postproc as pp  # noqa: E402

H, W = 1200, 900
BOOK = (150, 150, 750, 1050)   # x1, y1, x2, y2


def rect(box, h=H, w=W, value=255):
    a = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = box
    a[y1:y2, x1:x2] = value
    return a


def test_work_size():
    assert pp.work_size(2000, 1500) == (1536, 1152)
    assert pp.work_size(500, 300) == (512, 288)
    assert pp.work_size(100, 80) == (256, 256)
    for w, h in [(1999, 1333), (1024, 1366), (3000, 400)]:
        tw, th = pp.work_size(w, h)
        assert tw % 32 == 0 and th % 32 == 0 and max(tw, th) <= 1536 + 16


def test_deficit():
    full = rect(BOOK)
    assert pp.hull_deficit(full) <= 0.012, pp.hull_deficit(full)
    bitten = full.copy()
    bitten[900:1050, 300:600] = 0          # mordisco abajo en el centro (código de barras)
    assert pp.hull_deficit(bitten) > 0.035, pp.hull_deficit(bitten)
    holed = full.copy()
    holed[850:980, 350:550] = 0            # hueco interior cerrado
    assert pp.hull_deficit(holed) > 0.035, pp.hull_deficit(holed)


def test_components():
    a = np.zeros((H, W), np.uint8)
    for x in (100, 400, 700):
        a[200:400, x:x + 120] = 255
    assert len(pp.significant_components(a)) == 3
    assert len(pp.significant_components(rect(BOOK))) == 1


def test_clip_removes_stand_outside_box():
    a = rect(BOOK)
    a[300:1150, 760:860] = 255             # tira de atril pegada a la derecha
    det = pp.Detection(0.62, tuple(float(v) for v in BOOK), "book")
    plan = pp.choose_boxes(a, [det])
    assert plan.main is det
    out, info = pp.clip_outside(a, plan)
    assert info["clipped"], info
    assert out[700, 840] == 0              # el atril desaparece
    assert out[600, 450] == 255            # el libro queda intacto
    assert out[600, 749] == 255            # el canto del libro queda intacto


def test_clip_skips_when_box_disagrees():
    a = rect(BOOK)
    det = pp.Detection(0.7, (150.0, 150.0, 750.0, 500.0), "book")   # caja de media portada
    plan = pp.choose_boxes(a, [det])
    out, info = pp.clip_outside(a, plan)
    assert not info["clipped"] and info["reason"] == "box_disagrees", info
    assert np.array_equal(out, a)


def test_clip_skips_low_score():
    a = rect(BOOK)
    a[300:1150, 760:860] = 255
    det = pp.Detection(0.31, tuple(float(v) for v in BOOK), "book")
    out, info = pp.clip_outside(a, pp.choose_boxes(a, [det]))
    assert not info["clipped"], info


def test_needs_rescue():
    full = rect(BOOK)
    ok, info = pp.needs_rescue(full)
    assert not ok, info
    bitten = full.copy()
    bitten[900:1050, 300:600] = 0
    ok, info = pp.needs_rescue(bitten)
    assert ok and "deficit" in info["reasons"], info
    ok, info = pp.needs_rescue(bitten, kind="objeto_sobre_fondo")
    assert not ok, info
    frag = np.zeros((H, W), np.uint8)
    for x in (160, 380, 600):
        frag[300:500, x:x + 120] = 255
    ok, info = pp.needs_rescue(frag)
    assert ok and "fragmented" in info["reasons"], info


def test_merge_fixes_bite():
    full = rect(BOOK)
    bitten = full.copy()
    bitten[900:1050, 300:600] = 0
    sam = full > 0
    box = pp.rescue_box(bitten, None)
    merged, info = pp.merge_rescue(bitten, sam, box)
    assert merged is not None, info
    assert info["deficit_after"] < 0.02, info
    assert merged[1000, 450] == 255        # el hueco del código de barras queda relleno
    assert merged[100, 450] == 0           # fuera del libro sigue transparente


def test_merge_fixes_fragments():
    full = rect(BOOK)
    frag = np.zeros((H, W), np.uint8)
    frag[160:400, 160:740] = 255           # renglones / ilustraciones sueltas
    frag[500:700, 160:740] = 255
    frag[800:1040, 160:740] = 255
    sam = full > 0
    merged, info = pp.merge_rescue(frag, sam, pp.rescue_box(frag, None))
    assert merged is not None, info
    assert info["components_after"] == 1, info


def test_merge_never_adds_table_corners():
    # Libro girado (cuadrilátero convexo) con mordisco; SAM rellena TODA su caja.
    a = np.zeros((H, W), np.uint8)
    quad = np.array([[250, 150], [750, 250], [650, 1050], [150, 950]], np.int32)
    cv2.fillConvexPoly(a, quad, 255)
    a[850:1000, 300:500] = 0
    x, y, bw, bh = cv2.boundingRect(quad)
    sam = rect((x, y, x + bw, y + bh)) > 0
    merged, info = pp.merge_rescue(a, sam, (float(x), float(y), float(x + bw), float(y + bh)))
    assert merged is not None, info
    assert merged[160, 160] == 0 and merged[1040, 740] == 0, "añadió esquinas de mesa"
    assert merged[900, 400] == 255


def test_merge_rejects_non_convex_sam():
    bitten = rect(BOOK)
    bitten[900:1050, 300:600] = 0
    sam = rect(BOOK) > 0
    sam[150:600, 450:750] = False          # forma en L: no es una cara plana
    merged, info = pp.merge_rescue(bitten, sam, pp.rescue_box(bitten, None))
    assert merged is None and info["reject"] == "sam_not_convex", info


def test_merge_rejects_without_gain():
    full = rect(BOOK)
    merged, info = pp.merge_rescue(full, full > 0, pp.rescue_box(full, None))
    assert merged is None and info["reject"] == "no_gain", info


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
    print("{}/{} ok".format(len(tests) - failed, len(tests)))
    sys.exit(1 if failed else 0)
