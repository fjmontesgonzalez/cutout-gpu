"""Pruebas de la lógica de máscara v3 con escenas sintéticas (sin GPU ni modelos).

    python deploy/alibaba-cutout/test_postproc.py

Cada prueba reproduce un tipo de error visto en la revisión del 2026-10-26
(OBSERVACIONES.md): atril retenido (fotos 9, 20, 21, 83), mano que sujeta el libro
(47, 81, 84), producto que llena el encuadre o recortado sólo por lo impreso
(6, 14, 24), agujeros y zonas translúcidas en una cara plana (57, 66, 77), pila de
libros legítima (78), bolso con herraje perdido (87, 88) y restos de caja (Bria 87).
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


def rect(box, h=H, w=W, value=255, dtype=np.uint8):
    a = np.zeros((h, w), dtype)
    x1, y1, x2, y2 = box
    a[y1:y2, x1:x2] = value
    return a


def brect(box, h=H, w=W):
    return rect(box, h, w, 1, np.uint8).astype(bool)


def inst(mask, score=0.9, label="book"):
    return pp.Instance(mask=mask, score=score, label=label)


def scene(products=(), hands=(), supports=()):
    return pp.Scene(products=list(products), hands=list(hands), supports=list(supports))


def photo(book=BOOK, bg=(190, 180, 160), fg=(40, 60, 120), shelf=None):
    """RGB sintético: libro oscuro sobre mesa clara (y opcionalmente una repisa)."""
    img = np.full((H, W, 3), bg, np.uint8)
    x1, y1, x2, y2 = book
    img[y1:y2, x1:x2] = fg
    if shelf is not None:
        sx1, sy1, sx2, sy2, color = shelf
        img[sy1:sy2, sx1:sx2] = color
    return img


# --------------------------------------------------------------------------- básicos
def test_work_size():
    assert pp.work_size(2000, 1500) == (1536, 1152)
    assert pp.work_size(500, 300) == (512, 288)
    assert pp.work_size(100, 80) == (256, 256)
    for w, h in [(1999, 1333), (1024, 1366), (3000, 400)]:
        tw, th = pp.work_size(w, h)
        assert tw % 32 == 0 and th % 32 == 0 and max(tw, th) <= 1536 + 16


def test_deficit_and_components():
    full = rect(BOOK)
    assert pp.hull_deficit(full) <= 0.012, pp.hull_deficit(full)
    bitten = full.copy()
    bitten[900:1050, 300:600] = 0
    assert pp.hull_deficit(bitten) > 0.035
    assert pp.hull_deficit(brect(BOOK)) <= 0.012          # acepta booleanos
    a = np.zeros((H, W), np.uint8)
    for x in (100, 400, 700):
        a[200:400, x:x + 120] = 255
    assert len(pp.significant_components(a)) == 3
    assert len(pp.significant_components(full)) == 1


def test_fill_holes_and_hull():
    m = brect(BOOK)
    m[500:560, 400:460] = False                           # agujero interior
    m[1000:1050, 0:100] = True                            # trozo aparte que toca el borde: no es hueco
    filled = pp.fill_holes(m)
    assert filled[530, 430] and not filled[50, 50]
    hull = pp.hull_mask(brect(BOOK))
    assert hull[600, 450] and not hull[100, 100]
    assert abs(hull.sum() - brect(BOOK).sum()) < 0.01 * brect(BOOK).sum()


def test_frame_contact():
    m = brect((0, 100, 500, 1100))
    fc = pp.frame_contact(m, 7)
    assert fc["sides"] == ["left"], fc
    bottom = brect((200, 300, 700, H))
    fc = pp.frame_contact(bottom, 7)
    assert fc["sides"] == ["bottom"] and fc["bottom_span"] > 0.95, fc
    shadow = brect(BOOK)
    shadow[H - 7:, 440:460] = True                        # contacto fino (sombra/peana)
    fc = pp.frame_contact(shadow, 7)
    assert fc["sides"] == ["bottom"] and fc["bottom_span"] < 0.05, fc


def test_params_from_dict():
    p = pp.Params.from_dict({"bite_deficit": "0.05", "snap_delta_e": "22", "desconocido": 3, "hand_score": "x"})
    assert p.bite_deficit == 0.05 and p.snap_delta_e == 22.0
    assert p.hand_score == pp.Params().hand_score


# --------------------------------------------------------------------------- rechazos
def test_rejects_without_product():
    r = pp.analyze(rect(BOOK), None, scene(), flat=True)
    assert not r.ok and r.reason == "no_product", r.info
    r = pp.analyze(rect(BOOK), None, scene([inst(brect(BOOK), score=0.3)]), flat=True)
    assert not r.ok and r.reason == "no_product", r.info


def test_rejects_empty_mask():
    r = pp.analyze(np.zeros((H, W), np.uint8), None, scene([inst(brect(BOOK))]), flat=True)
    assert not r.ok and r.reason == "empty_mask"


def test_rejects_product_cut_by_frame():
    # Foto 24 / 14 / 6: el libro llega al borde del encuadre (SAM 3 lo ve tocando el marco).
    cut = (150, 0, 750, 1050)
    r = pp.analyze(rect(cut), None, scene([inst(brect(cut))]), flat=True)
    assert not r.ok and r.reason == "product_cut:top", r.info
    cut_bottom = (150, 150, 750, H)
    r = pp.analyze(rect(cut_bottom), None, scene([inst(brect(cut_bottom))]), flat=True)
    assert not r.ok and r.reason == "product_cut:bottom", r.info


def test_bottom_shadow_contact_is_tolerated():
    base = rect(BOOK)
    base[H - 4:, 440:460] = 200                           # sombra fina que llega abajo (modelo base)
    prod = brect(BOOK)
    prod[H - 4:, 445:455] = True                          # SAM 3 también roza el borde, estrecho
    r = pp.analyze(base, None, scene([inst(prod)]), flat=True)
    assert r.ok, r.info


def test_rejects_hand_holding_book():
    # Foto 84: la mano entra por la izquierda y tapa parte de la portada; la base borra el pulgar.
    base = rect(BOOK)
    base[600:720, 150:330] = 0                            # mordisco del pulgar
    prod = brect(BOOK)
    prod[600:720, 150:330] = False                        # SAM 3 tampoco ve el libro bajo el pulgar
    hand = np.zeros((H, W), bool)
    cv2.ellipse(hand.view(np.uint8), (120, 660), (200, 120), 0, 0, 360, 1, -1)
    r = pp.analyze(base, None, scene([inst(prod)], hands=[inst(hand, 0.8, "hand")]), flat=True)
    assert not r.ok and r.reason == "hand_occlusion", r.info


def test_printed_hand_on_cover_is_ignored():
    hand = brect((300, 400, 500, 650))                    # mano dibujada: enteramente dentro del libro
    r = pp.analyze(rect(BOOK), None, scene([inst(brect(BOOK))], hands=[inst(hand, 0.9, "hand")]), flat=True)
    assert r.ok, r.info


def test_hand_in_background_is_ignored():
    hand = brect((780, 50, 890, 130))                     # mano lejos del libro
    r = pp.analyze(rect(BOOK), None, scene([inst(brect(BOOK))], hands=[inst(hand, 0.9, "hand")]), flat=True)
    assert r.ok, r.info


def test_rejects_when_base_and_sam_disagree():
    base = rect((150, 150, 750, 1050))
    other = brect((600, 900, 880, 1190))                  # SAM 3 ve otro objeto que apenas solapa
    r = pp.analyze(base, None, scene([inst(other)]), flat=True)
    assert not r.ok and r.reason == "product_mismatch", r.info


def test_rejects_bitten_flat_cover_when_core_says_flat():
    base = rect(BOOK)
    base[850:1050, 150:450] = 0                           # esquina mordida, sin mano detectada
    prod = brect(BOOK)
    prod[850:1050, 150:450] = False                       # SAM 3 tampoco la ve
    r = pp.analyze(base, None, scene([inst(prod)]), flat=True, kind="cara_plana")
    assert not r.ok and r.reason == "bitten", r.info
    # Sin la clase del core puede ser un libro abierto: se acepta y se deja constancia.
    r = pp.analyze(base, None, scene([inst(prod)]), flat=True)
    assert r.ok and r.info.get("nonconvex_accepted"), r.info


# --------------------------------------------------------------------------- arreglos
def test_removes_stand_below_book():
    # Fotos 9, 20, 21, 83: la base incluye la peana/repisa del atril bajo el libro.
    base = rect(BOOK)
    base[1050:1120, 100:800] = 255                        # repisa más ancha que el libro
    r = pp.analyze(base, photo(), scene([inst(brect(BOOK))]), flat=True)
    assert r.ok, r.info
    assert r.alpha[1100, 450] == 0 and r.alpha[1100, 120] == 0, "la repisa debía desaparecer"
    assert r.alpha[600, 450] == 255 and r.alpha[1040, 450] == 255, "el libro queda entero"
    assert r.info.get("clipped", 0) > 0.05


def test_removes_stand_tab_inside_hull_margin_with_support_instance():
    base = rect(BOOK)
    base[1050:1058, 400:500] = 255                        # pestaña fina justo bajo el libro (foto 20)
    stand = brect((380, 1045, 520, 1070))
    r = pp.analyze(base, None, scene([inst(brect(BOOK))], supports=[inst(stand, 0.7, "book stand")]), flat=True)
    assert r.ok and r.alpha[1055, 450] == 0 and r.alpha[1049, 450] == 255, r.info


def test_color_snap_removes_shelf_sliver_inside_margin():
    # Foto 83: tras recortar por el casco de SAM 3 queda una lámina de repisa (clara) dentro del
    # margen de tolerancia bajo un libro oscuro. El color la delata y se quita; el canto no.
    rgb = photo(shelf=(100, 1050, 800, 1070, (235, 235, 230)))
    base = rect(BOOK)
    base[1050:1070, 100:800] = 255                        # repisa 20 px (dentro del margen de 14 px + algo)
    prod = brect((150, 150, 750, 1046))                   # SAM 3 se queda 4 px corto en el canto inferior
    r = pp.analyze(base, rgb, scene([inst(prod)]), flat=True)
    assert r.ok, r.info
    assert (r.alpha[1052:1064, 200:700] == 0).all(), "repisa fuera"
    assert (r.alpha[1046:1049, 200:700] == 255).all(), "el canto real del libro se conserva"
    assert r.info.get("snapped", 0) > 0


def test_rescues_print_only_mask():
    # Fotos 24 / Radar / King Richard: la base recortó sólo la etiqueta; SAM 3 ve el libro.
    label = rect((250, 650, 650, 1000))
    r = pp.analyze(label, photo(), scene([inst(brect(BOOK))]), flat=True)
    assert r.ok, r.info
    assert r.info.get("rescued", 0) > 0.5
    assert r.alpha[200, 200] >= 250 and r.alpha[1000, 700] >= 250, "todo el libro opaco"
    assert r.alpha[100, 100] == 0 and r.alpha[1150, 850] == 0


def test_fills_holes_and_solidifies_flat_face():
    # Fotos 57 / 66 / 77: motas transparentes y zona translúcida dentro de la portada.
    base = rect(BOOK)
    base[500:520, 400:420] = 0                            # agujero
    base[300:700, 700:750] = 110                          # canto translúcido (DVD)
    r = pp.analyze(base, photo(), scene([inst(brect(BOOK))]), flat=True)
    assert r.ok, r.info
    assert r.alpha[510, 410] == 255, "agujero relleno"
    assert r.alpha[500, 725] == 255, "interior solidificado"
    assert r.info.get("holes_filled") and r.info.get("solidified")


def test_fills_bite_when_sam_sees_whole_face():
    # Picoult: la base se come el pie de la contraportada; SAM 3 ve la cara entera.
    base = rect(BOOK)
    base[900:1050, 300:600] = 0
    r = pp.analyze(base, photo(), scene([inst(brect(BOOK))]), flat=True, kind="cara_plana")
    assert r.ok, r.info
    assert r.alpha[1000, 450] >= 250 and r.info["deficit"] <= 0.02, r.info


def test_stack_of_books_is_not_treated_as_flat_face():
    # Foto 78: pila de libros (no convexa, varias instancias). No se rellena ni se rechaza.
    stack = np.zeros((H, W), np.uint8)
    rows = [(200, 300, 700, 420), (180, 420, 720, 540), (220, 540, 680, 660), (160, 660, 740, 780)]
    insts = []
    for b in rows:
        stack[b[1]:b[3], b[0]:b[2]] = 255
        insts.append(inst(brect(b), 0.8))
    r = pp.analyze(stack, None, scene(insts), flat=True)
    assert r.ok and r.info["matched"] == 4, r.info
    assert r.alpha[350, 190] == 0, "no se rellenó el escalón de la pila"


def test_open_book_with_core_kind_object_is_accepted():
    pages = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(pages, np.array([[100, 300], [450, 250], [450, 1000], [100, 900]], np.int32), 255)
    cv2.fillConvexPoly(pages, np.array([[450, 250], [800, 300], [800, 900], [450, 1000]], np.int32), 255)
    r = pp.analyze(pages, None, scene([inst(pages > 0)]), flat=True, kind="objeto_sobre_fondo")
    assert r.ok and "nonconvex_accepted" not in r.info, r.info


def test_bag_keeps_strap_and_drops_box_remnants():
    # Foto 87 (Bria): trozos blancos de la caja junto a la correa; correa aparte como instancia.
    bag = brect((200, 500, 700, 900))
    strap = brect((250, 150, 650, 300))
    base = (bag | strap).astype(np.uint8) * 255
    base[150:300, 660:760] = 255                          # resto de caja pegado a la correa
    base[950:1000, 100:160] = 255                         # mota lejana
    r = pp.analyze(base, None, scene([inst(bag, 0.9, "handbag"), inst(strap, 0.7, "strap")]), flat=False)
    assert r.ok, r.info
    assert r.alpha[200, 700] == 0 and r.alpha[970, 130] == 0, "restos de caja fuera"
    assert r.alpha[220, 400] == 255 and r.alpha[700, 450] == 255, "bolso y correa intactos"


def test_bag_recovers_small_hardware_seen_by_sam():
    bag = brect((200, 500, 700, 900))
    clasp = brect((430, 470, 470, 500))                   # mosquetón pegado al bolso
    base = bag.astype(np.uint8) * 255                     # la base lo perdió
    r = pp.analyze(base, None, scene([inst(bag | clasp, 0.9, "handbag")]), flat=False)
    assert r.ok and r.alpha[485, 450] == 255 and r.info.get("recovered_parts"), r.info


def test_bag_keeps_small_parts_sam_missed_and_drops_big_foreign_chunks():
    # SAM 3 no ve un detalle fino pegado al bolso: por debajo del 1 % del área se conserva.
    # Un trozo grande fuera del producto (caja, mesa) sí se quita.
    bag = brect((200, 500, 700, 900))
    base = bag.astype(np.uint8) * 255
    base[350:500, 442:450] = 255                          # tira fina (0,6 % del área)
    base[500:900, 700:800] = 255                          # franja de caja pegada al lateral (20 %)
    r = pp.analyze(base, None, scene([inst(bag, 0.9, "handbag")]), flat=False)
    assert r.ok and r.alpha[400, 446] == 255, r.info
    assert r.alpha[700, 750] == 0 and r.info.get("clipped", 0) > 0.12, r.info


def test_refine_band_smooths_sam_staircase():
    # Borde en escalera (máscara SAM ampliada) sobre un borde real recto: el filtro lo alinea.
    rgb = photo()
    base = rect((150, 150, 750, 1050))
    jag = brect((150, 150, 750, 1050))
    for y in range(150, 1050, 40):
        jag[y:y + 20, 730:770] = True                     # dientes que salen 20 px fuera del libro real
    added = jag & ~(base >= pp.SOLID)
    alpha = base.copy()
    alpha[added] = 255
    out = pp.refine_band(alpha, rgb, added, radius=16)
    outside_before = int((alpha[:, 752:770] >= pp.SOLID).sum())
    outside_after = int((out[:, 752:770] >= pp.SOLID).sum())
    assert outside_after < outside_before * 0.3, (outside_before, outside_after)
    assert (out[200:1000, 200:700] == 255).all(), "el interior no se toca"


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
