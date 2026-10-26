"""Comprueba en CPU que cada modelo base de la imagen carga y produce un alfa (sin SAM 3).

    python check_models.py            # todos los presentes en MODELS_DIR
    python check_models.py rmbg2 ben2

Lo ejecuta la CI para los modelos que la prueba de humo no carga (allí sólo va BiRefNet,
por la RAM del runner). Imagen sintética pequeña: mide que el modelo funciona, no su calidad.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUTOUT_BASE_MODELS", ",".join(sys.argv[1:]) if len(sys.argv) > 1 else "")

import server  # noqa: E402


def main() -> int:
    import torch
    wanted = sys.argv[1:] or [n for n in server.BASE_SPECS if os.path.isdir(os.path.join(server.MODELS_DIR, server.BASE_SPECS[n]["dir"]))]
    failures = 0
    server._M.update(torch=torch, device="cpu", half=None,
                     mean=torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1),
                     std=torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1), bases={})
    rgb = server._synthetic_photo(480, 640)
    for name in wanted:
        spec = server.BASE_SPECS[name]
        path = os.path.join(server.MODELS_DIR, spec["dir"])
        if not os.path.isdir(path):
            print("{:9} ausente en {} (no va en esta imagen); se omite".format(name, server.MODELS_DIR))
            continue
        t = time.time()
        try:
            loader = server._load_ben2 if spec["mode"] == "ben2" else server._load_birefnet
            model, how = loader(path, "cpu", None)
            server._M["bases"][name] = {"model": model, "mode": spec["mode"], "loader": how}
            alpha, info = server._base_alpha(name, rgb)
            fg = float((alpha >= 192).mean())
            bg = float((alpha <= 8).mean())
            ok = alpha.shape == rgb.shape[:2] and fg > 0.05 and bg > 0.05
            print("{:9} {:8} carga={} {} fondo={:.2f} producto={:.2f} {:.1f}s {}".format(
                name, "OK" if ok else "FAIL", how, info["work"], bg, fg, time.time() - t, alpha.shape))
            failures += 0 if ok else 1
            del model
            server._M["bases"].pop(name, None)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("{:9} ERROR {}: {}".format(name, type(exc).__name__, str(exc)[:300]))
    print("OK" if not failures else "FALLOS: {}".format(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
