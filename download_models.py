"""Descarga los pesos dentro de la imagen durante `docker build` (revisiones fijadas) y
los guarda en fp16 para que la imagen pese menos (arranque en frío más corto).

Modelos (uso personal, no comercial):
  BiRefNet_dynamic (MIT)         modelo base por defecto, resolución nativa
  RMBG-2.0 (CC BY-NC 4.0)        modelo base alternativo: es el modelo de Bria, para comparar
  BEN2 base (MIT)                modelo base alternativo, ligero
  SAM 3 (SAM License)            producto / manos / soportes por texto

SAM 3 y RMBG-2.0 están "gated" en sus repos oficiales (hay que aceptar condiciones con una
cuenta). Para no depender de un token en el build se usan, en ese orden, el repo oficial
(si hay HF_TOKEN) y copias públicas cuyo `model.safetensors` es BYTE A BYTE el oficial:
el SHA-256 del fichero se comprueba siempre contra el del repo oficial (leído de su API el
2026-10-26) y, si no coincide, el build falla. El token llega por HF_TOKEN o por el fichero
HF_TOKEN_FILE (secreto de BuildKit); nunca se guarda en la imagen.
CUTOUT_MODELS limita qué se descarga (p.ej. "birefnet,sam3" para la imagen de producción).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

from huggingface_hub import snapshot_download

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
WANTED = [m.strip() for m in (os.environ.get("CUTOUT_MODELS", "birefnet,rmbg2,ben2,sam3")).split(",") if m.strip()]

# name -> (sha256 oficial de model.safetensors, patrones, [(repo, revisión, gated), ...])
MODELS = {
    "birefnet": (
        None,   # ya en fp16 y sin gate: se fía de la revisión fijada
        ["*.py", "*.json", "model.safetensors"],
        [("ZhengPeng7/BiRefNet_dynamic", "280306042f57b7a33854319da62fd86aaa89ec4c", False)],
    ),
    "rmbg2": (
        "566ed80c3d95f87ada6864d4cbe2290a1c5eb1c7bb0b123e984f60f76b02c3a7",
        ["*.py", "config.json", "preprocessor_config.json", "model.safetensors"],
        [("briaai/RMBG-2.0", "2fd7b4b0f2393ddb39a267fb62df6c010928a6c6", True),
         ("camenduru/RMBG-2.0", "00e2d9a92347d951e65d1c73f44fce8e04095189", False),
         ("1038lab/RMBG-2.0", "1cd4787601caeb4c8e826dba7ea8e2163b5208df", False)],
    ),
    "ben2": (
        None,
        ["BEN2.py", "config.json", "model.safetensors"],
        [("PramaLLC/BEN2", "e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f", False)],
    ),
    "sam3": (
        "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a",
        ["*.json", "*.txt", "LICENSE", "model.safetensors"],
        [("facebook/sam3", "3c879f39826c281e95690f02c7821c4de09afae7", True),
         ("JobHarvest/sam3", "90ab711c3b98c6d012c9ebf31f2b575b47b614bc", False),
         ("AnantP78/sam3_pt", "39166e439ec83d600c9caf80ea3dbf21fc29ed6a", False)],
    ),
}


def _token() -> str | None:
    tok = (os.environ.get("HF_TOKEN") or "").strip()
    path = os.environ.get("HF_TOKEN_FILE", "/run/secrets/hf_token")
    if not tok and os.path.exists(path):
        tok = open(path).read().strip()
    return tok or None


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_fp16(path: str) -> str:
    """Convierte los tensores float32 de model.safetensors a float16 (los enteros se dejan)."""
    import torch
    from safetensors.torch import load_file, save_file
    src = os.path.join(path, "model.safetensors")
    state = load_file(src)
    converted = 0
    for k, v in list(state.items()):
        if v.dtype == torch.float32:
            state[k] = v.to(torch.float16)
            converted += 1
    if converted:
        save_file(state, src, metadata={"format": "pt"})
    return "{} tensores a fp16".format(converted)


def _fetch(name: str, token: str | None) -> tuple[str, str, str] | None:
    expected, patterns, sources = MODELS[name]
    dest = os.path.join(MODELS_DIR, name)
    for repo, rev, gated in sources:
        if gated and not token:
            print("{}: {} es gated y no hay HF_TOKEN; se prueba la siguiente fuente".format(name, repo))
            continue
        try:
            path = snapshot_download(repo, revision=rev, local_dir=dest, allow_patterns=patterns,
                                     token=token if gated else None)
        except Exception as exc:  # noqa: BLE001
            print("{}: {}@{} falló ({}: {})".format(name, repo, rev[:12], type(exc).__name__, str(exc)[:160]))
            shutil.rmtree(dest, ignore_errors=True)
            continue
        weights = os.path.join(path, "model.safetensors")
        if not os.path.exists(weights):
            print("{}: {} no trae model.safetensors".format(name, repo))
            shutil.rmtree(dest, ignore_errors=True)
            continue
        digest = _sha256(weights)
        if expected and digest != expected:
            print("{}: SHA-256 de {} NO coincide con el oficial ({} != {}); se descarta".format(
                name, repo, digest[:16], expected[:16]))
            shutil.rmtree(dest, ignore_errors=True)
            continue
        return path, repo, digest
    return None


def main() -> int:
    token = _token()
    manifest = {}
    for name in WANTED:
        if name not in MODELS:
            print("modelo desconocido:", name)
            return 2
        got = _fetch(name, token)
        if got is None:
            print("ERROR: no se pudo obtener {} de ninguna fuente".format(name))
            if name in ("sam3",) or not any(k in manifest for k in ("birefnet", "rmbg2", "ben2")):
                return 1
            continue
        path, repo, digest = got
        note = _to_fp16(path) if name != "birefnet" else "ya en fp16"
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs
                   if ".cache" not in dp)
        manifest[name] = {"repo": repo, "sha256_model": digest, "bytes_fp16": size, "note": note}
        print(name, "->", path, "desde", repo, "{:.0f} MB".format(size / 1e6), note,
              sorted(f for f in os.listdir(path) if not f.startswith(".")))
    if "sam3" not in manifest:
        print("ERROR: SAM 3 no se ha descargado; el servidor v3 no arranca sin él")
        return 1
    with open(os.path.join(MODELS_DIR, "MANIFEST.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
