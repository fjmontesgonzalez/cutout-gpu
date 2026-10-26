"""Descarga los pesos dentro de la imagen durante `docker build` (revisiones fijadas).

Todos sin gating y con licencia comercial permisiva:
  BiRefNet_dynamic (MIT) · Grounding DINO tiny (Apache-2.0) · SAM 2.1 large (Apache-2.0)
"""
import os

from huggingface_hub import snapshot_download

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
MODELS = {
    "birefnet": ("ZhengPeng7/BiRefNet_dynamic", "280306042f57b7a33854319da62fd86aaa89ec4c",
                 ["*.py", "*.json", "model.safetensors"]),
    "gdino": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71",
              ["*.json", "*.txt", "model.safetensors"]),
    "sam2": ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610",
             ["*.json", "model.safetensors"]),
}

for name, (repo, rev, patterns) in MODELS.items():
    path = snapshot_download(repo, revision=rev, local_dir=os.path.join(MODELS_DIR, name),
                             allow_patterns=patterns)
    print(name, "->", path, sorted(f for f in os.listdir(path) if not f.startswith(".")))
