# cutout-gpu (v3)

Servidor de recorte de producto (quitar fondo) para GPU, pensado para Alibaba Cloud
Function Compute (contenedor propio, puerto 9000). Devuelve un PNG RGBA del mismo
tamaño que la foto: sólo calcula el alfa, los píxeles RGB no se tocan. Cuando la foto
no sirve (producto cortado por el encuadre, mano que lo tapa, sin producto, portada
mordida) lo dice con un 422 en vez de devolver un recorte malo.

Dos modelos con papeles distintos:

- **Modelo base** (borde fino): `BiRefNet_dynamic` por defecto; `RMBG-2.0` (el modelo
  de Bria) y `BEN2` opcionales, para comparar o cambiar sin tocar código.
- **SAM 3** (qué es producto): instancias por texto ("book", "handbag", "strap"...),
  manos y soportes, con puntuación de presencia. Un paso del codificador visual por
  foto y un prompt por concepto.

`postproc.analyze` cruza los dos: rechaza lo irreparable y arregla lo reparable (fondo o
atril fuera del producto, huecos y zonas translúcidas en caras planas, mordiscos que SAM 3
ve enteros, herrajes que la base perdió), refinando los bordes añadidos con el RGB.

## API

- `POST /` multipart `file` (+ opcional `category`, `kind`, `model`, `debug=1`) con
  `Authorization: Bearer <token>`
  - `200 image/png` + cabeceras `X-Cutout-*` (`X-Cutout-Verdict: ok`)
  - `422 {"error":"rejected","reason":...}` la foto no sirve (mismas cabeceras)
- `GET /categories` categorías, peligros y umbrales activos
- `POST /initialize` carga y calienta los modelos (hook de inicialización de FC)
- `GET /health`

```bash
curl -H "Authorization: Bearer $TOKEN" -F "file=@bolso.jpg" -F "category=bag" \
     https://<endpoint>/ -o recorte.png -D -
```

Motivos de rechazo: `no_product`, `product_mismatch`, `product_cut:<lados>`,
`hand_occlusion`, `bitten`, `empty_mask`. Arreglos aplicados (`X-Cutout-Fixes`):
`clipped`, `snapped`, `support_removed`, `holes_filled`, `solidified`, `bite_filled`,
`rescued`, `recovered_parts`, `dropped_components`.

## Configuración (variables de entorno de la función, sin reconstruir la imagen)

| Variable | Qué hace |
|---|---|
| `CUTOUT_TOKEN_SHA256` | SHA-256 del token Bearer. Sin ella, todo recorte responde 503. |
| `CUTOUT_CATEGORY` | categoría por defecto (`book`). |
| `CUTOUT_CATEGORIES` | JSON: añade categorías o cambia `prompts`, `flat` y `params` (cualquier campo de `postproc.Params`). |
| `CUTOUT_HAZARDS` | JSON: prompts de `hand` y `support`. |
| `CUTOUT_BASE_MODEL` | modelo base de producción (`birefnet`, `rmbg2`, `ben2`). |
| `CUTOUT_BASE_MODELS` | lista de modelos base a cargar (por defecto, todos los presentes en la imagen). |
| `SAM3_DTYPE` | `auto` (fp16 con comprobación en el calentamiento y vuelta a fp32), `fp16`, `fp32`. |
| `SAM3_THRESHOLD` | umbral de recogida de instancias (0,2); cada papel aplica el suyo (`params`). |
| `BIREFNET_MAX_SIDE` | lado largo de trabajo de BiRefNet_dynamic (1536). |
| `CUTOUT_ALLOW_DEBUG` | `1` habilita `debug=1` (alfas de todos los modelos base + máscaras de SAM 3 en JSON). |

```json
{"shoe": {"prompts": ["shoe", "sneaker"]},
 "bag": {"prompts": ["handbag", "bag", "strap", "buckle"]},
 "book": {"params": {"bite_deficit": 0.05, "bottom_contact_max": 0.15}}}
```

## Imagen

`Dockerfile` (python 3.12, torch 2.9.1 cu128, transformers 5.20). Los pesos van dentro de
la imagen en fp16 (`download_models.py`). SAM 3 y RMBG-2.0 son repos "gated": el build
necesita un token de lectura de Hugging Face de una cuenta que haya aceptado sus
condiciones, pasado como secreto de BuildKit (`--secret id=hf_token,env=HF_TOKEN`); no
queda en la imagen. `--build-arg CUTOUT_MODELS=birefnet,sam3` construye la imagen de
producción sin los modelos base alternativos (~0,6 GB menos).

## Pruebas

- `python test_postproc.py` lógica de máscara con escenas sintéticas (sin modelos).
- `python test_server_fake.py` contrato HTTP con modelos falsos (sin modelos).
- `python smoke_test.py http://127.0.0.1:9000 <token>` contra un servidor real (CI lo
  ejecuta en CPU con `CUTOUT_BASE_MODELS=birefnet` y `SAM3_IMAGE_SIZE=560`).

## Seguridad

La imagen no contiene secretos. El servidor sólo conoce el SHA-256 del token; sin esa
variable rechaza todas las peticiones de recorte.
