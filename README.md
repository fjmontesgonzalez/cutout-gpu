# cutout-gpu

Servidor de recorte de producto (quitar fondo) para GPU, pensado para Alibaba Cloud
Function Compute (contenedor propio, puerto 9000). Devuelve un PNG RGBA del mismo
tamaño que la foto: sólo calcula el alfa, los píxeles RGB no se tocan.

Pipeline: BiRefNet_dynamic (MIT) a resolución nativa, Grounding DINO tiny (Apache-2.0)
para quitar lo que quede fuera de la caja del producto, y SAM 2.1 large (Apache-2.0)
sólo para rellenar máscaras mordidas o despiezadas dentro del casco del producto.

- `POST /` multipart `file` + `Authorization: Bearer <token>` → `image/png`
- `POST /initialize` carga y calienta los modelos (hook de inicialización de FC)
- `GET /health`

La imagen no contiene secretos. El servidor sólo conoce el SHA-256 del token
(`CUTOUT_TOKEN_SHA256`); sin esa variable rechaza todas las peticiones de recorte.
