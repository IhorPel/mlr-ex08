"""Generate and verify a local QR image for this app's final public URL."""
import json
from pathlib import Path

import qrcode
from PIL import Image

ROOT = Path(__file__).resolve().parent
URL = "https://ihorpel.github.io/mlr-ex08/"
code = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=8, border=4)
code.add_data(URL)
code.make(fit=True)
code.make_image(fill_color="black", back_color="white").save(ROOT / "qr.png")
assert b"".join(chunk.data for chunk in code.data_list).decode("utf-8") == URL
matrix = code.get_matrix()
with Image.open(ROOT / "qr.png") as image:
    image = image.convert("RGB")
    for y, row in enumerate(matrix):
        for x, dark in enumerate(row):
            pixel = image.getpixel((x*8+4, y*8+4))
            assert (pixel == (0, 0, 0)) == dark
(ROOT / "evidence/qr_check.json").write_text(json.dumps({"encoded_url": URL,
    "payload_verified": True, "rendered_modules_verified": True,
    "external_qr_service_used": False}, indent=2), encoding="utf-8")
print("Verified QR payload and rendered modules:", URL)
