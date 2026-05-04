import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import glob, os, sys
sys.path.insert(0, ".")
from taming.models.vqgan import VQModel

CONFIG_PATH  = "configs/astro_vqgan.yaml"
CKPT_PATH    = "logs/2026-05-03T19-37-20_astro_vqgan/checkpoints/last.ckpt"  # el más completo
INPUT_DIR    = "data/astro_dataset/test"
OUTPUT_DIR   = "reconstrucciones"
IMAGE_SIZE   = 256

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Checkpoint: {CKPT_PATH}")
imagenes = glob.glob(f"{INPUT_DIR}/*.jpg") + glob.glob(f"{INPUT_DIR}/*.png")
print(f"Imágenes encontradas: {len(imagenes)}")

# Cargar modelo
config = OmegaConf.load(CONFIG_PATH)
model = VQModel(**config.model.params)
sd = torch.load(CKPT_PATH, map_location="cpu")["state_dict"]
model.load_state_dict(sd, strict=False)
model.eval().cuda()

for img_path in imagenes:
    img = Image.open(img_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    x = torch.tensor(np.array(img)).permute(2,0,1).float()
    x = (x / 127.5 - 1.0).unsqueeze(0).cuda()

    with torch.no_grad():
        xrec, _ = model(x)

    xrec = xrec.squeeze(0).permute(1,2,0).cpu().numpy()
    xrec = ((xrec + 1) * 127.5).clip(0, 255).astype(np.uint8)

    original = np.array(img)
    comparacion = np.concatenate([original, xrec], axis=1)
    nombre = os.path.basename(img_path)
    Image.fromarray(comparacion).save(f"{OUTPUT_DIR}/rec_{nombre}")
    print(f"Guardado: rec_{nombre}")

print(f"\nListo. Reconstrucciones en: {OUTPUT_DIR}/")