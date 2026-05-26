import os
import cv2
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, r"c:\Users\luisv\PycharmProjects\PuddleDet")
from rau_fcn.model import FCN8sRAU

PASTA_ENTRADA = r"c:\Users\luisv\PycharmProjects\PuddleDet\imagensNovas_Originais"
PASTA_SAIDA = r"c:\Users\luisv\PycharmProjects\PuddleDet\segmentacao_baseline_fullres"
EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODELO_PATH = r"c:\Users\luisv\PycharmProjects\PuddleDet\runs\rau_fcn\puddle1000_baseline\best.pt"

def preparar_tensor(img_bgr):
    # O modelo FCN aceita qualquer tamanho!
    # Certificando que largura e altura sejam pares ou múltiplos de 32 se der problema no pooling,
    # mas o MaxPool2d no modelo usa ceil_mode=True, então lida bem com ímpares.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.unsqueeze(0)

def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt = torch.load(MODELO_PATH, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    
    model = FCN8sRAU(num_classes=2, use_rau=False, rau_mode="light", head_dim=256, dropout=0.15)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    todas_imagens = [os.path.join(root, f) for root, _, files in os.walk(PASTA_ENTRADA) 
                     for f in files if Path(f).suffix.lower() in EXTENSOES_VALIDAS]

    with torch.no_grad():
        for caminho in todas_imagens:
            img_bgr = cv2.imread(caminho)
            if img_bgr is None: continue
            
            img_tensor = preparar_tensor(img_bgr).to(device)
            logits = model(img_tensor)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            mask_color = np.zeros_like(img_bgr)
            mask_color[preds == 1] = [255, 128, 0] # Azul Claro
            
            overlay = img_bgr.copy()
            overlay[preds == 1] = overlay[preds == 1] * 0.5 + mask_color[preds == 1] * 0.5

            nome_arquivo = Path(caminho).name
            cv2.imwrite(os.path.join(PASTA_SAIDA, f"fullres_{nome_arquivo}"), overlay)

if __name__ == "__main__":
    main()
