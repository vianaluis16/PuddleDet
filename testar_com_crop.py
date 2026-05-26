import os
import cv2
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, r"c:\Users\luisv\PycharmProjects\PuddleDet")
from rau_fcn.model import FCN8sRAU

PASTA_ENTRADA = r"c:\Users\luisv\PycharmProjects\PuddleDet\imagensNovas_Originais"
PASTA_SAIDA = r"c:\Users\luisv\PycharmProjects\PuddleDet\segmentacao_baseline_crop"
EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODELO_PATH = r"c:\Users\luisv\PycharmProjects\PuddleDet\runs\rau_fcn\puddle1000_baseline\best.pt"

IMG_W = 320
IMG_H = 180

def cortar_centro_inferior(img):
    """Corta uma região de 320x180 do centro-inferior da imagem original."""
    h, w = img.shape[:2]
    
    if w < IMG_W or h < IMG_H:
        # Se a imagem for menor que o crop, preenche com preto
        img_padded = np.zeros((max(h, IMG_H), max(w, IMG_W), 3), dtype=np.uint8)
        img_padded[0:h, 0:w] = img
        img = img_padded
        h, w = img.shape[:2]

    # Calcular coordenadas (parte inferior e centro horizontal)
    y_start = h - IMG_H  # Pegar a partir da base (chão)
    y_end = h
    x_start = (w - IMG_W) // 2
    x_end = x_start + IMG_W
    
    crop = img[y_start:y_end, x_start:x_end]
    return crop

def preparar_tensor(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img_tensor = img_tensor.permute(2, 0, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.unsqueeze(0)

def main():
    if not os.path.exists(MODELO_PATH):
        print(f"Modelo não encontrado: {MODELO_PATH}")
        return

    os.makedirs(PASTA_SAIDA, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    print("Carregando modelo BASELINE ORIGINAL...")
    ckpt = torch.load(MODELO_PATH, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    
    use_rau = False
    rau_mode = ckpt_args.get("rau_mode", "light")
    head_dim = ckpt_args.get("head_dim", 256)
    dropout = ckpt_args.get("dropout", 0.15)
    
    model = FCN8sRAU(num_classes=2, use_rau=use_rau, rau_mode=rau_mode, head_dim=head_dim, dropout=dropout)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    todas_imagens = []
    for root, _, files in os.walk(PASTA_ENTRADA):
        for f in files:
            if Path(f).suffix.lower() in EXTENSOES_VALIDAS:
                todas_imagens.append(os.path.join(root, f))
                
    print(f"\nCortando área de interesse (Crop {IMG_W}x{IMG_H}) em {len(todas_imagens)} imagens originais...")

    imagens_com_poca = 0

    with torch.no_grad():
        for n, caminho in enumerate(todas_imagens):
            img_bgr = cv2.imread(caminho)
            if img_bgr is None: continue
            
            # Cortar a área de interesse
            img_crop = cortar_centro_inferior(img_bgr)
            
            img_tensor = preparar_tensor(img_crop).to(device)
            logits = model(img_tensor)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            if 1 in preds:
                imagens_com_poca += 1
                
            mask_color = np.zeros_like(img_crop)
            mask_color[preds == 1] = [255, 128, 0] # Azul Claro
            
            alpha = 0.5
            mask_bool = preds == 1
            overlay = img_crop.copy()
            overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + mask_color[mask_bool] * alpha

            nome_arquivo = Path(caminho).name
            caminho_saida = os.path.join(PASTA_SAIDA, f"crop_baseline_{nome_arquivo}")
            cv2.imwrite(caminho_saida, overlay)
            
    print("\n--- RESUMO SEGMENTAÇÃO CROP ---")
    print(f"Imagens processadas: {len(todas_imagens)}")
    print(f"Imagens com detecção de poça na área central recortada: {imagens_com_poca}")
    print(f"Resultados salvos em: {PASTA_SAIDA}")

if __name__ == "__main__":
    main()
