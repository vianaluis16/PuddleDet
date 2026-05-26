import os
import cv2
import torch
import numpy as np
from pathlib import Path
from mapear_poca import gerar_mapa_poca

import sys
# Adiciona o diretório do projeto ao path para importar o modelo
sys.path.insert(0, r"c:\Users\luisv\PycharmProjects\PuddleDet")
from rau_fcn.model import FCN8sRAU

# Caminhos do seu pipeline
PASTA_ENTRADA = r"c:\Users\luisv\PycharmProjects\PuddleDet\imagensNovas_Originais"
PASTA_SAIDA = r"c:\Users\luisv\PycharmProjects\PuddleDet\resultados_com_mapa"
MODELO_PATH = r"c:\Users\luisv\PycharmProjects\PuddleDet\runs\rau_fcn\puddle1000_baseline\best.pt"

def nmea_para_decimal(nmea_lat, direcao_lat, nmea_lon, direcao_lon):
    """Converte o formato NMEA do GPS (graus e minutos) para graus decimais."""
    lat_deg = int(nmea_lat[:2])
    lat_min = float(nmea_lat[2:])
    lat_dec = lat_deg + (lat_min / 60.0)
    if direcao_lat == 'S': lat_dec = -lat_dec

    lon_deg = int(nmea_lon[:2])
    lon_min = float(nmea_lon[2:])
    lon_dec = lon_deg + (lon_min / 60.0)
    if direcao_lon == 'W': lon_dec = -lon_dec
    
    return lat_dec, lon_dec

# Vamos pegar a primeira coordenada real do seu log_filtrado.txt
# Linha 1 do log: NMEAGGA 2 172443.600000 2014.4062945 S 4014.6997404 W
# Convertendo para decimal para a API funcionar:
LAT_REAL, LON_REAL = nmea_para_decimal("2014.4062945", "S", "4014.6997404", "W")

def preparar_tensor(img_bgr):
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
    print(f"Carregando modelo FCN-8s...\n")
    
    ckpt = torch.load(MODELO_PATH, map_location=device, weights_only=False)
    model = FCN8sRAU(num_classes=2, use_rau=False, rau_mode="light", head_dim=256, dropout=0.15)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    todas_imagens = [os.path.join(PASTA_ENTRADA, f) for f in os.listdir(PASTA_ENTRADA) 
                     if f.endswith(('.png', '.jpg'))]

    with torch.no_grad():
        for caminho in todas_imagens:
            nome_arquivo = Path(caminho).name
            print(f"Processando: {nome_arquivo}")
            img_bgr = cv2.imread(caminho)
            if img_bgr is None: continue
            
            # 1. IA analisa a imagem
            img_tensor = preparar_tensor(img_bgr).to(device)
            logits = model(img_tensor)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            # Verifica se há poça (se tiver pelo menos 1 pixel predito como classe 1)
            tem_poca = np.any(preds == 1)
            
            if tem_poca:
                print(" -> [ALERTA] POÇA DETECTADA! Acionando módulo GPS e Mapa...")
                
                # Salva a imagem da câmera com a poça destacada
                mask_color = np.zeros_like(img_bgr)
                mask_color[preds == 1] = [0, 0, 255] # Vermelho em BGR
                overlay = img_bgr.copy()
                overlay[preds == 1] = overlay[preds == 1] * 0.5 + mask_color[preds == 1] * 0.5
                cv2.imwrite(os.path.join(PASTA_SAIDA, f"camera_{nome_arquivo}"), overlay)
                
                # 2. Gerar Mapa (usamos a coordenada REAL do log convertida)
                caminho_mapa = os.path.join(PASTA_SAIDA, f"mapa_{nome_arquivo}")
                sucesso = gerar_mapa_poca(LAT_REAL, LON_REAL, caminho_mapa)
                if sucesso:
                    print(f" -> Mapa salvo em: {caminho_mapa}\n")
            else:
                print(" -> Via limpa. Nenhuma poça detectada.\n")

if __name__ == "__main__":
    main()
