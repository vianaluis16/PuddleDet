"""
convert_data.py
Converte imagens RAW (.image) da pasta de auditoria para JPG.
Processa cada câmera separadamente para evitar mistura de frames.
Projeto PuddleDet - UFES
"""

import cv2
import numpy as np
import os
import sys
import re

# --- CONFIGURAÇÃO ---
PASTA_AUDIT = r"C:\Users\luisv\Downloads\audit_20250701_4"
PASTA_SAIDA_BASE = "dataset_jpg"
WIDTH, HEIGHT = 640, 480

# Qual câmera processar? Opções: "camera2", "camera3", ou "ambas" (pastas separadas)
CAMERA_ALVO = "camera2"  # << AJUSTE AQUI

MAX_IMAGENS = None  # None = todas, ou coloque um número para limitar


def natural_sort_key(texto):
    """Ordenação natural para manter ordem temporal."""
    return [
        int(parte) if parte.isdigit() else parte.lower()
        for parte in re.split(r'(\d+)', texto)
    ]


def buscar_imagens_raw(pasta_raiz, filtro_camera=None):
    """
    Busca recursivamente todos os arquivos .image na pasta.
    Retorna lista ordenada de caminhos completos.
    """
    imagens = []
    
    print(f"Buscando imagens em: {pasta_raiz}")
    
    for root, dirs, files in os.walk(pasta_raiz):
        for arquivo in files:
            if arquivo.endswith(".image"):
                # Aplicar filtro de câmera se especificado
                if filtro_camera:
                    if filtro_camera.lower() not in arquivo.lower():
                        continue
                
                caminho_completo = os.path.join(root, arquivo)
                imagens.append(caminho_completo)
    
    # Ordenar naturalmente para manter ordem temporal
    imagens.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    return imagens


def converter_raw_para_jpg(caminho_raw, caminho_saida):
    """
    Lê uma imagem RAW e salva como JPG.
    Retorna True se sucesso, False se falha.
    """
    try:
        # Ler dados binários
        with open(caminho_raw, "rb") as f:
            raw_data = f.read()
        
        # Verificar tamanho esperado
        tamanho_esperado = WIDTH * HEIGHT * 3
        if len(raw_data) < tamanho_esperado:
            return False
        
        # Pegar apenas os últimos bytes (caso tenha header)
        raw_data = raw_data[-tamanho_esperado:]
        
        # Converter para numpy array e reshape
        img = np.frombuffer(raw_data, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
        
        # RAW já está em BGR (formato OpenCV), salvar diretamente
        cv2.imwrite(caminho_saida, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True
        
    except Exception as e:
        return False


def processar_camera(camera_nome, pasta_saida):
    """Processa uma câmera específica."""
    
    # Criar pasta de saída
    os.makedirs(pasta_saida, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Processando: {camera_nome.upper()}")
    print(f"Pasta de saída: {os.path.abspath(pasta_saida)}")
    print(f"{'=' * 60}")
    
    # Buscar imagens desta câmera
    imagens_raw = buscar_imagens_raw(PASTA_AUDIT, camera_nome)
    
    if not imagens_raw:
        print(f"Nenhuma imagem encontrada para {camera_nome}")
        return 0
    
    total = len(imagens_raw)
    
    # Aplicar limite se especificado
    if MAX_IMAGENS and MAX_IMAGENS < total:
        imagens_raw = imagens_raw[:MAX_IMAGENS]
        total = MAX_IMAGENS
    
    print(f"Total de imagens: {total}")
    
    # Converter cada imagem
    sucessos = 0
    
    for idx, caminho_raw in enumerate(imagens_raw, start=1):
        nome_saida = f"img_{idx:05d}.jpg"
        caminho_saida = os.path.join(pasta_saida, nome_saida)
        
        if converter_raw_para_jpg(caminho_raw, caminho_saida):
            sucessos += 1
        
        if idx % 200 == 0 or idx == total:
            print(f"Progresso: {idx}/{total} ({(idx/total)*100:.1f}%)")
    
    print(f"✓ {sucessos} imagens convertidas para {pasta_saida}")
    return sucessos


def main():
    # 1. Verificar pasta de entrada
    if not os.path.isdir(PASTA_AUDIT):
        print(f"ERRO: Pasta de entrada não encontrada: {PASTA_AUDIT}")
        sys.exit(1)
    
    print(f"Fonte: {PASTA_AUDIT}")
    print(f"Câmera selecionada: {CAMERA_ALVO}")
    
    # 2. Processar conforme configuração
    if CAMERA_ALVO == "ambas":
        # Processar cada câmera em pasta separada
        total1 = processar_camera("camera2", f"{PASTA_SAIDA_BASE}_camera2")
        total2 = processar_camera("camera3", f"{PASTA_SAIDA_BASE}_camera3")
        print(f"\n{'=' * 60}")
        print("✓ AMBAS AS CÂMERAS PROCESSADAS!")
        print(f"  - Camera2: {total1} imagens em {PASTA_SAIDA_BASE}_camera2/")
        print(f"  - Camera3: {total2} imagens em {PASTA_SAIDA_BASE}_camera3/")
    else:
        # Processar apenas uma câmera
        pasta_saida = f"{PASTA_SAIDA_BASE}_{CAMERA_ALVO}"
        total = processar_camera(CAMERA_ALVO, pasta_saida)
        print(f"\nPróximo passo: python rodar_deteccao.py")
    
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()