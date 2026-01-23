"""
convert_data.py
Converte TODAS as imagens RAW (.image) da pasta de auditoria para JPG.
Projeto PuddleDet - UFES
"""

import cv2
import numpy as np
import os
import sys
import re

# --- CONFIGURAÇÃO ---
PASTA_AUDIT = r"C:\Users\luisv\Downloads\audit_20250701_4"
PASTA_SAIDA = "dataset_jpg"
WIDTH, HEIGHT = 640, 480

# Filtros opcionais (deixe None para converter TODAS as imagens)
CAMERA_FILTRO = None  # Opções: "camera2", "camera3", ou None para todas
MAX_IMAGENS = None    # Coloque um número para limitar, ou None para todas


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
        
        # Converter RGB -> BGR (formato OpenCV)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Salvar como JPG com qualidade 95%
        cv2.imwrite(caminho_saida, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True
        
    except Exception as e:
        return False


def main():
    # 1. Verificar pasta de entrada
    if not os.path.isdir(PASTA_AUDIT):
        print(f"ERRO: Pasta de entrada não encontrada: {PASTA_AUDIT}")
        sys.exit(1)
    
    # 2. Criar pasta de saída
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    print(f"Pasta de saída: {os.path.abspath(PASTA_SAIDA)}")
    
    # 3. Buscar todas as imagens RAW
    print("\nBuscando imagens RAW...")
    imagens_raw = buscar_imagens_raw(PASTA_AUDIT, CAMERA_FILTRO)
    
    if not imagens_raw:
        print("ERRO: Nenhuma imagem .image encontrada!")
        sys.exit(1)
    
    total = len(imagens_raw)
    
    # Aplicar limite se especificado
    if MAX_IMAGENS and MAX_IMAGENS < total:
        imagens_raw = imagens_raw[:MAX_IMAGENS]
        total = MAX_IMAGENS
        print(f"Limitado a {MAX_IMAGENS} imagens (de {len(imagens_raw)} disponíveis)")
    
    print(f"\n{'=' * 60}")
    print(f"Total de imagens a converter: {total}")
    print(f"{'=' * 60}\n")
    
    # 4. Converter cada imagem
    sucessos = 0
    falhas = 0
    
    for idx, caminho_raw in enumerate(imagens_raw, start=1):
        # Gerar nome do arquivo de saída (mantém ordem numérica)
        nome_saida = f"img_{idx:05d}.jpg"
        caminho_saida = os.path.join(PASTA_SAIDA, nome_saida)
        
        if converter_raw_para_jpg(caminho_raw, caminho_saida):
            sucessos += 1
        else:
            falhas += 1
        
        # Print de progresso a cada 100 imagens
        if idx % 100 == 0 or idx == total:
            porcentagem = (idx / total) * 100
            print(f"Progresso: {idx}/{total} ({porcentagem:.1f}%) - Sucessos: {sucessos} | Falhas: {falhas}")
    
    # 5. Relatório final
    print(f"\n{'=' * 60}")
    print("✓ CONVERSÃO CONCLUÍDA!")
    print(f"{'=' * 60}")
    print(f"  - Imagens convertidas: {sucessos}")
    print(f"  - Falhas: {falhas}")
    print(f"  - Pasta de saída: {os.path.abspath(PASTA_SAIDA)}")
    print(f"{'=' * 60}")
    print("\nPróximo passo: python rodar_deteccao.py")


if __name__ == "__main__":
    main()