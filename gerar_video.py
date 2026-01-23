"""
gerar_video.py
Script para gerar vídeo MP4 a partir das imagens processadas.
Projeto PuddleDet - UFES
"""

import cv2
import os
import sys
import glob
import re
from pathlib import Path

# --- CONFIGURAÇÃO ---
PASTA_IMAGENS = "pocas_encontradas_visual"
ARQUIVO_SAIDA = "resultado_deteccao.mp4"
FPS = 30
EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_sort_key(texto):
    """
    Função para ordenação natural (natural sorting).
    Garante que 'img2.jpg' venha antes de 'img10.jpg'.
    Isso evita que o vídeo fique 'pulando' no tempo.
    """
    return [
        int(parte) if parte.isdigit() else parte.lower()
        for parte in re.split(r'(\d+)', texto)
    ]


def main():
    # 1. Verificar se a pasta de imagens existe
    if not os.path.isdir(PASTA_IMAGENS):
        print(f"ERRO: Pasta '{PASTA_IMAGENS}' não encontrada.")
        print("Execute primeiro o script 'rodar_deteccao.py'.")
        sys.exit(1)

    # 2. Listar imagens válidas
    arquivos = [
        f for f in os.listdir(PASTA_IMAGENS)
        if Path(f).suffix.lower() in EXTENSOES_VALIDAS
    ]

    if not arquivos:
        print(f"ERRO: Nenhuma imagem encontrada em '{PASTA_IMAGENS}'")
        sys.exit(1)

    # 3. Ordenar naturalmente (IMPORTANTE para evitar flicker)
    arquivos.sort(key=natural_sort_key)
    
    total = len(arquivos)
    print(f"Encontradas {total} imagens para o vídeo.")
    print(f"Ordenação natural aplicada ✓")

    # 4. Ler primeira imagem para obter dimensões
    primeira_img_path = os.path.join(PASTA_IMAGENS, arquivos[0])
    primeira_img = cv2.imread(primeira_img_path)
    
    if primeira_img is None:
        print(f"ERRO: Não foi possível ler a imagem: {primeira_img_path}")
        sys.exit(1)

    altura, largura = primeira_img.shape[:2]
    print(f"Dimensões do vídeo: {largura}x{altura}")
    print(f"FPS: {FPS}")
    print(f"Codec: mp4v (MP4)")

    # 5. Configurar o VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(ARQUIVO_SAIDA, fourcc, FPS, (largura, altura))

    if not video_writer.isOpened():
        print("ERRO: Não foi possível criar o arquivo de vídeo.")
        print("Verifique se você tem permissão de escrita na pasta.")
        sys.exit(1)

    print(f"\n{'=' * 50}")
    print("Gerando vídeo...")
    print(f"{'=' * 50}")

    # 6. Processar cada imagem e adicionar ao vídeo
    for idx, nome_arquivo in enumerate(arquivos, start=1):
        caminho_img = os.path.join(PASTA_IMAGENS, nome_arquivo)
        
        try:
            frame = cv2.imread(caminho_img)
            
            if frame is None:
                print(f"AVISO: Imagem corrompida ou inválida: {nome_arquivo}")
                continue

            # Redimensionar se necessário (garante consistência)
            if frame.shape[:2] != (altura, largura):
                frame = cv2.resize(frame, (largura, altura))

            video_writer.write(frame)

            # Print de progresso a cada 50 frames
            if idx % 50 == 0 or idx == total:
                porcentagem = (idx / total) * 100
                print(f"Progresso: {idx}/{total} ({porcentagem:.1f}%)")

        except Exception as e:
            print(f"ERRO ao processar {nome_arquivo}: {e}")
            continue

    # 7. Finalizar
    video_writer.release()

    # Verificar se o arquivo foi criado
    if os.path.exists(ARQUIVO_SAIDA):
        tamanho_mb = os.path.getsize(ARQUIVO_SAIDA) / (1024 * 1024)
        duracao_seg = total / FPS
        
        print(f"\n{'=' * 50}")
        print("✓ VÍDEO GERADO COM SUCESSO!")
        print(f"{'=' * 50}")
        print(f"  - Arquivo: {ARQUIVO_SAIDA}")
        print(f"  - Tamanho: {tamanho_mb:.2f} MB")
        print(f"  - Duração: {duracao_seg:.1f} segundos")
        print(f"  - Frames: {total}")
        print(f"  - FPS: {FPS}")
        print(f"{'=' * 50}")
    else:
        print("ERRO: O arquivo de vídeo não foi criado.")
        sys.exit(1)


if __name__ == "__main__":
    main()