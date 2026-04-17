"""
rodar_deteccao.py
Script para detecção de poças d'água usando modelo YOLO treinado.
"""

import os
import sys
from pathlib import Path

# --- CONFIGURAÇÃO ---
MODELO_PATH = "best_fine_tuned.pt"  # Modelo ajustado com anotações customizadas

# Qual câmera processar? Ajuste para corresponder ao convert_data.py
CAMERA = "camera2"  # << AJUSTE AQUI: "camera2" ou "camera3"

PASTA_ENTRADA = f"dataset_jpg_{CAMERA}"
PASTA_SAIDA = f"pocas_encontradas_{CAMERA}"

EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONFIDENCE_THRESHOLD = 0.25  # Aumentado - modelo fine-tuned é mais confiante


def main():
    # 1. Verificar se a biblioteca ultralytics está instalada
    try:
        from ultralytics import YOLO
    except ImportError:
        print("=" * 60)
        print("ERRO: Biblioteca 'ultralytics' não encontrada.")
        print("Instale com: pip install ultralytics")
        print("=" * 60)
        sys.exit(1)

    # 2. Verificar se o modelo existe
    if not os.path.exists(MODELO_PATH):
        print(f"ERRO: Modelo '{MODELO_PATH}' não encontrado.")
        print("Verifique se o arquivo best.pt está na pasta do projeto.")
        sys.exit(1)

    # 3. Verificar se a pasta de entrada existe
    if not os.path.isdir(PASTA_ENTRADA):
        print(f"ERRO: Pasta de entrada '{PASTA_ENTRADA}' não encontrada.")
        sys.exit(1)

    # 4. Criar pasta de saída se não existir
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    print(f"Pasta de saída: {PASTA_SAIDA}")

    # 5. Carregar o modelo YOLO
    print(f"\nCarregando modelo: {MODELO_PATH}")
    try:
        modelo = YOLO(MODELO_PATH)
        print("✓ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"ERRO ao carregar o modelo: {e}")
        sys.exit(1)

    # 6. Listar imagens na pasta de entrada
    arquivos = [
        f for f in os.listdir(PASTA_ENTRADA)
        if Path(f).suffix.lower() in EXTENSOES_VALIDAS
    ]
    
    if not arquivos:
        print(f"ERRO: Nenhuma imagem encontrada em '{PASTA_ENTRADA}'")
        sys.exit(1)

    # Ordenação natural para manter ordem correta
    arquivos.sort(key=lambda x: _natural_sort_key(x))
    
    total = len(arquivos)
    print(f"\n{'=' * 50}")
    print(f"Total de imagens a processar: {total}")
    print(f"{'=' * 50}\n")

    # 7. Processar cada imagem
    deteccoes_total = 0
    imagens_com_deteccao = 0

    for idx, nome_arquivo in enumerate(arquivos, start=1):
        caminho_entrada = os.path.join(PASTA_ENTRADA, nome_arquivo)
        caminho_saida = os.path.join(PASTA_SAIDA, nome_arquivo)

        try:
            # Executar inferência
            resultados = modelo.predict(
                source=caminho_entrada,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False  # Evita spam no console
            )

            # Obter resultado (primeira imagem, pois processamos uma por vez)
            resultado = resultados[0]
            
            # Contar detecções nesta imagem
            num_deteccoes = len(resultado.boxes)
            deteccoes_total += num_deteccoes
            
            if num_deteccoes > 0:
                imagens_com_deteccao += 1

            # Salvar imagem com bounding boxes
            img_plotada = resultado.plot()  # Retorna numpy array com BBs desenhadas
            
            # Usar OpenCV para salvar (resultado.plot() retorna BGR)
            import cv2
            cv2.imwrite(caminho_saida, img_plotada)

            # Print de progresso
            status = f"[{num_deteccoes} poça(s)]" if num_deteccoes > 0 else ""
            print(f"Processando {idx}/{total}: {nome_arquivo} {status}")

        except Exception as e:
            print(f"ERRO ao processar {nome_arquivo}: {e}")
            continue

    # 8. Relatório final
    print(f"\n{'=' * 50}")
    print("✓ PROCESSAMENTO CONCLUÍDO!")
    print(f"{'=' * 50}")
    print(f"  - Imagens processadas: {total}")
    print(f"  - Imagens com detecção: {imagens_com_deteccao}")
    print(f"  - Total de detecções: {deteccoes_total}")
    print(f"  - Resultados salvos em: {PASTA_SAIDA}/")
    print(f"{'=' * 50}")


def _natural_sort_key(texto):
    """
    Função auxiliar para ordenação natural.
    Faz com que 'img2.jpg' venha antes de 'img10.jpg'.
    """
    import re
    return [
        int(parte) if parte.isdigit() else parte.lower()
        for parte in re.split(r'(\d+)', texto)
    ]


if __name__ == "__main__":
    main()
