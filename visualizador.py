import cv2
import numpy as np
import os
import sys


def extrair_path_img(log_filt, camera_alvo):
    """
    Lê o arquivo de log já filtrado e pega os caminhos das imagens para a câmera especificada.

    Argumentos:
        log_filt (str): Caminho para o arquivo log_filtrado.txt
        camera_alvo (str): O nome da câmera (ex: "CAMERA2" ou "CAMERA3")
    Returns:
        list: lista com os caminhos das imagens
    """
    caminhos = []
    if not os.path.exists(log_filt):
        print(f"Erro: Arquivo de log '{log_filt}' não encontrado.")
        return caminhos

    try:

        with open(log_filt, "r", encoding='utf-8') as f:
            for linha in f:
                # filtra apenas pela câmera do argumento
                if linha.startswith(camera_alvo):
                    partes = linha.split()
                    if len(partes) >= 2:
                        caminhos.append(partes[1])
        return caminhos
    except Exception as e:
        print(f"ERRO ao ler o arquivo de log '{log_filt}': {e}")
        return []


def exibir_imagem(path_img, l = 640, h = 480):
    """
    Lê um arquivo de imagem RAW, converte e exibe.
    """
    try:

        with open(path_img, "rb") as arquivo_img:
            dados_img = arquivo_img.read()
        # Converte os dados para imagem

        # O 'except' abaixo vai pegar um 'ValueError' se o arquivo for corrupto e o reshape falhar.
        imagem = np.frombuffer(dados_img, dtype=np.uint8).reshape((h, l, 3))
        cv2.imshow("Visualizador PuddleDet", imagem)
        return True

    except FileNotFoundError:
        # Caso o log aponte para um arquivo que não existe
        print(f"\nAVISO: Arquivo não encontrado: {path_img}. Pulando...")
        return False
    except Exception as e:
        # Pega qualquer outro erro
        print(f"\nERRO ao exibir {path_img}: {e}. Pulando...")
        return False

def main():
    # --- LENDO ARGUMENTOS DO TERMINAL ---
    if len(sys.argv) != 3:
        print("=" * 60)
        print("ERRO: Modo de usar incorreto.")
        print("Você deve passar 2 argumentos: o caminho da pasta de dados e a câmera.")
        print("\nExemplo no terminal:")
        print(r'  python visualizador.py "C:\caminho\para\audit_20250701_4" "CAMERA2"')
        print("=" * 60)
        return

    pasta_audit = sys.argv[1]  # Argumento 1: Caminho da pasta
    camera_alvo = sys.argv[2]  # Argumento 2: Nome da câmera

    # Validação dos argumentos
    if not os.path.isdir(pasta_audit):
        print(f"ERRO: O caminho da pasta não foi encontrado: {pasta_audit}")
        return

    if camera_alvo not in ["CAMERA2", "CAMERA3"]:
        print(f"ERRO: Nome da câmera inválido: '{camera_alvo}'. Use 'CAMERA2' ou 'CAMERA3'.")
        return

    print(f"Pasta de dados selecionada: {pasta_audit}")
    print(f"Filtrando apenas por: {camera_alvo}")

    # 2. Carregamos os caminhos das imagens do log
    log = "log_filtrado.txt"
    # Passa a câmera-alvo para a função de extração
    caminhos_do_log = extrair_path_img(log, camera_alvo)

    if not caminhos_do_log:
        print(f"Nenhuma imagem encontrada para '{camera_alvo}' no log.")
        return

    print(f"Encontradas {len(caminhos_do_log)} imagens no log para {camera_alvo}.")

    # 3. Lógica de mapeamento de caminhos
    # Usa o nome da pasta (basename) para construir o prefixo
    prefixo_nome_log = os.path.basename(os.path.normpath(pasta_audit)) + ".txt_"
    print(f"Aplicando padrão de caminho (prefixo: '{prefixo_nome_log}')")

    funcao_caminho_correto = lambda p: os.path.join(
        pasta_audit,
        p.replace('_camera2', prefixo_nome_log + 'camera2', 1)
        .replace('_camera3', prefixo_nome_log + 'camera3', 1)
        .replace('/', os.sep)
    )
    caminhos_completos = [funcao_caminho_correto(p) for p in caminhos_do_log]

    # 4. Iniciamos o visualizador
    print("\n--- Iniciando Visualizador ---")
    print("Pressione qualquer tecla para visualizar a próxima imagem.")
    print("Pressione 'ESC' para sair.")
    print("---------------------------------")

    indice = 0
    imagens_boas = 0
    while indice < len(caminhos_completos):
        caminho_completo = caminhos_completos[indice]
        print(f"\rVerificando imagem {indice + 1}/{len(caminhos_completos)}...", end="")

        if exibir_imagem(caminho_completo):
            imagens_boas += 1
            print(f"\rMostrando Imagem VÁLIDA #{imagens_boas} (Arquivo {indice + 1}/{len(caminhos_completos)})")
            print(f"Caminho: {caminho_completo}")

            tecla = cv2.waitKey(0)
            if tecla == 27:  # ESC
                print("\nSaindo...")
                break
            else:
                indice += 1
        else:
            indice += 1

    cv2.destroyAllWindows()
    print(f"\nVisualizador encerrado.")


if __name__ == "__main__":
    main()