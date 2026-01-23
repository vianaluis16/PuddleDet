import cv2
import numpy as np
import os

# --- CONFIGURAÇÃO ---
PASTA_RAIZ = r"C:\Users\luisv\Downloads\audit_20250701_4"
PASTA_SAIDA = "pocas_selecionadas"  # Onde as imagens salvas vão ficar

# Dimensões do RAW
LARGURA, ALTURA, CANAIS = 640, 480, 3


def ler_imagem_raw(caminho):
    try:
        raw_data = np.fromfile(caminho, dtype=np.uint8)
        esperado = LARGURA * ALTURA * CANAIS
        if raw_data.size >= esperado:
            raw_data = raw_data[-esperado:]
        else:
            return None
        img = raw_data.reshape((ALTURA, LARGURA, CANAIS))
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def main():
    # Cria a pasta de saída se não existir
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)
        print(f"Pasta criada: {PASTA_SAIDA}")

    print("Mapeando arquivos RAW (.image)...")
    arquivos = []
    for root, dirs, files in os.walk(PASTA_RAIZ):
        for file in files:
            if file.endswith(".image"):
                arquivos.append(os.path.join(root, file))

    total = len(arquivos)
    if total == 0:
        print("Nenhuma imagem encontrada.")
        return

    print(f"\n--- MODO DE COLETA ---")
    print(f"Encontradas {total} imagens.")
    print(" [D] ou [ESPAÇO] : Próxima")
    print(" [A]             : Anterior")
    print(" [W]             : Pular 100 imagens")
    print(f" [S]             : SALVAR imagem em '{PASTA_SAIDA}'")
    print(" [ESC]           : Sair")

    idx = 0
    while True:
        caminho = arquivos[idx]
        img = ler_imagem_raw(caminho)

        if img is None:
            idx = (idx + 1) % total
            continue

        # Cria uma cópia para escrever o texto (para não salvar o texto na imagem final)
        display_img = img.copy()
        nome_arquivo = os.path.basename(caminho)

        cv2.putText(display_img, f"{idx}/{total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img, f"Arquivo: {nome_arquivo}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    1)
        cv2.putText(display_img, "[S] Salvar | [D] Prox", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Coletor de Pocas (RAW)", display_img)

        k = cv2.waitKey(0)

        if k == 27:  # ESC
            break
        elif k == ord('d') or k == 32:  # D
            idx = (idx + 1) % total
        elif k == ord('a'):  # A
            idx = (idx - 1) % total
        elif k == ord('w'):  # W
            idx = (idx + 100) % total
        elif k == ord('s'):  # S - SALVAR
            # Gera um nome amigável: poca_ID_nomeoriginal.jpg
            nome_jpg = f"poca_{idx}_{nome_arquivo}.jpg"
            caminho_salvo = os.path.join(PASTA_SAIDA, nome_jpg)

            # Salva a imagem LIMPA (sem o texto verde)
            cv2.imwrite(caminho_salvo, img)
            print(f"[SALVO] {caminho_salvo}")

            # Feedback visual rápido
            cv2.putText(display_img, "SALVO COM SUCESSO!", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.imshow("Coletor de Pocas (RAW)", display_img)
            cv2.waitKey(500)  # Mostra a mensagem por 0.5s

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()