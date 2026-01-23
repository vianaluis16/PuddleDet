import cv2
import numpy as np
import os

# Pasta onde você salvou as imagens com 'S'
PASTA_AMOSTRAS = "pocas_selecionadas"


def nada(x): pass


def main():
    # Lista arquivos jpg na pasta
    arquivos = [os.path.join(PASTA_AMOSTRAS, f) for f in os.listdir(PASTA_AMOSTRAS) if f.endswith('.jpg')]

    if not arquivos:
        print(f"ERRO: Nenhuma imagem encontrada em '{PASTA_AMOSTRAS}'")
        return

    print(f"Carregadas {len(arquivos)} imagens para teste.")
    print("USE AS BARRAS para deixar a poça BRANCA e o resto PRETO.")
    print("Anote os valores de H, S, V (Min e Max).")
    print("[ESPAÇO] Próxima imagem | [ESC] Sair")

    cv2.namedWindow('Calibrador')

    # Valores iniciais (Reflexo brilhante)
    cv2.createTrackbar('H Min', 'Calibrador', 0, 179, nada)
    cv2.createTrackbar('S Min', 'Calibrador', 0, 255, nada)
    cv2.createTrackbar('V Min', 'Calibrador', 140, 255, nada)  # Brilho

    cv2.createTrackbar('H Max', 'Calibrador', 180, 179, nada)
    cv2.createTrackbar('S Max', 'Calibrador', 255, 255, nada)
    cv2.createTrackbar('V Max', 'Calibrador', 255, 255, nada)

    idx = 0
    while True:
        caminho = arquivos[idx]
        img = cv2.imread(caminho)

        if img is None:
            idx = (idx + 1) % len(arquivos)
            continue

        # Redimensiona para caber na tela se for grande
        img = cv2.resize(img, (640, 480))

        # Lê posições das barras
        h_min = cv2.getTrackbarPos('H Min', 'Calibrador')
        s_min = cv2.getTrackbarPos('S Min', 'Calibrador')
        v_min = cv2.getTrackbarPos('V Min', 'Calibrador')

        h_max = cv2.getTrackbarPos('H Max', 'Calibrador')
        s_max = cv2.getTrackbarPos('S Max', 'Calibrador')
        v_max = cv2.getTrackbarPos('V Max', 'Calibrador')

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        # Cria máscara
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # Mostra original + mascara
        resultado = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('Calibrador', np.hstack([img, resultado]))  # Mostra lado a lado

        k = cv2.waitKey(10)
        if k == 27:
            break  # ESC
        elif k == 32:  # Espaço
            idx = (idx + 1) % len(arquivos)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()