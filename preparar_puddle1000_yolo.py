"""
Converte o dataset Puddle-1000 com máscaras para formato YOLO.

Saídas:
1) YOLO Segmentação (polígonos)
2) YOLO Detecção (bounding boxes derivadas das máscaras)

Estrutura esperada do dataset de entrada:
Puddle-1000_Dataset2/
  Puddle-1000 Dataset_train/
    images/
    masks/
  Puddle-1000 Dataset_val/
    images/
    masks/

Observação: em alguns subconjuntos as máscaras podem estar em masks/0/.
"""

from pathlib import Path
import argparse
import shutil
import cv2


CLASS_ID = 0
CLASS_NAME = "puddle"


def _encontrar_mascara(mask_dir: Path, image_name: str) -> Path | None:
    direta = mask_dir / image_name
    if direta.exists():
        return direta

    em_zero = mask_dir / "0" / image_name
    if em_zero.exists():
        return em_zero

    return None


def _contornos_da_mascara(mask_path: Path, min_area: float):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [], None

    mask_bin = (mask > 0).astype("uint8")
    contornos, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtrados = [c for c in contornos if cv2.contourArea(c) >= min_area]
    return filtrados, mask.shape[:2]


def _linha_seg(contorno, largura: int, altura: int, simplificar: float) -> str | None:
    if simplificar > 0:
        epsilon = simplificar * cv2.arcLength(contorno, True)
        contorno = cv2.approxPolyDP(contorno, epsilon, True)

    pontos = contorno.reshape(-1, 2)
    if len(pontos) < 3:
        return None

    coords = []
    for x, y in pontos:
        coords.append(f"{x / largura:.6f}")
        coords.append(f"{y / altura:.6f}")

    return f"{CLASS_ID} " + " ".join(coords)


def _linha_det(contorno, largura: int, altura: int) -> str:
    x, y, w, h = cv2.boundingRect(contorno)
    xc = (x + w / 2) / largura
    yc = (y + h / 2) / altura
    wn = w / largura
    hn = h / altura
    return f"{CLASS_ID} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"


def _garantir_pasta(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _processar_split(
    split_nome: str,
    src_images: Path,
    src_masks: Path,
    out_seg_images: Path,
    out_seg_labels: Path,
    out_det_images: Path,
    out_det_labels: Path,
    min_area: float,
    simplificar: float,
):
    _garantir_pasta(out_seg_images)
    _garantir_pasta(out_seg_labels)
    _garantir_pasta(out_det_images)
    _garantir_pasta(out_det_labels)

    imagens = sorted([p for p in src_images.glob("*.png")])
    total = len(imagens)
    sem_mascara = 0
    com_objeto = 0
    sem_objeto = 0

    print(f"\n[{split_nome}] imagens encontradas: {total}")

    for idx, img_path in enumerate(imagens, start=1):
        mask_path = _encontrar_mascara(src_masks, img_path.name)
        if mask_path is None:
            sem_mascara += 1
            continue

        shutil.copy2(img_path, out_seg_images / img_path.name)
        shutil.copy2(img_path, out_det_images / img_path.name)

        contornos, shape = _contornos_da_mascara(mask_path, min_area=min_area)
        if shape is None:
            continue

        altura, largura = shape
        linhas_seg = []
        linhas_det = []

        for contorno in contornos:
            linha_seg = _linha_seg(contorno, largura, altura, simplificar=simplificar)
            if linha_seg is not None:
                linhas_seg.append(linha_seg)
                linhas_det.append(_linha_det(contorno, largura, altura))

        if linhas_seg:
            com_objeto += 1
        else:
            sem_objeto += 1

        label_name = img_path.with_suffix(".txt").name
        (out_seg_labels / label_name).write_text("\n".join(linhas_seg), encoding="utf-8")
        (out_det_labels / label_name).write_text("\n".join(linhas_det), encoding="utf-8")

        if idx % 500 == 0 or idx == total:
            print(f"[{split_nome}] progresso: {idx}/{total}")

    print(f"[{split_nome}] com objeto: {com_objeto} | sem objeto: {sem_objeto} | sem máscara: {sem_mascara}")


def _escrever_yaml(path: Path, task_name: str):
    conteudo = (
        "# Gerado automaticamente por preparar_puddle1000_yolo.py\n"
        f"path: {path.parent.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "nc: 1\n"
        f"names: ['{CLASS_NAME}']\n"
    )
    path.write_text(conteudo, encoding="utf-8")
    print(f"[{task_name}] data.yaml criado em: {path}")


def main():
    parser = argparse.ArgumentParser(description="Converter Puddle-1000 (máscaras) para YOLO")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Puddle-1000_Dataset2"),
        help="Pasta raiz do dataset Puddle-1000",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("puddle1000_yolo"),
        help="Pasta raiz de saída",
    )
    parser.add_argument("--min-area", type=float, default=20.0, help="Área mínima de contorno em pixels")
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.002,
        help="Fator de simplificação do polígono (0 desativa)",
    )
    args = parser.parse_args()

    train_dir = args.dataset_root / "Puddle-1000 Dataset_train"
    val_dir = args.dataset_root / "Puddle-1000 Dataset_val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            "Estrutura não encontrada. Esperado: 'Puddle-1000 Dataset_train' e 'Puddle-1000 Dataset_val'"
        )

    seg_root = args.out_root / "seg"
    det_root = args.out_root / "det"

    _processar_split(
        split_nome="train",
        src_images=train_dir / "images",
        src_masks=train_dir / "masks",
        out_seg_images=seg_root / "images" / "train",
        out_seg_labels=seg_root / "labels" / "train",
        out_det_images=det_root / "images" / "train",
        out_det_labels=det_root / "labels" / "train",
        min_area=args.min_area,
        simplificar=args.simplify,
    )

    _processar_split(
        split_nome="val",
        src_images=val_dir / "images",
        src_masks=val_dir / "masks",
        out_seg_images=seg_root / "images" / "val",
        out_seg_labels=seg_root / "labels" / "val",
        out_det_images=det_root / "images" / "val",
        out_det_labels=det_root / "labels" / "val",
        min_area=args.min_area,
        simplificar=args.simplify,
    )

    _escrever_yaml(seg_root / "data.yaml", "seg")
    _escrever_yaml(det_root / "data.yaml", "det")

    print("\nConversão concluída.")
    print(f"- Segmentação YOLO: {seg_root}")
    print(f"- Detecção YOLO:   {det_root}")


if __name__ == "__main__":
    main()
