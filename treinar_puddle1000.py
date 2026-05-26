"""
Treina YOLO usando o dataset Puddle-1000 convertido.

Modos:
- seg: usa labels de segmentação (recomendado para aproveitar máscaras)
- det: usa labels de detecção (bbox derivadas das máscaras)
"""

from pathlib import Path
import argparse
import shutil
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Treino YOLO com Puddle-1000")
    parser.add_argument("--mode", choices=["seg", "det"], default="seg")
    parser.add_argument("--data", type=Path, default=None, help="Caminho para data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--project", type=str, default="runs/puddle1000")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    default_data = Path("puddle1000_yolo") / args.mode / "data.yaml"
    data_yaml = args.data if args.data is not None else default_data

    if not data_yaml.exists():
        print(f"❌ data.yaml não encontrado: {data_yaml}")
        print("Execute primeiro: python preparar_puddle1000_yolo.py")
        return

    model_name = "yolov8n-seg.pt" if args.mode == "seg" else "yolov8n.pt"
    run_name = args.name or f"puddle1000_{args.mode}"

    print("=" * 60)
    print("TREINAMENTO YOLO - PUDDLE-1000")
    print("=" * 60)
    print(f"Modo: {args.mode}")
    print(f"Modelo base: {model_name}")
    print(f"Dataset: {data_yaml}")
    print(f"Épocas: {args.epochs}")
    print(f"Batch: {args.batch}")
    print(f"Img size: {args.imgsz}")

    model = YOLO(model_name)
    model.train(
        data=str(data_yaml.absolute()),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=20,
        optimizer="AdamW",
        lr0=0.01,
        augment=True,
        mosaic=1.0,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        project=args.project,
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    best_src = Path(args.project) / run_name / "weights" / "best.pt"
    if best_src.exists():
        best_dst = Path(f"best_puddle1000_{args.mode}.pt")
        shutil.copy(best_src, best_dst)
        print(f"\n✅ Modelo final salvo em: {best_dst}")
    else:
        print(f"\n⚠️ best.pt não encontrado em: {best_src}")


if __name__ == "__main__":
    main()
