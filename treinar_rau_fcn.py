"""
treinar_rau_fcn.py — Treino FCN-8s + RAU para detecção de poças (Puddle-1000)

Replica a abordagem de:
    Han et al., "Single Image Water Hazard Detection using FCN
    with Reflection Attention Units", ECCV 2018
    https://github.com/Cow911/SingleImageWaterHazardDetectionWithRAU

Uso:
    # Treinar FCN-8s **com** RAU (padrão)
    python treinar_rau_fcn.py --mode rau

    # Treinar baseline FCN-8s **sem** RAU (para comparação)
    python treinar_rau_fcn.py --mode baseline

    # Debug rápido (poucas amostras)
    python treinar_rau_fcn.py --mode rau --max-train 20 --max-val 10 --epochs 3
"""

from pathlib import Path
import argparse
import json
import time

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from rau_fcn.dataset import Puddle1000SegDataset
from rau_fcn.model import FCN8sRAU
from rau_fcn.metrics import combined_loss, segmentation_scores


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_eval(model, loader, device):
    """Roda validação e retorna métricas médias por batch."""
    model.eval()
    keys = ("precision", "recall", "f1", "iou", "pixel_acc")
    sums = {k: 0.0 for k in keys}
    loss_sum = 0.0
    n = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss_sum += combined_loss(logits, masks).item()
        scores = segmentation_scores(logits, masks)
        for k in keys:
            sums[k] += scores[k]
        n += 1

    if n == 0:
        return {k: 0.0 for k in keys} | {"val_loss": 0.0}

    return {k: sums[k] / n for k in keys} | {"val_loss": loss_sum / n}


# ---------------------------------------------------------------------------
# Loop de treino
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = combined_loss(logits, masks)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
    return loss_sum / max(1, len(loader))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Treino FCN-8s + RAU para Puddle-1000 (replica ECCV 2018)"
    )
    # Dados
    ap.add_argument("--dataset-root", type=Path, default=Path("Puddle-1000_Dataset2"))
    ap.add_argument("--train-split", default="train",
                    choices=["train", "train_on"],
                    help="Split de treino (completo ou só on-road)")
    ap.add_argument("--val-split", default="val",
                    choices=["val", "val_off"],
                    help="Split de validação (completo ou só off-road)")
    # Modelo
    ap.add_argument("--mode", choices=["baseline", "rau"], default="rau",
                    help="'rau' = FCN-8s + RAU; 'baseline' = FCN-8s puro")
    ap.add_argument("--rau-mode", choices=["full", "light"], default="light",
                    help="RAU 'full' (paper) ou 'light' (eficiente)")
    ap.add_argument("--head-dim", type=int, default=512,
                    help="Dim. dos canais fc6/fc7 (4096 no paper, 512 leve)")
    # Treino
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-step", type=int, default=10,
                    help="A cada N épocas, divide o LR por 2")
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--augment", action="store_true", default=True,
                    help="Ativar data augmentation no treino")
    # Imagem
    ap.add_argument("--img-h", type=int, default=360)
    ap.add_argument("--img-w", type=int, default=640)
    # Limites (debug)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    # Saída
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--save-root", type=Path, default=Path("runs/rau_fcn"))
    args = ap.parse_args()

    # ---- Setup ----
    use_rau = args.mode == "rau"
    run_name = f"puddle1000_{args.mode}_{args.rau_mode}" if use_rau else f"puddle1000_{args.mode}"
    save_dir = args.save_root / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Datasets ----
    train_ds = Puddle1000SegDataset(
        dataset_root=args.dataset_root,
        split=args.train_split,
        image_size=(args.img_h, args.img_w),
        augment=args.augment,
        max_samples=args.max_train,
    )
    val_ds = Puddle1000SegDataset(
        dataset_root=args.dataset_root,
        split=args.val_split,
        image_size=(args.img_h, args.img_w),
        augment=False,
        max_samples=args.max_val,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )

    # ---- Modelo ----
    model = FCN8sRAU(
        num_classes=2,
        use_rau=use_rau,
        rau_mode=args.rau_mode,
        head_dim=args.head_dim,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    # ---- Log ----
    print("=" * 72)
    print("  TREINO FCN-8s + RAU — Detecção de poças (Puddle-1000)")
    print("  Paper: Han et al., ECCV 2018")
    print("=" * 72)
    print(f"  Modo          : {args.mode}  (RAU mode: {args.rau_mode})")
    print(f"  Device        : {device}")
    print(f"  Parâmetros    : {n_params:,}")
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val samples   : {len(val_ds)}")
    print(f"  Epochs        : {args.epochs}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  LR            : {args.lr}  (step={args.lr_step}, gamma=0.5)")
    print(f"  Head dim      : {args.head_dim}")
    print(f"  Augment       : {args.augment}")
    print(f"  Save dir      : {save_dir}")
    print("=" * 72)

    # ---- Treino ----
    best_iou = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val = run_eval(model, val_loader, device)
        scheduler.step()
        dur = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]

        record = {"epoch": epoch, "train_loss": train_loss, **val,
                  "lr": current_lr, "duration_sec": round(dur, 1)}
        history.append(record)

        improved = val["iou"] > best_iou
        marker = " *" if improved else ""
        print(
            f"  [{epoch:03d}/{args.epochs}]  "
            f"loss={train_loss:.4f}  val_loss={val['val_loss']:.4f}  "
            f"IoU={val['iou']:.4f}  F1={val['f1']:.4f}  "
            f"P={val['precision']:.4f}  R={val['recall']:.4f}  "
            f"lr={current_lr:.2e}  {dur:.0f}s{marker}"
        )

        if improved:
            best_iou = val["iou"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_iou": best_iou,
                "args": vars(args),
            }, save_dir / "best.pt")

    # Salvar último checkpoint também
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "args": vars(args),
    }, save_dir / "last.pt")

    # Salvar histórico
    history_path = save_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ---- Resumo final ----
    if history:
        best = max(history, key=lambda r: r["iou"])
        print("\n" + "=" * 72)
        print("  MELHOR RESULTADO")
        print("=" * 72)
        for k, v in best.items():
            if isinstance(v, float):
                print(f"    {k:15s}: {v:.6f}")
            else:
                print(f"    {k:15s}: {v}")
        print(f"\n  Checkpoints em: {save_dir}")
        print(f"  Histórico em  : {history_path}")


if __name__ == "__main__":
    main()
