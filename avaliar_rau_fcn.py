"""
avaliar_rau_fcn.py — Avaliação e visualização de resultados do FCN-8s + RAU

Modos de uso:
    # Avaliar métricas no conjunto de validação
    python avaliar_rau_fcn.py --checkpoint runs/rau_fcn/puddle1000_rau_light/best.pt

    # Gerar imagens de visualização (overlay da predição sobre a imagem)
    python avaliar_rau_fcn.py --checkpoint runs/rau_fcn/puddle1000_rau_light/best.pt --visualize

    # Avaliar no split off-road
    python avaliar_rau_fcn.py --checkpoint runs/rau_fcn/puddle1000_rau_light/best.pt --val-split val_off

    # Limitar a N amostras (debug rápido)
    python avaliar_rau_fcn.py --checkpoint runs/rau_fcn/puddle1000_rau_light/best.pt --max-val 10 --visualize
"""

from pathlib import Path
import argparse
import json

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from rau_fcn.dataset import Puddle1000SegDataset
from rau_fcn.model import FCN8sRAU
from rau_fcn.metrics import segmentation_scores, combined_loss


# ---------------------------------------------------------------------------
# Avaliação quantitativa
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Roda o modelo no conjunto de validação e retorna métricas agregadas."""
    model.eval()
    keys = ("precision", "recall", "f1", "iou", "pixel_acc")
    sums = {k: 0.0 for k in keys}
    tp_total = fp_total = fn_total = tn_total = 0
    loss_sum = 0.0
    n = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss_sum += combined_loss(logits, masks).item()

        scores = segmentation_scores(logits, masks)
        for k in keys:
            sums[k] += scores[k]
        tp_total += scores["tp"]
        fp_total += scores["fp"]
        fn_total += scores["fn"]
        tn_total += scores["tn"]
        n += 1

    if n == 0:
        return {k: 0.0 for k in keys}

    # Métricas médias por batch
    avg = {k: sums[k] / n for k in keys}
    avg["val_loss"] = loss_sum / n

    # Métricas globais (micro-average — mais confiáveis)
    eps = 1e-9
    avg["global_precision"] = tp_total / (tp_total + fp_total + eps)
    avg["global_recall"] = tp_total / (tp_total + fn_total + eps)
    avg["global_f1"] = (2 * avg["global_precision"] * avg["global_recall"]
                        / (avg["global_precision"] + avg["global_recall"] + eps))
    avg["global_iou"] = tp_total / (tp_total + fp_total + fn_total + eps)
    avg["global_pixel_acc"] = (tp_total + tn_total) / (tp_total + tn_total + fp_total + fn_total + eps)
    avg["n_batches"] = n

    return avg


# ---------------------------------------------------------------------------
# Visualização (overlay)
# ---------------------------------------------------------------------------

def make_overlay(image_np: np.ndarray, mask_gt: np.ndarray,
                 mask_pred: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Cria imagem com 3 painéis: original + GT + predição.

    Água é mostrada em vermelho translúcido (como no paper).
    """
    h, w = image_np.shape[:2]

    # Overlay GT (verde)
    overlay_gt = image_np.copy()
    water_gt = mask_gt > 0
    overlay_gt[water_gt] = (
        (1 - alpha) * overlay_gt[water_gt] + alpha * np.array([0, 200, 0])
    ).astype(np.uint8)

    # Overlay predição (vermelho — como no paper)
    overlay_pred = image_np.copy()
    water_pred = mask_pred > 0
    overlay_pred[water_pred] = (
        (1 - alpha) * overlay_pred[water_pred] + alpha * np.array([220, 30, 30])
    ).astype(np.uint8)

    # Concatenar horizontalmente: Original | GT | Predição
    separator = np.ones((h, 3, 3), dtype=np.uint8) * 200
    combined = np.concatenate([image_np, separator, overlay_gt, separator, overlay_pred], axis=1)
    return combined


@torch.no_grad()
def generate_visualizations(model, dataset, device, output_dir: Path, max_images: int = 50):
    """Gera imagens de visualização para as primeiras N amostras."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(dataset), max_images)
    print(f"\n  Gerando {n} visualizações em: {output_dir}")

    for i in range(n):
        image_t, mask_t = dataset[i]
        image_t = image_t.unsqueeze(0).to(device)

        logits = model(image_t)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Converter imagem de volta para numpy (H, W, 3) uint8
        image_np = (image_t.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask_np = mask_t.numpy()

        vis = make_overlay(image_np, mask_np, pred)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        fname = f"vis_{i:04d}.png"
        cv2.imwrite(str(output_dir / fname), vis_bgr)

    print(f"  {n} imagens salvas.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Avaliar e visualizar resultados do FCN-8s + RAU (Puddle-1000)"
    )
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Caminho para o checkpoint .pt (ex: runs/rau_fcn/.../best.pt)")
    ap.add_argument("--dataset-root", type=Path, default=Path("Puddle-1000_Dataset2"))
    ap.add_argument("--val-split", default="val",
                    choices=["val", "val_off", "train", "train_on"],
                    help="Split para avaliar")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--img-h", type=int, default=360)
    ap.add_argument("--img-w", type=int, default=640)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    # Visualização
    ap.add_argument("--visualize", action="store_true",
                    help="Gerar imagens de visualização (overlay)")
    ap.add_argument("--max-vis", type=int, default=50,
                    help="Máximo de imagens de visualização")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Pasta de saída das visualizações (default: junto do checkpoint)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Carregar checkpoint ----
    print(f"\n  Carregando checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    ckpt_args = ckpt.get("args", {})

    # Reconstruir modelo com os mesmos parâmetros do treino
    use_rau = ckpt_args.get("mode", "rau") == "rau"
    rau_mode = ckpt_args.get("rau_mode", "light")
    head_dim = ckpt_args.get("head_dim", 512)
    dropout = ckpt_args.get("dropout", 0.15)

    model = FCN8sRAU(
        num_classes=2,
        use_rau=use_rau,
        rau_mode=rau_mode,
        head_dim=head_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Modelo: FCN-8s {'+ RAU' if use_rau else '(baseline)'}  "
          f"(rau_mode={rau_mode}, head_dim={head_dim})")
    print(f"  Parâmetros: {n_params:,}")
    print(f"  Epoch do checkpoint: {ckpt.get('epoch', '?')}")
    print(f"  Best IoU no treino: {ckpt.get('best_iou', '?')}")
    print(f"  Device: {device}")

    # ---- Dataset ----
    ds = Puddle1000SegDataset(
        dataset_root=args.dataset_root,
        split=args.val_split,
        image_size=(args.img_h, args.img_w),
        augment=False,
        max_samples=args.max_val,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    print(f"  Split: {args.val_split}  ({len(ds)} amostras)")

    # ---- Avaliação ----
    print("\n  Avaliando...")
    metrics = evaluate(model, loader, device)

    print("\n" + "=" * 60)
    print("  RESULTADOS — MÉTRICAS POR BATCH (macro-average)")
    print("=" * 60)
    for k in ("val_loss", "precision", "recall", "f1", "iou", "pixel_acc"):
        print(f"    {k:15s}: {metrics[k]:.6f}")

    print("\n  MÉTRICAS GLOBAIS (micro-average)")
    print("  " + "-" * 40)
    for k in ("global_precision", "global_recall", "global_f1", "global_iou", "global_pixel_acc"):
        print(f"    {k:20s}: {metrics[k]:.6f}")

    # Salvar métricas JSON ao lado do checkpoint
    metrics_path = args.checkpoint.parent / f"eval_{args.val_split}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n  Métricas salvas em: {metrics_path}")

    # ---- Visualização ----
    if args.visualize:
        out_dir = args.output_dir or (args.checkpoint.parent / f"vis_{args.val_split}")
        generate_visualizations(model, ds, device, out_dir, max_images=args.max_vis)


if __name__ == "__main__":
    main()
