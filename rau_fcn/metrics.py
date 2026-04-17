"""
Funções de perda e métricas para segmentação binária (água / não-água).

Perdas implementadas:
    - Focal loss (usada no paper original, adapatada para 2 classes)
    - Cross-entropy padrão
    - Combinação ponderada CE + Focal

Métricas: precision, recall, F1, IoU, pixel accuracy.
"""

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Focal loss (adaptada de Han et al. / Cow911)
# ---------------------------------------------------------------------------

def focal_loss(logits: torch.Tensor, target: torch.Tensor,
               alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss para segmentação binária (2 classes via softmax).

    ``alpha`` controla o peso da classe positiva (água). Valores > 0.5
    dão mais peso à classe minoritária (água geralmente ocupa menos pixels).
    ``gamma`` focaliza amostras difíceis.
    """
    probs = torch.softmax(logits, dim=1)[:, 1, :, :]   # P(água)
    target_f = target.float()

    pt = probs * target_f + (1.0 - probs) * (1.0 - target_f)
    alpha_t = alpha * target_f + (1.0 - alpha) * (1.0 - target_f)
    loss = -alpha_t * ((1.0 - pt) ** gamma) * torch.log(pt.clamp_min(1e-8))
    return loss.mean()


# ---------------------------------------------------------------------------
# Perda combinada (padrão do treinamento)
# ---------------------------------------------------------------------------

def combined_loss(logits: torch.Tensor, target: torch.Tensor,
                  focal_weight: float = 0.5) -> torch.Tensor:
    """CE + focal loss ponderadas."""
    ce = F.cross_entropy(logits, target)
    fl = focal_loss(logits, target)
    return (1.0 - focal_weight) * ce + focal_weight * fl


# ---------------------------------------------------------------------------
# Métricas de segmentação
# ---------------------------------------------------------------------------

@torch.no_grad()
def segmentation_scores(logits: torch.Tensor, target: torch.Tensor) -> dict:
    """Calcula precision, recall, F1, IoU e pixel accuracy."""
    pred = torch.argmax(logits, dim=1)

    pred_pos = pred == 1
    true_pos = target == 1

    tp = (pred_pos & true_pos).sum().item()
    fp = (pred_pos & ~true_pos).sum().item()
    fn = (~pred_pos & true_pos).sum().item()
    tn = (~pred_pos & ~true_pos).sum().item()

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "pixel_acc": acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
