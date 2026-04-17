"""
Dataset Puddle-1000 para segmentação semântica de poças d'água.

Compatível com o dataset "Puddle-1000" do paper:
    Han et al., "Single Image Water Hazard Detection using FCN
    with Reflection Attention Units", ECCV 2018

Estrutura esperada:
    <dataset_root>/
        Puddle-1000 Dataset_train/       # treino completo (on+off road)
            images/  *.png
            masks/   *.png  (ou masks/0/*.png)
        Puddle-1000 Dataset_train_on/    # somente on-road
            ...
        Puddle-1000 Dataset_val/         # validação completa
            images/  *.png
            masks/   *.png
        Puddle-1000 Dataset_val_off/     # somente off-road
            ...
"""

from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_mask(mask_dir: Path, image_name: str) -> Optional[Path]:
    """Procura a máscara em mask_dir/ ou mask_dir/0/ ."""
    for candidate in (mask_dir / image_name, mask_dir / "0" / image_name):
        if candidate.exists():
            return candidate
    return None


_SPLIT_DIRS = {
    "train":     "Puddle-1000 Dataset_train",
    "train_on":  "Puddle-1000 Dataset_train_on",
    "val":       "Puddle-1000 Dataset_val",
    "val_off":   "Puddle-1000 Dataset_val_off",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Puddle1000SegDataset(Dataset):
    """Carrega pares (imagem, máscara) do Puddle-1000.

    Args:
        dataset_root: Raiz do dataset (contém as pastas dos splits).
        split:        Um de ``"train"``, ``"train_on"``, ``"val"``, ``"val_off"``.
        image_size:   Tupla ``(H, W)`` para redimensionamento. Default 360×640
                      (mesmo tamanho usado no paper).
        augment:      Se True, aplica data augmentation (flip horizontal,
                      jitter de brilho/contraste).
        max_samples:  Limita o nº de amostras (útil para debug rápido).
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str = "train",
        image_size: Tuple[int, int] = (360, 640),
        augment: bool = False,
        max_samples: Optional[int] = None,
    ):
        if split not in _SPLIT_DIRS:
            raise ValueError(
                f"split deve ser um de {list(_SPLIT_DIRS.keys())}, recebeu '{split}'"
            )

        split_root = dataset_root / _SPLIT_DIRS[split]
        self.image_dir = split_root / "images"
        self.mask_dir = split_root / "masks"
        self.image_size = image_size
        self.augment = augment

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Pasta de imagens não encontrada: {self.image_dir}")

        # Monta pares (imagem, máscara) válidos
        all_images: List[Path] = sorted(
            list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        )
        valid_pairs: List[Tuple[Path, Path]] = []
        for img_path in all_images:
            mask_path = _find_mask(self.mask_dir, img_path.name)
            if mask_path is not None:
                valid_pairs.append((img_path, mask_path))

        if max_samples is not None:
            valid_pairs = valid_pairs[:max_samples]

        self.samples = valid_pairs

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]

        # --- Leitura ---
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Falha ao ler imagem: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Falha ao ler máscara: {mask_path}")

        # --- Resize ---
        th, tw = self.image_size
        if image.shape[:2] != (th, tw):
            image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
        if mask.shape[:2] != (th, tw):
            mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # --- Augmentation (apenas no treino) ---
        if self.augment:
            image, mask = self._augment(image, mask)

        # --- Binarizar máscara (>0 = água) ---
        mask = (mask > 0).astype(np.int64)

        # --- Para tensores ---
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).long()
        return image_t, mask_t

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray):
        """Data augmentation simples: flip horizontal + jitter de brilho."""
        # Flip horizontal (50 % de chance)
        if np.random.rand() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        # Jitter de brilho/contraste
        if np.random.rand() < 0.3:
            alpha = np.random.uniform(0.8, 1.2)   # contraste
            beta = np.random.randint(-20, 20)      # brilho
            image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return image, mask
