"""
FCN-8s com Reflection Attention Units (RAU)

Referência:
    Han et al., "Single Image Water Hazard Detection using FCN
    with Reflection Attention Units", ECCV 2018

Implementação em PyTorch baseada no código TensorFlow original:
    https://github.com/Cow911/SingleImageWaterHazardDetectionWithRAU

Arquitetura:
    Encoder VGG16-like (5 blocos) → RAU após cada bloco →
    Head fc6/fc7 (convoluções 1x1/3x3) → Decoder FCN-8s (skip pool3, pool4)

O RAU captura padrões de reflexão vertical dividindo o feature map
em faixas horizontais, computando a diferença absoluta pixel-a-pixel
entre cada faixa expandida e o feature map original.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Blocos auxiliares
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Bloco de convoluções estilo VGG (conv 3x3 + ReLU, repetido n vezes)."""

    def __init__(self, in_channels: int, out_channels: int, n_convs: int):
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_channels
        for _ in range(n_convs):
            layers.append(nn.Conv2d(ch, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch = out_channels
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Reflection Attention Unit (RAU)
# ---------------------------------------------------------------------------

class RAUBlock(nn.Module):
    """Reflection Attention Unit.

    Divide o feature map em *n_strips* faixas horizontais via average pooling,
    redimensiona cada faixa de volta ao tamanho original e calcula |x − strip_k|.

    Modos
    -----
    ``"full"``  (fiel ao paper):
        Concatena cada diferença com todos os C canais → saída C·(1+n) canais.
        Consome **muita** VRAM (ex.: 512×17 = 8 704 canais no bloco 4).
    ``"light"`` (eficiente):
        Calcula somente a média da diferença absoluta por strip (1 canal/strip)
        → saída C+n canais.  Muito mais leve, resultado parecido.
    """

    def __init__(self, channels: int, n_strips: int = 16, mode: str = "light"):
        super().__init__()
        self.channels = channels
        self.n_strips = n_strips
        self.mode = mode

        fuse_in = channels * (1 + n_strips) if mode == "full" else channels + n_strips
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Avg-pool em n_strips linhas × W/2 colunas (como no paper)
        pooled = F.adaptive_avg_pool2d(x, (self.n_strips, max(1, W // 2)))

        diffs: list[torch.Tensor] = []
        for k in range(self.n_strips):
            strip = pooled[:, :, k : k + 1, :]                       # [B,C,1,W/2]
            strip_up = F.interpolate(strip, size=(H, W),
                                     mode="bilinear", align_corners=False)  # [B,C,H,W]
            diff = torch.abs(x - strip_up)                            # [B,C,H,W]

            if self.mode == "full":
                diffs.append(diff)                                     # C canais
            else:
                diffs.append(diff.mean(dim=1, keepdim=True))           # 1 canal

        attention = torch.cat(diffs, dim=1)      # [B, n*C, H, W] ou [B, n, H, W]
        out = torch.cat([x, attention], dim=1)   # [B, C*(1+n), H, W] ou [B, C+n, H, W]
        return self.fuse(out)


# ---------------------------------------------------------------------------
# FCN-8s com RAU
# ---------------------------------------------------------------------------

class FCN8sRAU(nn.Module):
    """FCN-8s + Reflection Attention Units para detecção de água/poças.

    Args:
        num_classes: Nº de classes (2 = água / não-água).
        use_rau:     Se True, aplica RAU após cada bloco encoder.
        rau_mode:    ``"full"`` (paper original) ou ``"light"`` (eficiente).
        head_dim:    Dimensão dos canais no head fc6/fc7.
                     4096 replica o VGG16 original; 512 é mais leve.
        dropout:     Taxa de dropout no head.
    """

    def __init__(
        self,
        num_classes: int = 2,
        use_rau: bool = True,
        rau_mode: str = "light",
        head_dim: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.use_rau = use_rau

        # --- Encoder (VGG-16 style) ---
        self.enc1 = ConvBlock(3, 64, 2)
        self.enc2 = ConvBlock(64, 128, 2)
        self.enc3 = ConvBlock(128, 256, 3)
        self.enc4 = ConvBlock(256, 512, 3)
        self.enc5 = ConvBlock(512, 512, 3)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # --- RAU blocks (um por bloco encoder) ---
        self.rau1 = RAUBlock(64,  n_strips=16, mode=rau_mode)
        self.rau2 = RAUBlock(128, n_strips=16, mode=rau_mode)
        self.rau3 = RAUBlock(256, n_strips=16, mode=rau_mode)
        self.rau4 = RAUBlock(512, n_strips=16, mode=rau_mode)
        self.rau5 = RAUBlock(512, n_strips=8,  mode=rau_mode)

        # --- Head (fc6 / fc7 / score — convolucionais) ---
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.score_fr = nn.Conv2d(head_dim, num_classes, kernel_size=1)

        # --- Decoder: projeções dos skips (pool3 / pool4 → num_classes) ---
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)

        self._init_weights()

    # ---- inicialização de pesos (Kaiming, como é padrão em PyTorch) ----
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _maybe_rau(self, x: torch.Tensor, rau: RAUBlock) -> torch.Tensor:
        return rau(x) if self.use_rau else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]  # (H, W)

        # ---- Encoder ----
        x1 = self._maybe_rau(self.enc1(x), self.rau1)
        p1 = self.pool(x1)            # /2

        x2 = self._maybe_rau(self.enc2(p1), self.rau2)
        p2 = self.pool(x2)            # /4

        x3 = self._maybe_rau(self.enc3(p2), self.rau3)
        p3 = self.pool(x3)            # /8

        x4 = self._maybe_rau(self.enc4(p3), self.rau4)
        p4 = self.pool(x4)            # /16

        x5 = self._maybe_rau(self.enc5(p4), self.rau5)
        p5 = self.pool(x5)            # /32

        # ---- Head ----
        h = self.fc6(p5)
        h = self.fc7(h)
        score = self.score_fr(h)

        # ---- Decoder (FCN-8s) ----
        # up 2× + skip pool4
        up2 = F.interpolate(score, size=p4.shape[2:],
                            mode="bilinear", align_corners=False)
        fuse4 = up2 + self.score_pool4(p4)

        # up 2× + skip pool3
        up4 = F.interpolate(fuse4, size=p3.shape[2:],
                            mode="bilinear", align_corners=False)
        fuse3 = up4 + self.score_pool3(p3)

        # up 8× ao tamanho original
        out = F.interpolate(fuse3, size=input_size,
                            mode="bilinear", align_corners=False)
        return out
