"""Multi-branch forgery classifier with tensor-only FFT and HOG streams."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features * self.gate(features.unsqueeze(-1)).squeeze(-1)


class TensorHOG(nn.Module):
    """Differentiable HOG-like descriptor; all work stays batched on the tensor device."""

    def __init__(self, bins: int = 9, cells: int = 7) -> None:
        super().__init__()
        self.bins, self.cells = bins, cells
        self.register_buffer("sobel_x", torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3))
        self.register_buffer("sobel_y", torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3))

    def forward(self, image: Tensor) -> Tensor:
        gray = image.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-8)
        orientation = torch.remainder(torch.atan2(gy, gx), torch.pi)
        centers = torch.arange(self.bins, device=image.device, dtype=image.dtype).view(1, self.bins, 1, 1)
        centers = centers * (torch.pi / self.bins)
        distance = torch.abs(torch.remainder(orientation - centers + torch.pi / 2, torch.pi) - torch.pi / 2)
        votes = magnitude * F.relu(1.0 - distance / (torch.pi / self.bins))
        pooled = F.adaptive_avg_pool2d(votes, (self.cells, self.cells))
        return F.normalize(pooled.flatten(1), dim=1)


class ForgeryNet(nn.Module):
    """ResNet-50 image branch + FFT magnitude and HOG auxiliary branches."""

    def __init__(self, *, imagenet: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if imagenet else None
        backbone = resnet50(weights=weights)
        self.image_branch = nn.Sequential(*list(backbone.children())[:-1])
        self.fft_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        self.hog = TensorHOG()
        self.hog_branch = nn.Sequential(nn.Linear(9 * 7 * 7, 128), nn.ReLU(inplace=True))
        self.attention = ChannelAttention(2048 + 128 + 128)
        self.classifier = nn.Linear(2048 + 128 + 128, 1)

    def fft_magnitude(self, image: Tensor) -> Tensor:
        """Return normalized log-magnitude spectra without CPU/Python image loops."""
        gray = image.mean(dim=1, keepdim=True)
        spectrum = torch.fft.rfft2(gray, norm="ortho")
        magnitude = torch.log1p(torch.abs(spectrum))
        magnitude = (magnitude - magnitude.mean(dim=(-2, -1), keepdim=True)) / (
            magnitude.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        )
        return magnitude

    def forward(self, image: Tensor) -> Tensor:
        visual = self.image_branch(image).flatten(1)
        fft = self.fft_branch(self.fft_magnitude(image))
        hog = self.hog_branch(self.hog(image))
        return self.classifier(self.attention(torch.cat((visual, fft, hog), dim=1))).squeeze(1)
