from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.PReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        scale = self.avg_pool(x).view(batch, channels)
        scale = self.fc(scale).view(batch, channels, 1, 1)
        return x * scale


class IRBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None, use_se: bool = False):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.use_se = use_se
        self.se = SEBlock(planes) if use_se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.prelu(out)


class ResNet18Face(nn.Module):
    """ResNet18 face embedding model extracted from the ArcFace codebase."""

    def __init__(
        self,
        input_size: int = 64,
        embedding_size: int = 512,
        input_channels: int = 1,
        use_se: bool = False,
        dropout: float = 0.5,
    ):
        super().__init__()
        if input_size % 16 != 0:
            raise ValueError("input_size must be divisible by 16 for this ResNet18Face layout")

        self.inplanes = 64
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.use_se = use_se

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(64, blocks=2)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.bn4 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=dropout)

        feature_map_size = input_size // 16
        self.fc5 = nn.Linear(512 * feature_map_size * feature_map_size, embedding_size)
        self.bn5 = nn.BatchNorm1d(embedding_size)
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * IRBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * IRBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * IRBlock.expansion),
            )

        layers = [IRBlock(self.inplanes, planes, stride, downsample, use_se=self.use_se)]
        self.inplanes = planes * IRBlock.expansion
        for _ in range(1, blocks):
            layers.append(IRBlock(self.inplanes, planes, use_se=self.use_se))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc5(x)
        return self.bn5(x)


def resnet18_face(**kwargs) -> ResNet18Face:
    return ResNet18Face(**kwargs)


class NormalizedEmbeddingModel(nn.Module):
    """Expose a unit-length face embedding for metric learning and verification."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), p=2, dim=1)
