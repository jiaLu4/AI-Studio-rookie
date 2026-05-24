from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .config import MOS_INPUT_HEIGHT, MOS_INPUT_WIDTH, MOS_MODEL_NAME, MOS_MODEL_PATH


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MOS_MIN_SCORE = 0.0
MOS_MAX_SCORE = 4.0
MOS_NORMALIZATION_INTERVAL = 4.0


@dataclass
class MosPrediction:
    mos_raw: float
    mos_score: float
    quality_score: float


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block: type[nn.Module], layers: list[int], num_outputs: int = 1) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_outputs)

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(model_name: str) -> nn.Module:
    if model_name == "resnet18":
        return ResNet(BasicBlock, [2, 2, 2, 2], num_outputs=1)
    elif model_name == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3], num_outputs=1)
    else:
        raise ValueError(f"Unsupported MOS model name: {model_name}")


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = image.resize((MOS_INPUT_WIDTH, MOS_INPUT_HEIGHT))
    array = np.asarray(image).astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0)


def mos_to_100(mos_score: float) -> float:
    normalized_score = ((mos_score - MOS_MIN_SCORE) / MOS_NORMALIZATION_INTERVAL) * 100.0
    return float(np.clip(normalized_score, 0.0, 100.0))


class MosService:
    def __init__(
        self,
        model_path: Path = MOS_MODEL_PATH,
        model_name: str = MOS_MODEL_NAME,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: nn.Module | None = None

    def load(self) -> None:
        if self.model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"MOS model file not found: {self.model_path}")

        model = build_model(self.model_name).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        self.model = model

    @torch.no_grad()
    def predict(self, image: Image.Image) -> MosPrediction:
        self.load()
        assert self.model is not None

        tensor = preprocess_image(image).to(self.device)
        mos_raw = float(self.model(tensor).squeeze().item())
        mos_score = float(np.clip(mos_raw, MOS_MIN_SCORE, MOS_MAX_SCORE))
        return MosPrediction(
            mos_raw=mos_raw,
            mos_score=mos_score,
            quality_score=mos_to_100(mos_score),
        )

    def predict_quality(self, image: Image.Image) -> Dict[str, Any]:
        prediction = self.predict(image)
        return {
            "skipped": False,
            "mos_score": prediction.mos_score,
            "quality_score": prediction.quality_score,
            "mos_min_score": MOS_MIN_SCORE,
            "mos_max_score": MOS_MAX_SCORE,
            "normalization_interval": MOS_NORMALIZATION_INTERVAL,
            "message": "Stage 2 MOS overall image quality score generated.",
        }

    def health(self) -> Dict[str, str]:
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "device": str(self.device),
            "loaded": str(self.model is not None),
            "mos_min_score": str(MOS_MIN_SCORE),
            "mos_max_score": str(MOS_MAX_SCORE),
            "normalization_interval": str(MOS_NORMALIZATION_INTERVAL),
        }
