"""Custom CNN and memory-conscious MobileNetV2 transfer-learning classifier."""
import warnings
import torch
from torch import nn

from config import NUM_CLASSES


class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True), nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False), nn.BatchNorm2d(output_channels), nn.ReLU(inplace=True), nn.MaxPool2d(2), nn.Dropout2d(0.12))

    def forward(self, x):
        return self.layers(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(ConvBlock(3, 24), ConvBlock(24, 48), ConvBlock(48, 96), ConvBlock(96, 160))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.35), nn.Linear(160, num_classes))

    def forward(self, images):
        return self.classifier(self.features(images))


class MobileNetV2Classifier(nn.Module):
    """Load torchvision MobileNetV2 weights only when explicitly permitted."""
    def __init__(self, num_classes=NUM_CLASSES, pretrained=False, weights_path=None):
        super().__init__()
        try:
            from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("MobileNetV2 requires a compatible torchvision installation. Install the project's requirements and retry.") from exc
        weights = MobileNet_V2_Weights.DEFAULT if pretrained and weights_path is None else None
        initialized_from_imagenet = weights is not None
        try:
            self.backbone = mobilenet_v2(weights=weights, progress=True)
        except Exception as exc:
            if weights_path is not None:
                raise RuntimeError(f"Could not load pretrained weights from {weights_path}: {exc}") from exc
            if pretrained:
                warnings.warn(f"ImageNet MobileNetV2 weights unavailable; falling back to random initialization. This is not pretrained transfer learning. Details: {exc}", RuntimeWarning, stacklevel=2)
                self.backbone = mobilenet_v2(weights=None, progress=False)
                initialized_from_imagenet = False
            else:
                raise
        if weights_path is not None:
            self.backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, num_classes)
        if initialized_from_imagenet or weights_path is not None:
            for parameter in self.backbone.features.parameters():
                parameter.requires_grad = False

    def unfreeze_tail(self, blocks=3):
        for parameter in self.backbone.features[-blocks:].parameters():
            parameter.requires_grad = True

    def forward(self, images):
        return self.backbone(images)


def build_model(name="custom_cnn", pretrained=False, weights_path=None):
    if name == "custom_cnn":
        return CustomCNN()
    if name == "mobilenetv2":
        return MobileNetV2Classifier(pretrained=pretrained, weights_path=weights_path)
    raise ValueError(f"Unknown model {name!r}; choose custom_cnn or mobilenetv2.")
