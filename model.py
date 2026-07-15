"""
Model definition and inference utilities for NeuroVox.

A ResNet-18 (transfer learning) trained to classify brain MRI scans into
four categories: glioma, meningioma, pituitary tumor, and no tumor.
"""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# Class order MUST match the training dataset (alphabetical folder order on the
# Kaggle "Brain Tumor MRI Dataset").
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
PRETTY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary Tumor",
}

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Inference transform (no augmentation).
_infer_tf = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def build_model(num_classes: int = 4, pretrained: bool = False) -> nn.Module:
    """Create a ResNet-18 with a fresh classification head."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(weights_path: str | None, device: torch.device) -> Tuple[nn.Module, bool]:
    """
    Load trained weights if available.

    Returns (model, is_real). ``is_real`` is False when no trained weights were
    found and a randomly initialised model is used as a placeholder so the UI
    can still be demonstrated.
    """
    model = build_model(num_classes=len(CLASS_NAMES), pretrained=False)
    is_real = False
    if weights_path:
        try:
            state = torch.load(weights_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            is_real = True
        except Exception:
            # Fall back to placeholder weights.
            is_real = False
    model.to(device)
    model.eval()
    return model, is_real


def preprocess(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image into a normalised 1x3x224x224 tensor."""
    image = image.convert("RGB")
    return _infer_tf(image).unsqueeze(0)


@torch.no_grad()
def predict(model: nn.Module, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    """Return softmax probabilities over the four classes."""
    tensor = tensor.to(device)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def load_pil_from_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")
