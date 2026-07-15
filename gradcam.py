"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for NeuroVox.

Produces a heatmap that highlights the regions of the MRI the model relied on
most when making its prediction. Works on the last convolutional block of
ResNet-18 (``layer4``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from model import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        # Global-average-pool the gradients to get channel weights.
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def _jet_colormap(gray: np.ndarray) -> np.ndarray:
    """Map a [0,1] array to an RGB jet-style colormap without matplotlib."""
    x = np.clip(gray, 0, 1)
    r = np.clip(1.5 - np.abs(4 * x - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def make_heatmap_images(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int | None = None,
    alpha: float = 0.45,
) -> Tuple[Image.Image, Image.Image]:
    """
    Return (heatmap_image, overlay_image) as PIL Images.
    """
    target_layer = model.layer4[-1]
    cam_engine = GradCAM(model, target_layer)
    try:
        cam = cam_engine(input_tensor, class_idx=class_idx)
    finally:
        cam_engine.remove()

    heat = _jet_colormap(cam)
    base = _denormalize(input_tensor)
    overlay = (alpha * heat + (1 - alpha) * base).astype(np.uint8)

    return Image.fromarray(heat), Image.fromarray(overlay)
