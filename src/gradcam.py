"""
FACIAL PALSY GRAD-CAM
=====================

Grad-CAM explainability for the recovered ResNet18
facial palsy severity model.

Target layer:
    backbone.layer4   (for wrapped model)
    layer4             (for raw ResNet18 inference model)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ============================================================================
# CONFIGURATION
# ============================================================================

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "Moderate Severe",
    "Severe",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# PREPROCESSING
# ============================================================================

def get_transform():

    return transforms.Compose([
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])


# ============================================================================
# FIND RESNET18 LAYER4
# ============================================================================

def get_target_layer(model):
    """
    Supports both:

        model.layer4

    and:

        model.backbone.layer4
    """

    if hasattr(model, "layer4"):
        return model.layer4

    if hasattr(model, "backbone"):
        if hasattr(model.backbone, "layer4"):
            return model.backbone.layer4

    raise RuntimeError(
        "Could not find ResNet18 layer4 in model."
    )


# ============================================================================
# GRAD-CAM
# ============================================================================

class GradCAM:

    def __init__(self, model):

        self.model = model
        self.model.eval()

        self.activations = None
        self.gradients = None

        self.target_layer = get_target_layer(
            model
        )

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self._save_activation
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self._save_gradient
            )
        )

    def _save_activation(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output.detach()

    def _save_gradient(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = (
            grad_output[0].detach()
        )

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        image,
        target_class=None,
    ):
        """
        Generate Grad-CAM heatmap.

        Returns:
            heatmap
            predicted_class
            confidence
            probabilities
        """

        if not isinstance(
            image,
            Image.Image
        ):

            image = Image.open(
                image
            )

        image = image.convert(
            "RGB"
        )

        transform = get_transform()

        tensor = transform(
            image
        ).unsqueeze(
            0
        ).to(
            DEVICE
        )

        self.model.zero_grad(
            set_to_none=True
        )

        # ------------------------------------------------------------
        # Forward
        # ------------------------------------------------------------

        output = self.model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = int(
            probabilities.argmax(
                dim=1
            ).item()
        )

        confidence = float(
            probabilities[
                0,
                predicted_class
            ].item()
        )

        # If no target specified, explain prediction.
        if target_class is None:
            target_class = predicted_class

        # ------------------------------------------------------------
        # Backward
        # ------------------------------------------------------------

        score = output[
            0,
            target_class
        ]

        score.backward()

        # ------------------------------------------------------------
        # Activations and gradients
        # ------------------------------------------------------------

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activations = self.activations
        gradients = self.gradients

        # ------------------------------------------------------------
        # Global-average-pool gradients
        # ------------------------------------------------------------

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ------------------------------------------------------------
        # Weighted activation maps
        # ------------------------------------------------------------

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(
            cam
        )

        # ------------------------------------------------------------
        # Resize to image dimensions
        # ------------------------------------------------------------

        cam = F.interpolate(
            cam,
            size=(
                IMAGE_SIZE,
                IMAGE_SIZE
            ),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()

        # ------------------------------------------------------------
        # Normalize 0 → 1
        # ------------------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:

            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )

        else:

            cam = np.zeros_like(
                cam
            )

        return {
            "heatmap": cam,
            "predicted_class":
                predicted_class,
            "predicted_label":
                CLASS_NAMES[
                    predicted_class
                ],
            "confidence":
                confidence,
            "probabilities": {
                CLASS_NAMES[i]:
                    float(
                        probabilities[
                            0,
                            i
                        ].item()
                    )
                for i in range(
                    len(CLASS_NAMES)
                )
            },
        }


# ============================================================================
# HEATMAP OVERLAY
# ============================================================================

def create_heatmap_overlay(
    image,
    heatmap,
    alpha=0.45,
):
    """
    Create a visual Grad-CAM overlay.

    Returns a PIL RGB image.
    """

    image = image.convert(
        "RGB"
    )

    image = image.resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image_array = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    # Simple heatmap using matplotlib.
    import matplotlib.cm as cm

    heatmap_rgb = cm.jet(
        heatmap
    )[:, :, :3]

    overlay = (
        (1 - alpha)
        * image_array
        +
        alpha
        * heatmap_rgb
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    overlay = (
        overlay * 255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        overlay
    )


# ============================================================================
# SAVE GRAD-CAM IMAGE
# ============================================================================

def save_gradcam(
    image,
    output_path,
    model,
    target_class=None,
):

    if not isinstance(
        image,
        Image.Image
    ):

        image = Image.open(
            image
        )

    image = image.convert(
        "RGB"
    )

    gradcam = GradCAM(
        model
    )

    try:

        result = gradcam.generate(
            image,
            target_class=target_class
        )

    finally:

        gradcam.remove_hooks()

    overlay = create_heatmap_overlay(
        image,
        result["heatmap"]
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    overlay.save(
        output_path
    )

    result["overlay_path"] = str(
        output_path
    )

    return result