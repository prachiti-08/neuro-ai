"""
Facial Palsy Severity Model
===========================

Architecture matching the recovered final_model.pth:

- ResNet18
- ImageNet pretrained backbone
- Fine-tuned layers:
    layer3
    layer4
    fc
- 4 severity classes
"""

import torch
import torch.nn as nn
from torchvision import models


# ============================================================
# CONFIGURATION
# ============================================================

CLASSES = [
    "Mild",
    "Moderate",
    "Moderate Severe",
    "Severe",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(CLASSES)
}

NUM_CLASSES = len(CLASSES)


# ============================================================
# MODEL
# ============================================================

class FacialPalsyResNet18(nn.Module):

    def __init__(self):

        super().__init__()

        # Create ResNet18 architecture.
        #
        # We don't need pretrained weights here because the trained
        # parameters will be loaded from final_model.pth.
        self.backbone = models.resnet18(
            weights=None
        )

        # Replace the original 1000-class ImageNet classifier.
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(
                in_features,
                NUM_CLASSES
            )
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# FACTORY
# ============================================================

def create_model():

    model = FacialPalsyResNet18()

    return model