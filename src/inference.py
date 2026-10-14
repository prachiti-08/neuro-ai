"""
FACIAL PALSY INFERENCE
======================

Loads the existing final_model.pth without retraining.

The saved model contains a ResNet18 state_dict whose keys are:
    conv1.weight
    layer1...
    layer4...
    fc...

This module reconstructs the same architecture and loads those weights.
"""

from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "outputs"
    / "models"
    / "final_model.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "Moderate Severe",
    "Severe",
]


# ============================================================================
# MODEL
# ============================================================================

def create_inference_model():
    """
    Recreate the ResNet18 architecture used by the saved model.
    """

    model = models.resnet18(
        weights=None
    )

    # The saved model has a sequential FC head.
    model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.3),

    torch.nn.Linear(
        512,
        256
    ),

    torch.nn.ReLU(),

    torch.nn.Dropout(p=0.3),

    torch.nn.Linear(
        256,
        4
    )
)

    return model


# ============================================================================
# LOAD MODEL
# ============================================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # ------------------------------------------------------------------------
    # final_model.pth format
    # ------------------------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        state_dict = checkpoint

    model = create_inference_model()

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # The saved state_dict uses:
    #     conv1.weight
    #
    # NOT:
    #     backbone.conv1.weight
    #
    # Therefore we load directly into torchvision ResNet18.
    # ------------------------------------------------------------------------

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        f"✓ Model loaded: {MODEL_PATH}"
    )

    print(
        f"✓ Device: {DEVICE}"
    )

    return model


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
# PREDICTION
# ============================================================================

def predict(
    image,
    model=None
):

    if model is None:

        model = load_model()

    transform = get_transform()

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

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = (
            probabilities.max(
                dim=1
            )
        )

    predicted_index = (
        predicted.item()
    )

    confidence_value = (
        confidence.item()
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_index
        ]
    )

    probabilities_dict = {
        CLASS_NAMES[i]:
            float(probabilities[0][i])
        for i in range(
            len(CLASS_NAMES)
        )
    }

    return {
        "class": predicted_class,
        "class_index": predicted_index,
        "confidence": confidence_value,
        "probabilities": probabilities_dict,
    }