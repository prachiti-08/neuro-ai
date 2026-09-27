# ============================================================
# Neuro.ai - CT Prediction Module
# Uses the EXISTING trained best_model.pth
# NO RETRAINING
# ============================================================

import os

import torch
import torch.nn as nn

from PIL import Image

from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================
# 1. PATHS
# ============================================================

# Root project directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "best_model.pth"
)


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. CT MODEL ARCHITECTURE
# ============================================================

class StrokeNet(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(
            weights=None
        )

        # Freeze settings are not required
        # during inference.

        self.model.fc = nn.Sequential(

            nn.Dropout(
                p=0.25
            ),

            nn.Linear(
                self.model.fc.in_features,
                3
            )
        )


    def forward(self, x):

        return self.model(x)


# ============================================================
# 4. LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nCT model not found:\n{MODEL_PATH}"
    )


checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


model = StrokeNet().to(device)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


model.eval()


# ============================================================
# 5. CLASS NAMES
# ============================================================

# Prefer the classes saved inside the checkpoint.

if "classes" in checkpoint:

    CLASS_NAMES = checkpoint["classes"]

else:

    CLASS_NAMES = [
        "Normal",
        "Ischemic Stroke",
        "Hemorrhagic Stroke"
    ]


# ============================================================
# 6. IMAGE TRANSFORMATION
# ============================================================

# IMPORTANT:
# This is the SAME preprocessing used for
# validation and test images during training.

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


# ============================================================
# 7. CT PREDICTION FUNCTION
# ============================================================

def predict_ct(image):

    """
    Run CT prediction using the existing
    trained Neuro.ai CT model.

    Parameters
    ----------
    image : PIL.Image.Image
        Brain CT image.

    Returns
    -------
    dict
        Prediction results.
    """

    # --------------------------------------------------------
    # Convert image to RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )


    # --------------------------------------------------------
    # Apply preprocessing
    # --------------------------------------------------------

    image_tensor = transform(
        image
    )


    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(
        0
    )


    image_tensor = image_tensor.to(
        device
    )


    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        confidence, predicted_class = (
            torch.max(
                probabilities,
                dim=1
            )
        )


    predicted_index = (
        predicted_class.item()
    )


    confidence_value = (
        confidence.item() * 100
    )


    # --------------------------------------------------------
    # All class probabilities
    # --------------------------------------------------------

    class_probabilities = {}

    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        class_probabilities[
            class_name
        ] = round(
            probabilities[0][i].item() * 100,
            2
        )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    result = {

        "prediction":
            CLASS_NAMES[predicted_index],

        "confidence":
            round(
                confidence_value,
                2
            ),

        "probabilities":
            class_probabilities
    }


    return result


# ============================================================
# 8. TEST THE MODULE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "NEURO.AI CT PREDICTION MODULE"
    )

    print("=" * 60)


    print(
        f"\nModel path:\n{MODEL_PATH}"
    )


    print(
        f"\nDevice: {device}"
    )


    print(
        f"\nClasses: {CLASS_NAMES}"
    )


    print(
        "\nBest validation Macro F1:",
        round(
            checkpoint["val_f1"],
            4
        )
        if "val_f1" in checkpoint
        else "Not available"
    )


    print(
        "\nModel loaded successfully."
    )


    # --------------------------------------------------------
    # Test image
    # --------------------------------------------------------

    test_image_path = os.path.join(
        BASE_DIR,
        "test images",
        "test_image.png"
    )


    if os.path.exists(
        test_image_path
    ):

        print(
            f"\nTesting image:\n"
            f"{test_image_path}"
        )


        test_image = Image.open(
            test_image_path
        )


        result = predict_ct(
            test_image
        )


        print(
            "\nPrediction:",
            result["prediction"]
        )


        print(
            "Confidence:",
            f"{result['confidence']}%"
        )


        print(
            "\nClass probabilities:"
        )


        for class_name, probability in (
            result["probabilities"].items()
        ):

            print(
                f"{class_name}: "
                f"{probability}%"
            )


    else:

        print(
            "\nNo test image found."
        )


    print(
        "\n" + "=" * 60
    )

    print(
        "CT PREDICTION TEST COMPLETE"
    )

    print("=" * 60)