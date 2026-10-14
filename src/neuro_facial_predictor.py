"""
NEURO.AI - FACIAL PALSY PREDICTION MODULE
==========================================

Reusable facial palsy inference module for Neuro.ai.

Uses the existing trained final_model.pth through
src.inference without retraining.
"""

from pathlib import Path
import sys


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(BASE_DIR)
)


# ============================================================
# EXISTING MODEL INFERENCE
# ============================================================

from src.inference import (
    load_model,
    predict,
)


# ============================================================
# MODEL
# ============================================================

MODEL = None


def initialize_model():
    """
    Load the existing facial palsy model once.
    """

    global MODEL

    if MODEL is None:

        print("=" * 65)
        print("NEURO.AI FACIAL PALSY PREDICTION MODULE")
        print("=" * 65)

        MODEL = load_model()

        print("Model loaded successfully.\n")

    return MODEL


# ============================================================
# PREDICTION
# ============================================================

def predict_facial_palsy(image):
    """
    Predict facial palsy severity from an image.

    Parameters
    ----------
    image:
        Image path or PIL Image.

    Returns
    -------
    dict
        Prediction, confidence and class probabilities.
    """

    model = initialize_model()

    result = predict(
        image,
        model=model
    )

    return {
        "prediction": result["class"],

        "confidence": round(
            result["confidence"] * 100,
            2
        ),

        "probabilities": {
            class_name: round(
                probability * 100,
                2
            )
            for class_name, probability
            in result["probabilities"].items()
        }
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Change this only if you want to test
    # another facial image.

    test_image = (
        BASE_DIR.parent
        / "test images"
        / "facial_test.png"
    )

    if not test_image.exists():

        print(
            "No test image found."
        )

        print(
            "\nExpected test image:"
        )

        print(test_image)

        print(
            "\nCreate/place a facial image there "
            "and run this file again."
        )

    else:

        result = predict_facial_palsy(
            test_image
        )

        print("=" * 65)
        print("FACIAL PALSY PREDICTION")
        print("=" * 65)

        print(
            f"\nPrediction: "
            f"{result['prediction']}"
        )

        print(
            f"Confidence: "
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

        print("\n" + "=" * 65)
        print(
            "FACIAL PREDICTION TEST COMPLETE"
        )
        print("=" * 65)