"""
NEURO.AI - MULTIMODAL FUSION
============================

Combines:
1. CT assessment
2. Facial palsy assessment
3. Clinical risk

This is a prototype decision-support layer.
It is NOT a clinical diagnostic system.
"""


# ============================================================
# FUSION
# ============================================================

def generate_fusion_result(
    ct_result,
    facial_result,
    clinical_result,
):
    """
    Combine CT, facial and clinical outputs.
    """

    ct_prediction = (
        ct_result.get(
            "prediction",
            "unknown"
        ).lower()
    )

    ct_confidence = (
        ct_result.get(
            "confidence",
            0.0
        )
    )

    facial_assessment = (
        facial_result.get(
            "facial_assessment",
            "Unknown"
        )
    )

    asymmetry = (
        facial_result.get(
            "asymmetry_score",
            0.0
        )
    )

    clinical_probability = (
        clinical_result.get(
            "risk_probability",
            0.0
        )
    )

    # --------------------------------------------------------
    # Basic interpretation
    # --------------------------------------------------------

    if ct_prediction == "hemorrhagic":

        stroke_type = "Hemorrhagic Stroke"

    elif ct_prediction == "ischemic":

        stroke_type = "Ischemic Stroke"

    elif ct_prediction == "normal":

        stroke_type = "No Stroke Detected"

    else:

        stroke_type = "Unknown"

    # --------------------------------------------------------
    # Overall interpretation
    # --------------------------------------------------------

    if ct_prediction in [
        "hemorrhagic",
        "ischemic"
    ]:

        overall_status = (
            "Stroke pattern detected on CT"
        )

    elif ct_prediction == "normal":

        overall_status = (
            "No stroke pattern detected on CT"
        )

    else:

        overall_status = (
            "Unable to determine"
        )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "stroke_type":
            stroke_type,

        "ct_prediction":
            ct_prediction,

        "ct_confidence":
            ct_confidence,

        "facial_assessment":
            facial_assessment,

        "facial_asymmetry":
            asymmetry,

        "clinical_risk_probability":
            clinical_probability,

        "overall_status":
            overall_status,
    }

if __name__ == "__main__":

    ct = {
        "prediction": "hemorrhagic",
        "confidence": 96.5
    }

    facial = {
        "facial_assessment": "Moderate asymmetry",
        "asymmetry_score": 0.14,
        "asymmetry_percentage": 14.0
    }

    clinical = {
        "risk_probability": 0.618,
        "risk_category": "High"
    }

    result = generate_fusion_result(
        ct,
        facial,
        clinical
    )

    print("\nNEURO.AI FUSION RESULT")
    print("=" * 50)

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )