# ============================================================
# Neuro.ai - Multimodal AI Controller
# ============================================================
#
# Modules:
#   1. CT Stroke Analysis
#   2. Facial Palsy Live-Camera Assessment
#   3. Clinical Risk Analysis
#   4. Multimodal Fusion
#   5. Case Database
#
# Later:
#   6. Clinical Assistant
#   7. Stroke Report
#   8. PDF Generator
#   9. Streamlit UI
#
# ============================================================

import sys
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import cv2

from PIL import Image
from torchvision import models, transforms
from pdf_generator import generate_pdf_report
from clinical_assistant import assistant_response

# ============================================================
# CASE DATABASE
# ============================================================

from case_database import initialize_database, add_case


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "model"
    / "best_model.pth"
)

CLINICAL_SRC = (
    BASE_DIR
    / "clinical module"
    / "src"
)

FACIAL_MODULE_DIR = (
    BASE_DIR
    / "facial_palsy"
)


# ============================================================
# 2. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 3. CT CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "hemorrhagic",
    "ischemic",
    "normal"
]


DISPLAY_NAMES = {

    "hemorrhagic":
        "Hemorrhagic Stroke",

    "ischemic":
        "Ischemic Stroke",

    "normal":
        "Normal"
}


# ============================================================
# 4. CT IMAGE TRANSFORMATION
# ============================================================

CT_TRANSFORM = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# 5. CT MODEL ARCHITECTURE
# ============================================================

class StrokeInferenceModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet18(
            weights=None
        )

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
# 6. CREATE CT MODEL
# ============================================================

def create_ct_model():

    return StrokeInferenceModel()


# ============================================================
# 7. LOAD CT MODEL
# ============================================================

def load_ct_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nCT model not found at:\n"
            f"{MODEL_PATH}"
        )


    model = create_ct_model()


    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )


    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    try:

        model.load_state_dict(
            state_dict,
            strict=True
        )

    except RuntimeError:

        new_state_dict = {}

        for key, value in (
            state_dict.items()
        ):

            if key.startswith(
                "model."
            ):

                new_key = key

            else:

                new_key = (
                    "model."
                    + key
                )

            new_state_dict[
                new_key
            ] = value


        model.load_state_dict(
            new_state_dict,
            strict=True
        )


    model.to(
        DEVICE
    )

    model.eval()


    return model


# ============================================================
# 8. GENERATE CT GRAD-CAM
# ============================================================

def generate_ct_gradcam(
    model,
    image_tensor
):

    from ct_module.gradcam_utils import (
        generate_gradcam
    )

    return generate_gradcam(
        model,
        image_tensor
    )


# ============================================================
# 9. CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    image,
    cam
):

    image_np = np.array(
        image.resize(
            (224, 224)
        )
    )


    if image_np.ndim == 2:

        image_np = cv2.cvtColor(
            image_np,
            cv2.COLOR_GRAY2RGB
        )


    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY
    )


    # --------------------------------------------------------
    # Brain mask
    # --------------------------------------------------------

    brain_mask = (
        gray > 20
    ).astype(
        np.uint8
    ) * 255


    # --------------------------------------------------------
    # Smooth CAM
    # --------------------------------------------------------

    cam_smooth = cv2.GaussianBlur(
        cam,
        (9, 9),
        0
    )


    cam_smooth[
        cam_smooth < 0.20
    ] = 0


    cam_smooth = (
        cam_smooth
        * (brain_mask / 255.0)
    )


    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heatmap = np.uint8(
        255 * cam_smooth
    )


    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )


    overlay = cv2.addWeighted(
        image_bgr,
        0.65,
        heatmap,
        0.35,
        0
    )


    overlay = cv2.cvtColor(
        overlay,
        cv2.COLOR_BGR2RGB
    )


    return overlay


# ============================================================
# 10. CT PREDICTION
# ============================================================

def predict_ct(
    image,
    model=None
):

    if model is None:

        model = load_ct_model()


    image_rgb = (
        image.convert("RGB")
    )


    image_tensor = CT_TRANSFORM(
        image_rgb
    ).unsqueeze(0)


    image_tensor = (
        image_tensor.to(
            DEVICE
        )
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            image_tensor
        )


        probabilities = (
            torch.softmax(
                output,
                dim=1
            )[0]
        )


        predicted_index = (
            torch.argmax(
                probabilities
            ).item()
        )


        confidence = (
            probabilities[
                predicted_index
            ].item()
            * 100
        )


    raw_prediction = (
        CLASS_NAMES[
            predicted_index
        ]
    )


    display_prediction = (
        DISPLAY_NAMES[
            raw_prediction
        ]
    )


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probability_dict = {}


    for index, class_name in (
        enumerate(CLASS_NAMES)
    ):

        probability_dict[
            DISPLAY_NAMES[
                class_name
            ]
        ] = round(
            probabilities[
                index
            ].item()
            * 100,
            2
        )


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gradcam = generate_ct_gradcam(
        model,
        image_tensor
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    gradcam_overlay = (
        create_gradcam_overlay(
            image_rgb,
            gradcam
        )
    )


    return {

        "prediction":
            display_prediction,

        "raw_prediction":
            raw_prediction,

        "confidence":
            round(
                confidence,
                2
            ),

        "probabilities":
            probability_dict,

        "gradcam":
            gradcam,

        "gradcam_overlay":
            gradcam_overlay
    }


# ============================================================
# 11. CLINICAL MODULE
# ============================================================

if str(
    CLINICAL_SRC
) not in sys.path:

    sys.path.insert(
        0,
        str(CLINICAL_SRC)
    )


from clinical_predictor import (
    predict_clinical_risk
)


# ============================================================
# 12. CLINICAL PREDICTION
# ============================================================

def predict_clinical(
    patient_data
):

    result = (
        predict_clinical_risk(
            patient_data
        )
    )


    return {

        "prediction":
            result[
                "prediction"
            ],

        "clinical_risk_probability":
            result[
                "clinical_risk_probability"
            ],

        "risk_category":
            result[
                "risk_category"
            ],

        "interpretation":
            result[
                "interpretation"
            ]
    }


# ============================================================
# 13. FACIAL MODULE
# ============================================================

if str(
    FACIAL_MODULE_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(FACIAL_MODULE_DIR)
    )


from live.live_camera import (
    run_live_assessment
)


# ============================================================
# 14. RUN FACIAL ASSESSMENT
# ============================================================

def predict_facial():

    """
    Launch the existing live-camera
    facial palsy assessment.

    The webcam starts only when this
    function is called.
    """

    print("\n")
    print("=" * 60)
    print(
        "STARTING LIVE FACIAL ASSESSMENT"
    )
    print("=" * 60)


    result = (
        run_live_assessment()
    )


    if result is None:

        print(
            "\nFacial assessment "
            "was cancelled."
        )

        return None


    return {

        "facial_assessment":
            result[
                "facial_assessment"
            ],

        "asymmetry_score":
            result[
                "asymmetry_score"
            ],

        "asymmetry_percentage":
            result[
                "asymmetry_percentage"
            ],

        "step_results":
            result.get(
                "step_results",
                {}
            )
    }


# ============================================================
# 15. THREE-MODALITY FUSION
# ============================================================

def fuse_multimodal_results(
    ct_result,
    facial_result,
    clinical_result
):

    """
    Combine the three Neuro.ai modalities.

    CT:
        Imaging-based stroke classification.

    Facial:
        Live facial asymmetry assessment.

    Clinical:
        Clinical stroke-risk indicator.

    This fusion is a prototype integration
    and should not be treated as a
    clinically validated diagnostic system.
    """


    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

    ct_prediction = (
        ct_result[
            "prediction"
        ]
    )

    ct_confidence = (
        ct_result[
            "confidence"
        ]
    )


    # --------------------------------------------------------
    # Facial
    # --------------------------------------------------------

    facial_assessment = (
        facial_result[
            "facial_assessment"
        ]
    )

    facial_asymmetry = (
        facial_result[
            "asymmetry_percentage"
        ]
    )


    # --------------------------------------------------------
    # Clinical
    # --------------------------------------------------------

    clinical_probability = (
        clinical_result[
            "clinical_risk_probability"
        ]
    )

    clinical_category = (
        clinical_result[
            "risk_category"
        ]
    )


    # --------------------------------------------------------
    # Overall CT status
    # --------------------------------------------------------

    if (
        ct_prediction
        != "Normal"
    ):

        overall_status = (
            "Stroke pattern detected on CT"
        )

    else:

        overall_status = (
            "No stroke pattern detected on CT"
        )


    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {

        "ct_prediction":
            ct_prediction,

        "ct_confidence":
            ct_confidence,

        "facial_assessment":
            facial_assessment,

        "facial_asymmetry":
            facial_asymmetry,

        "clinical_risk_probability":
            clinical_probability,

        "clinical_risk_category":
            clinical_category,

        "overall_status":
            overall_status
    }


# ============================================================
# 16. COMPLETE MULTIMODAL ANALYSIS
# ============================================================

def analyze_multimodal(
    image,
    patient_data,
    run_facial=True,
    ct_model=None
):

    """
    Run CT, facial and clinical analysis.
    """

    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

    ct_result = predict_ct(
        image,
        ct_model
    )


    # --------------------------------------------------------
    # Clinical
    # --------------------------------------------------------

    clinical_result = (
        predict_clinical(
            patient_data
        )
    )


    # --------------------------------------------------------
    # Facial
    # --------------------------------------------------------

    facial_result = None


    if run_facial:

        facial_result = (
            predict_facial()
        )


    # --------------------------------------------------------
    # Fusion
    # --------------------------------------------------------

    if facial_result is not None:

        fusion_result = (
            fuse_multimodal_results(
                ct_result,
                facial_result,
                clinical_result
            )
        )

    else:

        fusion_result = (
            fuse_ct_and_clinical(
                ct_result,
                clinical_result
            )
        )


    # --------------------------------------------------------
    # Return everything
    # --------------------------------------------------------

    return {

        "ct":
            ct_result,

        "facial":
            facial_result,

        "clinical":
            clinical_result,

        "fusion":
            fusion_result
    }


# ============================================================
# 17. CT + CLINICAL FUSION
# ============================================================

def fuse_ct_and_clinical(
    ct_result,
    clinical_result
):

    ct_prediction = (
        ct_result[
            "prediction"
        ]
    )

    ct_confidence = (
        ct_result[
            "confidence"
        ]
    )

    clinical_probability = (
        clinical_result[
            "clinical_risk_probability"
        ]
    )

    clinical_category = (
        clinical_result[
            "risk_category"
        ]
    )


    if (
        ct_prediction
        != "Normal"
    ):

        overall_status = (
            "Stroke pattern detected on CT"
        )

    else:

        overall_status = (
            "No stroke pattern detected on CT"
        )


    return {

        "ct_prediction":
            ct_prediction,

        "ct_confidence":
            ct_confidence,

        "clinical_risk_probability":
            clinical_probability,

        "clinical_risk_category":
            clinical_category,

        "overall_status":
            overall_status
    }


# ============================================================
# 18. THREE-MODALITY TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("NEURO.AI THREE-MODALITY TEST")
    print("=" * 60)


    # ========================================================
    # LOAD CT MODEL
    # ========================================================

    print("\nLoading CT model...")

    ct_model = load_ct_model()

    print("✓ CT model loaded")
    print(f"✓ Device: {DEVICE}")


    # ========================================================
    # TEST CT IMAGE
    # ========================================================

    test_image_path = (
        BASE_DIR
        / "test images"
        / "test_image.png"
    )

    if not test_image_path.exists():

        raise FileNotFoundError(
            f"\nTest CT image not found:\n"
            f"{test_image_path}"
        )

    image = (
        Image.open(
            test_image_path
        ).convert("RGB")
    )


    # ========================================================
    # CT ANALYSIS
    # ========================================================

    print("\n")
    print("-" * 60)
    print("CT ANALYSIS")
    print("-" * 60)

    ct_result = predict_ct(
        image,
        ct_model
    )

    print(
        f"\nPrediction: "
        f"{ct_result['prediction']}"
    )

    print(
        f"Confidence: "
        f"{ct_result['confidence']:.2f}%"
    )

    print("\nClass probabilities:")

    for name, probability in (
        ct_result["probabilities"].items()
    ):

        print(
            f"{name}: "
            f"{probability:.2f}%"
        )

    print(
        f"\nGrad-CAM generated: "
        f"{ct_result['gradcam'].shape}"
    )

    print(
        f"Overlay generated: "
        f"{ct_result['gradcam_overlay'].shape}"
    )


    # ========================================================
    # CLINICAL PATIENT DATA
    # ========================================================

    patient = {

        "gender":
            "Male",

        "age":
            65,

        "hypertension":
            1,

        "heart_disease":
            0,

        "ever_married":
            "Yes",

        "work_type":
            "Private",

        "Residence_type":
            "Urban",

        "avg_glucose_level":
            180.0,

        "bmi":
            27.5,

        "smoking_status":
            "formerly smoked"
    }


    # ========================================================
    # CLINICAL ANALYSIS
    # ========================================================

    print("\n")
    print("-" * 60)
    print("CLINICAL ANALYSIS")
    print("-" * 60)

    clinical_result = predict_clinical(
        patient
    )

    print(
        f"\nClinical Prediction: "
        f"{clinical_result['prediction']}"
    )

    print(
        f"Clinical Risk Probability: "
        f"{clinical_result['clinical_risk_probability']:.2f}%"
    )

    print(
        f"Risk Category: "
        f"{clinical_result['risk_category']}"
    )


    # ========================================================
    # FACIAL PALSY ASSESSMENT
    # ========================================================

    print("\n")
    print("-" * 60)
    print("FACIAL PALSY ASSESSMENT")
    print("-" * 60)

    print(
        "\nThe live camera will now start."
    )

    print(
        "Complete all 5 facial assessment steps."
    )

    print(
        "The result will be returned "
        "automatically after completion."
    )

    facial_result = predict_facial()


    # --------------------------------------------------------
    # Handle cancelled facial assessment
    # --------------------------------------------------------

    if facial_result is None:

        print(
            "\nFacial assessment "
            "was not completed."
        )

        print(
            "\nCT + Clinical analysis "
            "remains available."
        )

        sys.exit(0)


    print("\nFacial Result:")

    print(
        f"Facial Assessment: "
        f"{facial_result['facial_assessment']}"
    )

    print(
        f"Asymmetry Score: "
        f"{facial_result['asymmetry_score']}"
    )

    print(
        f"Asymmetry Percentage: "
        f"{facial_result['asymmetry_percentage']}%"
    )


    # ========================================================
    # THREE-MODALITY FUSION
    # ========================================================

    print("\n")
    print("-" * 60)
    print("NEURO.AI MULTIMODAL FUSION")
    print("-" * 60)

    fusion_result = fuse_multimodal_results(
        ct_result,
        facial_result,
        clinical_result
    )

    print(
        f"\nCT Pattern: "
        f"{fusion_result['ct_prediction']}"
    )

    print(
        f"CT Confidence: "
        f"{fusion_result['ct_confidence']:.2f}%"
    )

    print(
        f"Facial Assessment: "
        f"{fusion_result['facial_assessment']}"
    )

    print(
        f"Facial Asymmetry: "
        f"{fusion_result['facial_asymmetry']:.2f}%"
    )

    print(
        f"Clinical Risk: "
        f"{fusion_result['clinical_risk_probability']:.2f}%"
    )

    print(
        f"Clinical Category: "
        f"{fusion_result['clinical_risk_category']}"
    )

    print(
        f"Overall Status: "
        f"{fusion_result['overall_status']}"
    )


    # ========================================================
    # SAVE CASE TO DATABASE
    # ========================================================

    print("\n")
    print("-" * 60)
    print("SAVING CASE")
    print("-" * 60)

    case = {

        "patient_id":
            "NV001",

        "patient_name":
            "Sample Patient",

        "age":
            patient["age"],

        "gender":
            patient["gender"],

        "ct_file":
            str(test_image_path),

        "ct_metadata":
            "CT brain scan - sample image",

        "prediction":
            ct_result["prediction"],

        "confidence":
            ct_result["confidence"],

        "probabilities":
            ct_result["probabilities"],

        "facial_assessment":
            facial_result[
                "facial_assessment"
            ],

        "facial_asymmetry":
            facial_result[
                "asymmetry_percentage"
            ],

        "clinical_risk_probability":
            clinical_result[
                "clinical_risk_probability"
            ],

        "clinical_inputs":
            patient,

        "gradcam":
            ct_result["gradcam"],

        "lesion_analysis":
            (
                "Potential abnormal region identified "
                "by AI analysis. Further clinical "
                "evaluation is required."
            ),

        "overall_status":
            fusion_result[
                "overall_status"
            ],

        "report_file":
            ""
    }


    # ========================================================
    # INITIALIZE DATABASE
    # ========================================================

    initialize_database()


    # ========================================================
    # SAVE CASE
    # ========================================================

    case_id = add_case(
        case
    )

    print("\n")
    print("=" * 60)
    print("CASE SAVED SUCCESSFULLY")
    print("=" * 60)
    print(f"Case ID: {case_id}")


    # ========================================================
    # GENERATE PDF REPORT
    # ========================================================

    print("\n")
    print("-" * 60)
    print("GENERATING PDF REPORT")
    print("-" * 60)

    pdf_filename = (
        f"NeuroAI_Case_{case_id}.pdf"
    )

    pdf_path = generate_pdf_report(
        case,
        pdf_filename
    )

    print("\n")
    print("✓ PDF REPORT GENERATED SUCCESSFULLY")
    print(f"Report: {pdf_path}")


    # ========================================================
    # CLINICAL ASSISTANT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("NEURO.AI CLINICAL ASSISTANT")
    print("=" * 60)

    print(
        "\nAsk questions about the current Neuro.ai case."
    )

    print(
        "Type 'exit' to close the assistant."
    )

    print()


    while True:

        question = input(
            "You: "
        ).strip()


        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if question.lower() == "exit":

            print()
            print(
                "Clinical Assistant closed."
            )

            break


        # ----------------------------------------------------
        # Ignore empty input
        # ----------------------------------------------------

        if not question:

            continue


        # ----------------------------------------------------
        # Generate assistant response
        # ----------------------------------------------------

        response = assistant_response(
            question,
            case
        )

        print()
        print("Neuro.ai:")
        print(response)
        print()


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print(
        "NEURO.AI THREE-MODALITY TEST COMPLETE"
    )
    print("=" * 60)