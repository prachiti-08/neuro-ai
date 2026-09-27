# ============================================================
# NEURO.AI - STROKE ANALYSIS REPORT
# ============================================================

from datetime import datetime


# ============================================================
# GENERATE STROKE REPORT
# ============================================================

def generate_stroke_report(case):
    """
    Generates a complete Neuro.ai multimodal
    stroke analysis report.
    """

    # ========================================================
    # PATIENT INFORMATION
    # ========================================================

    patient_id = case.get(
        "patient_id",
        "Not available"
    )

    patient_name = case.get(
        "patient_name",
        "Not available"
    )

    age = case.get(
        "age",
        "Not available"
    )

    gender = case.get(
        "gender",
        "Not available"
    )

    # ========================================================
    # CT INFORMATION
    # ========================================================

    ct_file = case.get(
        "ct_file",
        "Not available"
    )

    ct_metadata = case.get(
        "ct_metadata",
        "Not available"
    )

    # ========================================================
    # CT PREDICTION
    # ========================================================

    prediction = case.get(
        "prediction",
        "Not available"
    )

    confidence = float(
        case.get(
            "confidence",
            0
        )
    )

    probabilities = case.get(
        "probabilities",
        {
            "Ischemic Stroke": 0,
            "Hemorrhagic Stroke": 0,
            "Normal": 0
        }
    )

    # ========================================================
    # FACIAL ASSESSMENT
    # ========================================================

    facial_assessment = case.get(
        "facial_assessment",
        "Not available"
    )

    facial_asymmetry = float(
        case.get(
            "facial_asymmetry",
            0
        )
    )


    # ========================================================
    # CLINICAL RISK
    # ========================================================

    clinical_risk_probability = float(
        case.get(
            "clinical_risk_probability",
            0
        )
    )


    # ========================================================
    # CLINICAL INPUTS
    # ========================================================

    clinical_inputs = case.get(
        "clinical_inputs",
        {}
    )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    gradcam = case.get(
        "gradcam",
        "Grad-CAM visualization not available."
    )

    # ========================================================
    # LESION ANALYSIS
    # ========================================================

    lesion_analysis = case.get(
        "lesion_analysis",
        "Lesion analysis is not available."
    )

    # ========================================================
    # FUSION RESULT
    # ========================================================

    overall_status = case.get(
        "overall_status",
        "Not available"
    )

    stroke_type = case.get(
        "stroke_type",
        prediction
    )

    # ========================================================
    # PRIORITY
    # ========================================================

    if confidence >= 90:

        priority = "High"

    elif confidence >= 70:

        priority = "Medium"

    else:

        priority = "Low"

    # ========================================================
    # AI SUMMARY
    # ========================================================

    ai_summary = (
        f"Neuro.ai analyzed the available CT, facial and "
        f"clinical information. The CT model predicted "
        f"{prediction} with a confidence of "
        f"{confidence:.2f}%. "

        f"The live facial assessment recorded "
        f"{facial_assessment} with an estimated facial "
        f"asymmetry of {facial_asymmetry_percentage:.2f}%. "

        f"The clinical risk model produced an estimated "
        f"risk probability of "
        f"{clinical_risk_percentage:.2f}%. "

        f"The overall system status was: "
        f"{overall_status}."
    )

    # ========================================================
    # DATE AND TIME
    # ========================================================

    timestamp = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    # ========================================================
    # BUILD REPORT
    # ========================================================

    report = f"""
============================================================
                     NEURO.AI
              STROKE ANALYSIS REPORT
============================================================

REPORT DATE & TIME
------------------
{timestamp}


1. PATIENT INFORMATION
----------------------
Patient ID      : {patient_id}
Patient Name    : {patient_name}
Age             : {age}
Gender          : {gender}


2. CT ANALYSIS
--------------
CT File        : {ct_file}
CT Metadata    : {ct_metadata}


3. CT AI PREDICTION
-------------------
Predicted Condition : {prediction}
Confidence          : {confidence:.2f}%


4. CT CLASS PROBABILITIES
-------------------------
"""

    for class_name, probability in (
        probabilities.items()
    ):

        report += (
            f"{class_name:<25}: "
            f"{float(probability):.2f}%\n"
        )

    report += f"""

5. FACIAL PALSY ASSESSMENT
--------------------------
Assessment          : {facial_assessment}
Facial Asymmetry    : {facial_asymmetry_percentage:.2f}%


6. CLINICAL RISK ASSESSMENT
---------------------------
Clinical Risk Probability : {clinical_risk_percentage:.2f}%


7. CLINICAL INPUTS
------------------
"""

    if clinical_inputs:

        for field, value in (
            clinical_inputs.items()
        ):

            report += (
                f"{field:<25}: "
                f"{value}\n"
            )

    else:

        report += (
            "No clinical inputs available.\n"
        )

    report += f"""

8. GRAD-CAM ANALYSIS
--------------------
{gradcam}


9. LESION ANALYSIS
------------------
{lesion_analysis}


10. MULTIMODAL FUSION RESULT
----------------------------
Stroke Type / CT Result : {stroke_type}

Overall System Status   : {overall_status}

Priority Level          : {priority}


11. NEURO.AI SUMMARY
--------------------
{ai_summary}


12. LIMITATIONS AND DISCLAIMER
------------------------------
This report is generated using an AI-based decision-support
prototype and is intended to assist clinical review.

The CT prediction, facial assessment and clinical risk
estimation should not be considered definitive medical
diagnoses.

The clinical risk probability is an estimate produced by
the trained clinical model and should not be interpreted
as a validated individual clinical risk score.

Facial asymmetry measurements are based on the prototype
live-camera assessment and are not standalone diagnostic
criteria.

Grad-CAM visualizations are interpretation aids and should
not be considered definitive evidence of a lesion.

Clinical decisions should be made by qualified healthcare
professionals using appropriate clinical, imaging and
diagnostic information.


============================================================
                    END OF REPORT
============================================================
"""

    return report


# ============================================================
# SAMPLE DATA FOR TESTING
# ============================================================

if __name__ == "__main__":

    sample_case = {

        # ----------------------------------------------------
        # Patient
        # ----------------------------------------------------

        "patient_id":
            "NV001",

        "patient_name":
            "Sample Patient",

        "age":
            62,

        "gender":
            "Male",

        # ----------------------------------------------------
        # CT
        # ----------------------------------------------------

        "ct_file":
            "sample_ct_scan.png",

        "ct_metadata":
            "CT brain scan - sample image",

        "prediction":
            "Hemorrhagic Stroke",

        "confidence":
            96.5,

        "probabilities": {

            "Hemorrhagic Stroke":
                96.5,

            "Ischemic Stroke":
                1.21,

            "Normal":
                2.29
        },

        # ----------------------------------------------------
        # Facial
        # ----------------------------------------------------

        "facial_assessment":
            "Moderate asymmetry",

        "facial_asymmetry":
            0.14,

        # ----------------------------------------------------
        # Clinical
        # ----------------------------------------------------

        "clinical_risk_probability":
            0.618,

        "clinical_inputs": {

            "Blood Pressure":
                "150/95 mmHg",

            "Symptoms":
                "Sudden weakness and speech difficulty",

            "Medical History":
                "Hypertension"
        },

        # ----------------------------------------------------
        # Explainability
        # ----------------------------------------------------

        "gradcam":
            (
                "Grad-CAM visualization available. "
                "Highlighted regions represent areas that "
                "contributed to the model prediction."
            ),

        "lesion_analysis":
            (
                "Potential abnormal region identified by "
                "the AI analysis. Further clinical "
                "evaluation is required."
            ),

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        "stroke_type":
            "Hemorrhagic Stroke",

        "overall_status":
            "Stroke pattern detected on CT"
    }

    # ========================================================
    # GENERATE REPORT
    # ========================================================

    report = generate_stroke_report(
        sample_case
    )

    print(report)