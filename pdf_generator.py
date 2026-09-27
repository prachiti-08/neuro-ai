# ============================================================
# NEURO.AI - PDF REPORT GENERATOR
# ============================================================

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.units import mm
from datetime import datetime


# ============================================================
# GENERATE PDF REPORT
# ============================================================

def generate_pdf_report(
    case,
    filename="neurovision_stroke_report.pdf"
):
    """
    Generates a complete Neuro.ai multimodal PDF report.

    Includes:
        - Patient information
        - CT analysis
        - CT probabilities
        - Facial assessment
        - Clinical risk
        - Clinical inputs
        - Grad-CAM
        - Lesion analysis
        - Multimodal fusion
        - AI summary
        - Disclaimer
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

    facial_asymmetry_percentage = float(
    case.get(
        "facial_asymmetry",
        0
    )
)


    # ========================================================
    # CLINICAL INFORMATION
    # ========================================================

    clinical_risk_percentage = float(
    case.get(
        "clinical_risk_probability",
        0
    )
)

    clinical_inputs = case.get(
        "clinical_inputs",
    {}
)

    # ========================================================
    # EXPLAINABILITY
    # ========================================================

    gradcam = case.get(
        "gradcam",
        "Grad-CAM visualization not available."
    )

    lesion_analysis = case.get(
        "lesion_analysis",
        "Lesion analysis is not available."
    )

    # ========================================================
    # FUSION
    # ========================================================

    stroke_type = case.get(
        "stroke_type",
        prediction
    )

    overall_status = case.get(
        "overall_status",
        "Not available"
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
        f"asymmetry of "
        f"{facial_asymmetry_percentage:.2f}%. "

        f"The clinical risk model produced an estimated "
        f"risk probability of "
        f"{clinical_risk_percentage:.2f}%. "

        f"The overall system status was: "
        f"{overall_status}."
    )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    timestamp = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    # ========================================================
    # CREATE PDF DOCUMENT
    # ========================================================

    document = SimpleDocTemplate(

        filename,

        pagesize=A4,

        rightMargin=20 * mm,
        leftMargin=20 * mm,

        topMargin=20 * mm,
        bottomMargin=20 * mm
    )

    # ========================================================
    # STYLES
    # ========================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",

        parent=styles["Title"],

        alignment=TA_CENTER,

        fontSize=18,

        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",

        parent=styles["Normal"],

        alignment=TA_CENTER,

        fontSize=11,

        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "SectionHeading",

        parent=styles["Heading2"],

        fontSize=12,

        spaceBefore=12,

        spaceAfter=8
    )

    normal_style = ParagraphStyle(
        "NormalText",

        parent=styles["Normal"],

        fontSize=9,

        leading=14
    )

    small_style = ParagraphStyle(
        "SmallText",

        parent=styles["Normal"],

        fontSize=8,

        leading=11
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "NEURO.AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            "MULTIMODAL STROKE ANALYSIS REPORT",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"Report Date & Time: {timestamp}",
            normal_style
        )
    )

    story.append(
        Spacer(
            1,
            10
        )
    )

    # ========================================================
    # 1. PATIENT INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "1. Patient Information",
            heading_style
        )
    )

    patient_data = [

        [
            "Patient ID",
            str(patient_id)
        ],

        [
            "Patient Name",
            str(patient_name)
        ],

        [
            "Age",
            str(age)
        ],

        [
            "Gender",
            str(gender)
        ]
    ]

    patient_table = Table(
        patient_data,

        colWidths=[
            50 * mm,
            110 * mm
        ]
    )

    patient_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        patient_table
    )

    # ========================================================
    # 2. CT ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "2. CT Analysis",
            heading_style
        )
    )

    ct_data = [

        [
            "CT File",
            str(ct_file)
        ],

        [
            "CT Metadata",
            str(ct_metadata)
        ]
    ]

    ct_table = Table(
        ct_data,

        colWidths=[
            50 * mm,
            110 * mm
        ]
    )

    ct_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        ct_table
    )

    # ========================================================
    # 3. CT AI PREDICTION
    # ========================================================

    story.append(
        Paragraph(
            "3. CT AI Prediction",
            heading_style
        )
    )

    prediction_data = [

        [
            "Predicted Condition",
            str(prediction)
        ],

        [
            "Confidence",
            f"{confidence:.2f}%"
        ],

        [
            "Priority",
            priority
        ]
    ]

    prediction_table = Table(
        prediction_data,

        colWidths=[
            50 * mm,
            110 * mm
        ]
    )

    prediction_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        prediction_table
    )

    # ========================================================
    # 4. CLASS PROBABILITIES
    # ========================================================

    story.append(
        Paragraph(
            "4. CT Class Probabilities",
            heading_style
        )
    )

    probability_data = [

        [
            "Class",
            "Probability"
        ]
    ]

    for class_name, probability in (
        probabilities.items()
    ):

        probability_data.append(

            [
                str(class_name),

                f"{float(probability):.2f}%"
            ]
        )

    probability_table = Table(
        probability_data,

        colWidths=[
            100 * mm,
            60 * mm
        ]
    )

    probability_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        probability_table
    )

    # ========================================================
    # 5. FACIAL PALSY ASSESSMENT
    # ========================================================

    story.append(
        Paragraph(
            "5. Facial Palsy Assessment",
            heading_style
        )
    )

    facial_data = [

        [
            "Assessment",
            str(facial_assessment)
        ],

        [
            "Facial Asymmetry",
            f"{facial_asymmetry_percentage:.2f}%"
        ]
    ]

    facial_table = Table(
        facial_data,

        colWidths=[
            70 * mm,
            90 * mm
        ]
    )

    facial_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        facial_table
    )

    # ========================================================
    # 6. CLINICAL RISK
    # ========================================================

    story.append(
        Paragraph(
            "6. Clinical Risk Assessment",
            heading_style
        )
    )

    clinical_risk_data = [

        [
            "Clinical Risk Probability",
            f"{clinical_risk_percentage:.2f}%"
        ]
    ]

    clinical_risk_table = Table(
        clinical_risk_data,

        colWidths=[
            70 * mm,
            90 * mm
        ]
    )

    clinical_risk_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        clinical_risk_table
    )

    # ========================================================
    # 7. CLINICAL INPUTS
    # ========================================================

    story.append(
        Paragraph(
            "7. Clinical Inputs",
            heading_style
        )
    )

    if clinical_inputs:

        clinical_data = [

            [
                "Clinical Field",
                "Value"
            ]
        ]

        for field, value in (
            clinical_inputs.items()
        ):

            clinical_data.append(

                [
                    str(field),
                    str(value)
                ]
            )

        clinical_table = Table(
            clinical_data,

            colWidths=[
                70 * mm,
                90 * mm
            ]
        )

        clinical_table.setStyle(
            TableStyle([

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    9
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ])
        )

        story.append(
            clinical_table
        )

    else:

        story.append(
            Paragraph(
                "No clinical inputs available.",
                normal_style
            )
        )

    # ========================================================
    # 8. GRAD-CAM
    # ========================================================

    story.append(
        Paragraph(
            "8. Grad-CAM Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            str(gradcam),
            normal_style
        )
    )

    # ========================================================
    # 9. LESION ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "9. Lesion Analysis",
            heading_style
        )
    )

    story.append(
        Paragraph(
            str(lesion_analysis),
            normal_style
        )
    )

    # ========================================================
    # 10. MULTIMODAL FUSION
    # ========================================================

    story.append(
        Paragraph(
            "10. Multimodal Fusion Result",
            heading_style
        )
    )

    fusion_data = [

        [
            "Stroke Type / CT Result",
            str(stroke_type)
        ],

        [
            "Facial Assessment",
            str(facial_assessment)
        ],

        [
            "Clinical Risk",
            f"{clinical_risk_percentage:.2f}%"
        ],

        [
            "Overall Status",
            str(overall_status)
        ],

        [
            "Priority",
            priority
        ]
    ]

    fusion_table = Table(
        fusion_data,

        colWidths=[
            70 * mm,
            90 * mm
        ]
    )

    fusion_table.setStyle(
        TableStyle([

            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),

            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (0, -1),
                "Helvetica-Bold"
            ),

            (
                "FONTSIZE",
                (0, 0),
                (-1, -1),
                9
            ),

            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),

            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(
        fusion_table
    )

    # ========================================================
    # 11. AI SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "11. Neuro.ai Summary",
            heading_style
        )
    )

    story.append(
        Paragraph(
            ai_summary,
            normal_style
        )
    )

    # ========================================================
    # 12. DISCLAIMER
    # ========================================================

    story.append(
        Paragraph(
            "12. Limitations and Disclaimer",
            heading_style
        )
    )

    disclaimer = (
        "This report is generated using an AI-based decision-support "
        "prototype and is intended to assist clinical review. The CT "
        "prediction, facial assessment and clinical risk estimation "
        "should not be considered definitive medical diagnoses. "
        "The clinical risk probability is an estimate produced by "
        "the trained clinical model and should not be interpreted "
        "as a validated individual clinical risk score. Facial "
        "asymmetry measurements are based on the prototype live-camera "
        "assessment and are not standalone diagnostic criteria. "
        "Grad-CAM visualizations are interpretation aids and should "
        "not be considered definitive evidence of a lesion. Clinical "
        "decisions should be made by qualified healthcare professionals "
        "using appropriate clinical, imaging and diagnostic information."
    )

    story.append(
        Paragraph(
            disclaimer,
            small_style
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )

    return filename


# ============================================================
# SAMPLE TEST
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
                "Grad-CAM visualization is available. "
                "Highlighted regions represent areas that "
                "contributed to the model prediction."
            ),

        "lesion_analysis":
            (
                "Potential abnormal region identified by "
                "AI analysis. Further clinical evaluation "
                "is required."
            ),

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        "stroke_type":
            "Hemorrhagic Stroke",

        "overall_status":
            "Stroke pattern detected on CT"
    }

    # --------------------------------------------------------
    # Generate PDF
    # --------------------------------------------------------

    pdf_file = generate_pdf_report(
        sample_case,
        "neurovision_stroke_report.pdf"
    )

    print(
        "--------------------------------------------"
    )

    print(
        "PDF REPORT GENERATED SUCCESSFULLY!"
    )

    print(
        "--------------------------------------------"
    )

    print(
        f"File name: {pdf_file}"
    )

    print(
        "--------------------------------------------"
    )