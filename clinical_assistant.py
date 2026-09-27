# ============================================================
# NEURO.AI - CLINICAL ASSISTANT
# ============================================================


# ============================================================
# CT PREDICTION EXPLANATION
# ============================================================

def explain_prediction(
    prediction,
    confidence
):
    """
    Explains the CT prediction in simple clinical language.
    """

    prediction = str(
        prediction
    ).lower()

    if "ischemic" in prediction:

        explanation = (
            "The AI model predicted an ischemic stroke. "
            "An ischemic stroke occurs when blood flow to part "
            "of the brain is blocked. The CT model confidence "
            f"is {confidence:.2f}%. "
            "This result is intended to assist clinical review "
            "and does not replace professional assessment."
        )

    elif "hemorrhagic" in prediction:

        explanation = (
            "The AI model predicted a hemorrhagic stroke. "
            "A hemorrhagic stroke occurs when a blood vessel "
            "in the brain ruptures and causes bleeding. "
            f"The CT model confidence is {confidence:.2f}%. "
            "This result is intended to assist clinical review "
            "and does not replace professional assessment."
        )

    elif "normal" in prediction:

        explanation = (
            "The AI model predicted a normal CT pattern. "
            f"The model confidence is {confidence:.2f}%. "
            "A normal model prediction does not by itself "
            "exclude every possible neurological condition."
        )

    else:

        explanation = (
            f"The AI model predicted {prediction} "
            f"with a confidence of {confidence:.2f}%."
        )

    return explanation


# ============================================================
# FACIAL ASSESSMENT EXPLANATION
# ============================================================

def explain_facial_assessment(
    assessment,
    asymmetry
):
    """
    Explains the facial assessment.
    """

    assessment = str(
        assessment
    )

    asymmetry_percentage = float(asymmetry)

    return (
        f"The live facial assessment recorded "
        f"{assessment.lower()}. "
        f"The calculated facial asymmetry was "
        f"{asymmetry_percentage:.2f}%. "
        "This is a prototype assessment based on facial "
        "landmark measurements and should not be interpreted "
        "as a standalone clinical diagnosis."
    )


# ============================================================
# CLINICAL RISK EXPLANATION
# ============================================================

def explain_clinical_risk(
    probability
):
    """
    Explains the clinical risk model output.
    """

    probability_percentage = float(probability)

    return (
        f"The clinical risk model produced an estimated "
        f"risk probability of {probability_percentage:.2f}%. "
        "This model provides additional clinical context "
        "and should not be interpreted as a confirmed "
        "stroke diagnosis."
    )


# ============================================================
# FUSION EXPLANATION
# ============================================================

def explain_fusion(
    case
):
    """
    Explains the combined Neuro.ai assessment.
    """

    prediction = case.get(
        "prediction",
        "Not available"
    )

    confidence = case.get(
        "confidence",
        0
    )

    facial_assessment = case.get(
        "facial_assessment",
        "Not available"
    )

    facial_asymmetry = case.get(
        "facial_asymmetry",
        0
    )

    clinical_probability = case.get(
        "clinical_risk_probability",
        0
    )

    overall_status = case.get(
        "overall_status",
        "Not available"
    )

    return (
        "NEURO.AI MULTIMODAL ASSESSMENT\n\n"

        f"CT Assessment:\n"
        f"{prediction} "
        f"({float(confidence):.2f}% confidence)\n\n"

        f"Facial Assessment:\n"
        f"{facial_assessment} "
        f"({float(facial_asymmetry):.2f}% asymmetry)\n\n"

        f"Clinical Assessment:\n"
        f"{float(clinical_probability):.2f}% "
        f"clinical risk probability\n\n"

        f"Overall System Status:\n"
        f"{overall_status}\n\n"

        "The three components provide complementary "
        "information for clinical review. Neuro.ai is an "
        "AI-assisted decision-support prototype and does "
        "not replace assessment by a qualified healthcare "
        "professional."
    )


# ============================================================
# GRAD-CAM EXPLANATION
# ============================================================

def explain_gradcam():

    return (
        "Grad-CAM is a visualization technique used to show "
        "regions of an image that contributed to the AI model's "
        "prediction. Highlighted areas indicate regions that "
        "had greater influence on the model's decision. "
        "Grad-CAM is an interpretation aid and not a definitive "
        "indication of a lesion."
    )


# ============================================================
# STROKE TYPE EXPLANATION
# ============================================================

def explain_stroke_types():

    return (
        "There are two major stroke types supported by the CT "
        "classifier. Ischemic stroke occurs when blood flow to "
        "part of the brain is blocked, while hemorrhagic stroke "
        "occurs when a blood vessel ruptures and causes bleeding "
        "in or around the brain."
    )


# ============================================================
# CLINICAL FIELD EXPLANATION
# ============================================================

def explain_clinical_fields():

    return (
        "Clinical assessment fields provide additional patient "
        "context. These may include age, blood pressure, "
        "medical history, glucose level, heart disease, "
        "hypertension and other relevant observations. "
        "These factors provide context alongside the AI "
        "image and facial assessments."
    )


# ============================================================
# CASE SUMMARY
# ============================================================

def summarize_case(
    case
):
    """
    Creates a summary of the complete Neuro.ai case.
    """

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

    prediction = case.get(
        "prediction",
        "Not available"
    )

    confidence = case.get(
        "confidence",
        0
    )

    facial_assessment = case.get(
        "facial_assessment",
        "Not available"
    )

    facial_asymmetry = case.get(
        "facial_asymmetry",
        0
    )

    clinical_probability = case.get(
        "clinical_risk_probability",
        0
    )

    overall_status = case.get(
        "overall_status",
        "Not available"
    )

    summary = (

        "NEURO.AI CASE SUMMARY\n\n"

        f"Patient ID: {patient_id}\n"
        f"Patient Name: {patient_name}\n"
        f"Age: {age}\n"
        f"Gender: {gender}\n\n"

        f"CT Prediction: {prediction}\n"
        f"CT Confidence: {float(confidence):.2f}%\n\n"

        f"Facial Assessment: "
        f"{facial_assessment}\n"

        f"Facial Asymmetry: "
        f"{float(facial_asymmetry):.2f}%\n\n"

        f"Clinical Risk Probability: "
        f"{float(clinical_probability):.2f}%\n\n"

        f"Overall Status: "
        f"{overall_status}\n\n"

        "The results are intended to assist clinical review "
        "and should not replace assessment by a qualified "
        "healthcare professional."
    )

    return summary


# ============================================================
# ASSISTANT RESPONSE
# ============================================================

def assistant_response(
    question,
    case
):
    """
    Provides controlled responses to supported
    Neuro.ai clinical-assistant questions.
    """

    question = (
        str(question)
        .lower()
        .strip()
    )

    prediction = case.get(
        "prediction",
        "Not available"
    )

    confidence = case.get(
        "confidence",
        0
    )

    # --------------------------------------------------------
    # Multimodal / fusion questions
    # --------------------------------------------------------

    if (
        "fusion" in question
        or "combined" in question
        or "multimodal" in question
        or "overall" in question
    ):

        return explain_fusion(
            case
        )

    # --------------------------------------------------------
    # Facial questions
    # --------------------------------------------------------

    elif (
        "facial" in question
        or "palsy" in question
        or "asymmetry" in question
    ):

        return explain_facial_assessment(
            case.get(
                "facial_assessment",
                "Not available"
            ),

            case.get(
                "facial_asymmetry",
                0
            )
        )

    # --------------------------------------------------------
    # Clinical risk questions
    # --------------------------------------------------------

    elif (
        "risk" in question
        or "clinical probability" in question
    ):

        return explain_clinical_risk(
            case.get(
                "clinical_risk_probability",
                0
            )
        )

    # --------------------------------------------------------
    # CT prediction questions
    # --------------------------------------------------------

    elif (
        "prediction" in question
        or "result" in question
        or "ct" in question
    ):

        return explain_prediction(
            prediction,
            confidence
        )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    elif (
        "grad-cam" in question
        or "gradcam" in question
        or "heatmap" in question
    ):

        return explain_gradcam()

    # --------------------------------------------------------
    # Stroke types
    # --------------------------------------------------------

    elif (
        "ischemic" in question
        or "hemorrhagic" in question
        or "stroke type" in question
    ):

        return explain_stroke_types()

    # --------------------------------------------------------
    # Clinical fields
    # --------------------------------------------------------

    elif (
        "clinical" in question
        or "field" in question
        or "medical history" in question
    ):

        return explain_clinical_fields()

    # --------------------------------------------------------
    # Case summary
    # --------------------------------------------------------

    elif (
        "summary" in question
        or "case" in question
        or "patient" in question
    ):

        return summarize_case(
            case
        )

    # --------------------------------------------------------
    # Default response
    # --------------------------------------------------------

    else:

        return (
            "I can help explain the CT prediction, "
            "facial assessment, clinical risk, "
            "multimodal fusion result, Grad-CAM visualization, "
            "stroke types, clinical assessment fields, "
            "or summarize the current case."
        )


# ============================================================
# SAMPLE DATA FOR TESTING
# ============================================================

if __name__ == "__main__":

    sample_case = {

        "patient_id":
            "NV001",

        "patient_name":
            "Sample Patient",

        "age":
            62,

        "gender":
            "Male",

        "prediction":
            "Ischemic Stroke",

        "confidence":
            94.5,

        "facial_assessment":
            "Moderate asymmetry",

        "facial_asymmetry":
            14.0,

        "clinical_risk_probability":
            61.8,

        "overall_status":
            "Stroke pattern detected on CT"
    }

    print(
        "=" * 65
    )

    print(
        "NEURO.AI - CLINICAL ASSISTANT"
    )

    print(
        "=" * 65
    )

    # --------------------------------------------------------
    # Test 1
    # --------------------------------------------------------

    print(
        "\nQuestion: What does the prediction mean?"
    )

    print(
        assistant_response(
            "What does the prediction mean?",
            sample_case
        )
    )

    # --------------------------------------------------------
    # Test 2
    # --------------------------------------------------------

    print(
        "\nQuestion: What is Grad-CAM?"
    )

    print(
        assistant_response(
            "What is Grad-CAM?",
            sample_case
        )
    )

    # --------------------------------------------------------
    # Test 3
    # --------------------------------------------------------

    print(
        "\nQuestion: What is the facial assessment?"
    )

    print(
        assistant_response(
            "What is the facial assessment?",
            sample_case
        )
    )

    # --------------------------------------------------------
    # Test 4
    # --------------------------------------------------------

    print(
        "\nQuestion: What is the clinical risk?"
    )

    print(
        assistant_response(
            "What is the clinical risk?",
            sample_case
        )
    )

    # --------------------------------------------------------
    # Test 5
    # --------------------------------------------------------

    print(
        "\nQuestion: Give me the multimodal result."
    )

    print(
        assistant_response(
            "Give me the multimodal result.",
            sample_case
        )
    )

    # --------------------------------------------------------
    # Test 6
    # --------------------------------------------------------

    print(
        "\nQuestion: Give me a case summary."
    )

    print(
        assistant_response(
            "Give me a case summary.",
            sample_case
        )
    )