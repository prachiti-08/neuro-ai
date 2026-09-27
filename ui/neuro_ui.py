import sys
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd
from PIL import Image


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Neuro.ai",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b
    );

    color: #e2e8f0;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        sans-serif;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.30);
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 20px;
}

.metric-card {
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.07);
}

section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.97);
}

.stButton button {
    border-radius: 12px;
    background: linear-gradient(
        135deg,
        #3b82f6,
        #6366f1
    );
    color: white;
    border: none;
    font-weight: 600;
    min-height: 42px;
}

.stButton button:hover {
    border: 1px solid #60a5fa;
}

h1, h2, h3 {
    font-weight: 600;
}

hr {
    border-color: rgba(255,255,255,0.12);
}

.footer {
    text-align: center;
    font-size: 12px;
    color: #94a3b8;
    margin-top: 45px;
}

.small-text {
    color: #94a3b8;
    font-size: 14px;
}

.result-normal {
    color: #34d399;
    font-weight: 700;
}

.result-warning {
    color: #fbbf24;
    font-weight: 700;
}

.result-danger {
    color: #f87171;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "patient_data": {},
    "ct_result": None,
    "ct_image": None,
    "ct_filename": None,
    "facial_result": None,
    "clinical_result": None,
    "fusion_result": None,
    "complete_result": None,
    "case_id": None,
    "pdf_path": None,
    "case_saved": False,
    "assistant_case": None
}

for key, value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# BACKEND IMPORTS
# ============================================================

try:

    from neuro_ai import (
        predict_ct,
        predict_facial,
        predict_clinical,
        fuse_multimodal_results,
        fuse_ct_and_clinical
    )

    from case_database import (
        initialize_database,
        add_case,
        get_all_cases
    )

    from clinical_assistant import (
        assistant_response
    )

    from pdf_generator import (
        generate_pdf_report
    )

    BACKEND_READY = True

except Exception as e:

    BACKEND_READY = False
    BACKEND_ERROR = str(e)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

if BACKEND_READY:

    try:
        initialize_database()
    except Exception:
        pass


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            text-align:center;
            padding:10px 5px 5px 5px;
        ">

        <h1 style="
            color:#60a5fa;
            margin-bottom:0;
        ">
        Neuro.ai
        </h1>

        <p style="
            color:#94a3b8;
            margin-top:5px;
        ">
        Multimodal Stroke Assessment
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Navigation",
        [
            "Home",
            "Patient Information",
            "CT Analysis",
            "Facial Assessment",
            "Clinical Risk",
            "Multimodal Analysis",
            "Case History",
            "AI Assistant"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    if st.session_state.patient_data:

        st.markdown("### Current Case")

        st.caption(
            f"Patient: "
            f"{st.session_state.patient_data.get('patient_name', 'N/A')}"
        )

        st.caption(
            f"ID: "
            f"{st.session_state.patient_data.get('patient_id', 'N/A')}"
        )

    else:

        st.caption("No patient loaded.")

    st.divider()

    st.markdown(
        """
        <div style="
            color:#64748b;
            font-size:12px;
        ">

        <b>Neuro.ai</b><br>
        Explainable Multimodal AI<br>
        Stroke Assessment<br><br>

        Research Prototype

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# BACKEND ERROR
# ============================================================

if not BACKEND_READY:

    st.error(
        "Neuro.ai backend could not be imported."
    )

    st.code(
        BACKEND_ERROR
    )

    st.stop()


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="card">

        <h1 style="
            font-size:44px;
            color:#f1f5f9;
            margin-bottom:5px;
        ">
        Neuro.ai
        </h1>

        <p style="
            font-size:21px;
            color:#94a3b8;
        ">
        Explainable Multimodal AI for Stroke Assessment
        </p>

        <p style="
            font-size:15px;
            color:#cbd5e1;
            line-height:1.8;
        ">
        Neuro.ai integrates brain CT analysis, facial motor
        assessment, and clinical indicators into a unified
        AI-assisted stroke assessment workflow.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("## Assessment Modules")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="card">

            <h2>CT Analysis</h2>

            <p class="small-text">
            ResNet18-based classification of brain CT scans
            into Normal, Ischemic Stroke, or Hemorrhagic Stroke.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">

            <h2>Facial Assessment</h2>
            <p class="small-text">
            Real-time facial motor assessment using MediaPipe
            facial landmarks and left-right asymmetry.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="card">

            <h2>Clinical Risk</h2>

            <p class="small-text">
            Clinical risk estimation using demographic,
            medical, and lifestyle factors.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("## Neuro.ai Workflow")

    st.markdown(
        """
        <div class="card">

        <p style="
            text-align:center;
            font-size:17px;
            color:#cbd5e1;
        ">

        Patient
        &nbsp; → &nbsp;
        CT
        &nbsp; + &nbsp;
        Facial
        &nbsp; + &nbsp;
        Clinical
        &nbsp; → &nbsp;
        Fusion
        &nbsp; → &nbsp;
        Report

        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.ct_result:

        st.markdown("## Current Assessment")

        ct = st.session_state.ct_result

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "CT Prediction",
                ct["prediction"]
            )

        with col2:
            st.metric(
                "CT Confidence",
                f'{ct["confidence"]:.2f}%'
            )

        with col3:

            if st.session_state.fusion_result:

                st.metric(
                    "Overall Status",
                    st.session_state.fusion_result[
                        "overall_status"
                    ]
                )

    st.info(
        "Neuro.ai is a research and decision-support "
        "prototype and is not intended to replace "
        "professional medical diagnosis."
    )


# ============================================================
# PATIENT INFORMATION
# ============================================================

elif page == "Patient Information":

    st.markdown("## Patient Information")

    st.markdown(
        """
        <div class="card">

        <p class="small-text">
        Enter the patient information used by the clinical
        risk module and case management system.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    existing = st.session_state.patient_data

    col1, col2 = st.columns(2)

    with col1:

        patient_id = st.text_input(
            "Patient ID",
            value=existing.get(
                "patient_id",
                ""
            )
        )

        patient_name = st.text_input(
            "Patient Name",
            value=existing.get(
                "patient_name",
                ""
            )
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=int(
                existing.get(
                    "age",
                    50
                )
            )
        )

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ],
            index=[
                "Male",
                "Female",
                "Other"
            ].index(
                existing.get(
                    "gender",
                    "Male"
                )
            )
        )

        hypertension = st.selectbox(
            "Hypertension",
            [0, 1],
            index=int(
                existing.get(
                    "hypertension",
                    0
                )
            ),
            format_func=lambda x:
                "Yes" if x else "No"
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            [0, 1],
            index=int(
                existing.get(
                    "heart_disease",
                    0
                )
            ),
            format_func=lambda x:
                "Yes" if x else "No"
        )

    with col2:

        ever_married = st.selectbox(
            "Ever Married",
            [
                "Yes",
                "No"
            ],
            index=[
                "Yes",
                "No"
            ].index(
                existing.get(
                    "ever_married",
                    "Yes"
                )
            )
        )

        work_type = st.selectbox(
            "Work Type",
            [
                "Private",
                "Self-employed",
                "Govt_job",
                "children",
                "Never_worked"
            ],
            index=[
                "Private",
                "Self-employed",
                "Govt_job",
                "children",
                "Never_worked"
            ].index(
                existing.get(
                    "work_type",
                    "Private"
                )
            )
        )

        residence_type = st.selectbox(
            "Residence Type",
            [
                "Urban",
                "Rural"
            ],
            index=[
                "Urban",
                "Rural"
            ].index(
                existing.get(
                    "Residence_type",
                    "Urban"
                )
            )
        )

        avg_glucose_level = st.number_input(
            "Average Glucose Level",
            min_value=0.0,
            max_value=500.0,
            value=float(
                existing.get(
                    "avg_glucose_level",
                    100.0
                )
            ),
            step=0.1
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            max_value=100.0,
            value=float(
                existing.get(
                    "bmi",
                    25.0
                )
            ),
            step=0.1
        )

        smoking_status = st.selectbox(
            "Smoking Status",
            [
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown"
            ],
            index=[
                "never smoked",
                "formerly smoked",
                "smokes",
                "Unknown"
            ].index(
                existing.get(
                    "smoking_status",
                    "never smoked"
                )
            )
        )

    st.markdown("")

    if st.button(
        "Save Patient Information",
        type="primary",
        use_container_width=True
    ):

        if not patient_id.strip():

            st.warning(
                "Please enter a Patient ID."
            )

        elif not patient_name.strip():

            st.warning(
                "Please enter the Patient Name."
            )

        else:

            st.session_state.patient_data = {

                "patient_id":
                    patient_id.strip(),

                "patient_name":
                    patient_name.strip(),

                "age":
                    int(age),

                "gender":
                    gender,

                "hypertension":
                    int(hypertension),

                "heart_disease":
                    int(heart_disease),

                "ever_married":
                    ever_married,

                "work_type":
                    work_type,

                "Residence_type":
                    residence_type,

                "avg_glucose_level":
                    float(avg_glucose_level),

                "bmi":
                    float(bmi),

                "smoking_status":
                    smoking_status
            }

            st.success(
                "Patient information saved successfully."
            )

            st.session_state.case_saved = False
            st.session_state.case_id = None


# ============================================================
# CT ANALYSIS
# ============================================================

elif page == "CT Analysis":

    st.markdown("## Brain CT Analysis")

    st.markdown(
        """
        <div class="card">

        <h3>Brain CT Classification</h3>

        <p class="small-text">
        Upload a brain CT image to classify it as Normal,
        Ischemic Stroke, or Hemorrhagic Stroke and generate
        a Grad-CAM visualization.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CT Scan",
        type=[
            "png",
            "jpg",
            "jpeg"
        ],
        key="ct_upload"
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.session_state.ct_image = image
        st.session_state.ct_filename = uploaded_file.name

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader("CT Scan")

            st.image(
                image,
                width="stretch"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader("Analysis")

            if st.button(
                "🔍 Analyze CT Scan",
                type="primary",
                use_container_width=True
            ):

                with st.spinner(
                    "Analyzing CT scan..."
                ):

                    try:

                        result = predict_ct(
                            image
                        )

                        st.session_state.ct_result = result

                        st.success(
                            "CT analysis completed."
                        )

                    except Exception as e:

                        st.error(
                            f"CT analysis failed: {e}"
                        )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    result = st.session_state.ct_result

    if result is not None:

        st.markdown("---")

        st.markdown("## CT Analysis Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Prediction",
                result["prediction"]
            )

        with col2:

            st.metric(
                "Confidence",
                f'{result["confidence"]:.2f}%'
            )

        with col3:

            raw = result["raw_prediction"]

            if raw == "normal":

                st.markdown(
                    '<p class="result-normal">'
                    '✓ Normal classification'
                    '</p>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<p class="result-warning">'
                    '⚠ Possible abnormality'
                    '</p>',
                    unsafe_allow_html=True
                )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("Class Probabilities")

        probabilities = result[
            "probabilities"
        ]

        pcols = st.columns(
            len(probabilities)
        )

        for i, (
            class_name,
            probability
        ) in enumerate(
            probabilities.items()
        ):

            with pcols[i]:

                st.metric(
                    class_name,
                    f"{probability:.2f}%"
                )

        probability_df = pd.DataFrame(
            {
                "Class":
                    list(
                        probabilities.keys()
                    ),
                "Probability":
                    list(
                        probabilities.values()
                    )
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Class"
            )
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # Grad-CAM
        # ----------------------------------------------------

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Grad-CAM Explainability"
        )

        st.caption(
            "The visualization highlights regions contributing "
            "to the model prediction."
        )

        if result.get(
            "gradcam_overlay"
        ) is not None:

            st.image(
                result["gradcam_overlay"],
                width="stretch"
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# FACIAL ASSESSMENT
# ============================================================

elif page == "Facial Assessment":

    st.markdown("## Facial Motor Assessment")

    st.markdown(
        """
        <div class="card">

        <h3>Live Facial Motor Assessment</h3>

        <p class="small-text">
        The existing MediaPipe-based module guides the patient
        through facial movements and calculates left-right
        facial asymmetry.
        </p>

        <p class="small-text">
        Required movements include neutral face, smile,
        eyebrow raise, eye closure, and mouth opening.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.warning(
        "The facial assessment uses the existing local webcam "
        "workflow. Make sure your webcam is available."
    )

    if st.button(
        "Start Facial Assessment",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Starting facial assessment..."
        ):

            try:

                facial_result = predict_facial()

                if facial_result is not None:

                    st.session_state.facial_result = (
                        facial_result
                    )

                    st.success(
                        "Facial assessment completed."
                    )

                else:

                    st.warning(
                        "Facial assessment was cancelled."
                    )

            except Exception as e:

                st.error(
                    f"Facial assessment failed: {e}"
                )

    # ========================================================
    # DISPLAY FACIAL RESULT
    # ========================================================

    facial = st.session_state.facial_result

    if facial is not None:

        st.markdown("---")

        st.markdown(
            "## Facial Assessment Result"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Assessment",
                facial[
                    "facial_assessment"
                ]
            )

        with col2:

            st.metric(
                "Asymmetry",
                f'{facial["asymmetry_percentage"]:.2f}%'
            )

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader(
            "Step-wise Assessment"
        )

        step_results = facial.get(
            "step_results",
            {}
        )

        if step_results:

            rows = []

            for step, data in step_results.items():

                if isinstance(
                    data,
                    dict
                ):

                    rows.append(
                        {
                            "Movement":
                                step,
                            "Asymmetry (%)":
                                round(
                                    float(
                                        data.get(
                                            "asymmetry",
                                            0
                                        )
                                    ) * 100,
                                    2
                                )
                        }
                    )

            if rows:

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                    hide_index=True
                )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.info(
            "Facial asymmetry thresholds in the current prototype "
            "are not clinically validated."
        )


# ============================================================
# CLINICAL RISK
# ============================================================

elif page == "Clinical Risk":

    st.markdown("## Clinical Risk Assessment")

    if not st.session_state.patient_data:

        st.warning(
            "Please complete Patient Information first."
        )

    else:

        patient = st.session_state.patient_data

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.write(
            f"**Patient:** "
            f"{patient.get('patient_name', 'N/A')}"
        )

        st.write(
            f"**Patient ID:** "
            f"{patient.get('patient_id', 'N/A')}"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Calculate Clinical Risk",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Calculating clinical risk..."
            ):

                try:

                    result = predict_clinical(
                        patient
                    )

                    st.session_state.clinical_result = (
                        result
                    )

                    st.success(
                        "Clinical risk calculated."
                    )

                except Exception as e:

                    st.error(
                        f"Clinical analysis failed: {e}"
                    )

        clinical = (
            st.session_state.clinical_result
        )

        if clinical is not None:

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Risk Probability",
                    f'{clinical["clinical_risk_probability"]:.2f}%'
                )

            with col2:

                st.metric(
                    "Risk Category",
                    clinical["risk_category"]
                )

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader(
                "Interpretation"
            )

            st.write(
                clinical[
                    "interpretation"
                ]
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


# ============================================================
# MULTIMODAL ANALYSIS
# ============================================================

elif page == "Multimodal Analysis":

    st.markdown("## Multimodal Analysis")

    st.markdown(
        """
        <div class="card">

        <h3>Neuro.ai Unified Assessment</h3>

        <p class="small-text">
        This section combines CT classification, facial motor
        assessment, and clinical risk information through the
        existing rule-based fusion layer.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    patient_ready = bool(
        st.session_state.patient_data
    )

    ct_ready = (
        st.session_state.ct_result
        is not None
    )

    facial_ready = (
        st.session_state.facial_result
        is not None
    )

    clinical_ready = (
        st.session_state.clinical_result
        is not None
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "CT",
            "Ready" if ct_ready else "Pending"
        )

    with col2:

        st.metric(
            "Facial",
            "Ready" if facial_ready else "Pending"
        )

    with col3:

        st.metric(
            "Clinical",
            "Ready" if clinical_ready else "Pending"
        )

    if not patient_ready:

        st.warning(
            "Complete Patient Information first."
        )

    if not ct_ready:

        st.warning(
            "Complete CT Analysis first."
        )

    if not clinical_ready:

        st.warning(
            "Complete Clinical Risk analysis first."
        )

    if st.button(
        "Generate Multimodal Assessment",
        type="primary",
        use_container_width=True,
        disabled=not (
            patient_ready
            and ct_ready
            and clinical_ready
        )
    ):

        try:

            ct_result = (
                st.session_state.ct_result
            )

            clinical_result = (
                st.session_state.clinical_result
            )

            facial_result = (
                st.session_state.facial_result
            )

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

            st.session_state.fusion_result = (
                fusion_result
            )

            st.success(
                "Multimodal assessment generated."
            )

        except Exception as e:

            st.error(
                f"Fusion failed: {e}"
            )

    fusion = (
        st.session_state.fusion_result
    )

    if fusion is not None:

        st.markdown("---")

        st.markdown(
            "## Unified Assessment"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader("CT")

            st.write(
                f"**Prediction:** "
                f"{fusion['ct_prediction']}"
            )

            st.write(
                f"**Confidence:** "
                f"{fusion['ct_confidence']:.2f}%"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                '<div class="card">',
                unsafe_allow_html=True
            )

            st.subheader("Clinical")

            st.write(
                f"**Risk:** "
                f"{fusion['clinical_risk_category']}"
            )

            st.write(
                f"**Probability:** "
                f"{fusion['clinical_risk_probability']:.2f}%"
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        if (
            "facial_assessment"
            in fusion
        ):

            col3, col4 = st.columns(2)

            with col3:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.subheader("Facial")

                st.write(
                    f"**Assessment:** "
                    f"{fusion['facial_assessment']}"
                )

                st.write(
                    f"**Asymmetry:** "
                    f"{fusion['facial_asymmetry']:.2f}%"
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

            with col4:

                st.markdown(
                    '<div class="card">',
                    unsafe_allow_html=True
                )

                st.subheader("Overall Status")

                st.write(
                    fusion[
                        "overall_status"
                    ]
                )

                st.markdown(
                    '</div>',
                    unsafe_allow_html=True
                )

        # ====================================================
        # SAVE CASE + GENERATE PDF REPORT
        # ====================================================

        st.markdown("---")

        st.markdown(
            "## Case Report"
        )

        st.markdown(
            """
            <div class="card">

            <h3>Save Assessment & Generate Report</h3>

            <p class="small-text">
            Save the complete patient assessment to the Neuro.ai
            case database and generate a structured PDF report.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "Save Case & Generate PDF Report",
            type="primary",
            use_container_width=True
        ):

            try:

                # ------------------------------------------------
                # Collect current results
                # ------------------------------------------------

                patient = (
                    st.session_state.patient_data
                )

                ct = (
                    st.session_state.ct_result
                )

                facial = (
                    st.session_state.facial_result
                )

                clinical = (
                    st.session_state.clinical_result
                )

                fusion = (
                    st.session_state.fusion_result
                )

                # ------------------------------------------------
                # Build complete case dictionary
                # ------------------------------------------------

                case = {

                    # Patient information
                    "patient_id":
                        patient.get(
                            "patient_id",
                            ""
                        ),

                    "patient_name":
                        patient.get(
                            "patient_name",
                            ""
                        ),

                    "age":
                        patient.get(
                            "age",
                            ""
                        ),

                    "gender":
                        patient.get(
                            "gender",
                            ""
                        ),

                    # CT information
                    "ct_file":
                        st.session_state.ct_filename
                        or "Uploaded CT Scan",

                    "ct_metadata":
                        "Brain CT image analyzed using "
                        "the Neuro.ai ResNet18 model.",

                    "prediction":
                        ct.get(
                            "prediction",
                            ""
                        ),

                    "confidence":
                        ct.get(
                            "confidence",
                            0
                        ),

                    "probabilities":
                        ct.get(
                            "probabilities",
                            {}
                        ),

                    # Facial assessment
                    "facial_assessment":
                        (
                            facial.get(
                                "facial_assessment",
                                "Not performed"
                            )
                            if facial
                            else "Not performed"
                        ),

                    "facial_asymmetry":
                        (
                            facial.get(
                                "asymmetry_percentage",
                                0
                            )
                            if facial
                            else 0
                        ),

                    # Clinical assessment
                    "clinical_risk_probability":
                        clinical.get(
                            "clinical_risk_probability",
                            0
                        ),

                    "clinical_inputs":
                        patient,

                    # Explainability
                    "gradcam":
                        (
                            "Grad-CAM visualization generated "
                            "for the CT prediction."
                            if ct.get(
                                "gradcam_overlay"
                            ) is not None
                            else
                            "Grad-CAM visualization not available."
                        ),

                    "lesion_analysis":
                        (
                            "Grad-CAM highlights image regions "
                            "that contributed to the model "
                            "prediction. It does not constitute "
                            "definitive lesion localization."
                        ),

                    # Fusion
                    "overall_status":
                        fusion.get(
                            "overall_status",
                            ""
                        ),

                    # Report placeholder
                    "report_file":
                        ""
                }

                # ------------------------------------------------
                # Save case to SQLite
                # ------------------------------------------------

                initialize_database()

                case_id = add_case(
                    case
                )

                st.session_state.case_id = (
                    case_id
                )

                st.session_state.case_saved = True

                # ------------------------------------------------
                # Generate PDF
                # ------------------------------------------------

                reports_dir = (
                    BASE_DIR / "reports"
                )

                reports_dir.mkdir(
                    exist_ok=True
                )

                pdf_path = (
                    reports_dir /
                    f"NeuroAI_Case_{case_id}.pdf"
                )

                generate_pdf_report(
                    case,
                    str(pdf_path)
                )

                st.session_state.pdf_path = (
                    str(pdf_path)
                )

                # ------------------------------------------------
                # Update report filename in database
                # ------------------------------------------------

                case["report_file"] = (
                    pdf_path.name
                )

                st.success(
                    f"Case saved successfully! "
                    f"Case ID: {case_id}"
                )

                st.success(
                    "PDF report generated successfully."
                )

            except Exception as e:

                st.error(
                    f"Could not save case or generate "
                    f"PDF report: {e}"
                )

        # ====================================================
        # SHOW SAVED CASE + DOWNLOAD BUTTON
        # ====================================================

        if (
            st.session_state.case_saved
            and st.session_state.pdf_path
        ):

            pdf_path = Path(
                st.session_state.pdf_path
            )

            if pdf_path.exists():

                st.markdown("---")

                st.markdown(
                    "### Report Ready"
                )

                st.info(
                    f"Case ID: "
                    f"{st.session_state.case_id}"
                )

                with open(
                    pdf_path,
                    "rb"
                ) as pdf_file:

                    st.download_button(
                        label=(
                            "Download PDF Report"
                        ),
                        data=pdf_file.read(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                        use_container_width=True
                    )


# ============================================================
# CASE HISTORY
# ============================================================

elif page == "Case History":

    st.markdown("## Case History")

    try:

        cases = get_all_cases()

        if cases:

            columns = [
                "Case ID",
                "Patient ID",
                "Patient Name",
                "Age",
                "Gender",
                "CT Prediction",
                "CT Confidence",
                "Facial Assessment",
                "Facial Asymmetry",
                "Clinical Risk",
                "Overall Status",
                "Priority",
                "Timestamp"
            ]

            df = pd.DataFrame(
                cases,
                columns=columns
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No cases have been saved yet."
            )

    except Exception as e:

        st.error(
            f"Could not load case history: {e}"
        )


# ============================================================
# AI ASSISTANT
# ============================================================

elif page == "AI Assistant":

    st.markdown(
        "## Neuro.ai Clinical Assistant"
    )

    st.markdown(
        """
        <div class="card">

        <h3>Ask about the current case</h3>

        <p class="small-text">
        The assistant can explain the CT prediction, facial
        assessment, clinical risk, fusion result, and Grad-CAM.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Build current case
    # --------------------------------------------------------

    ct = st.session_state.ct_result
    facial = st.session_state.facial_result
    clinical = st.session_state.clinical_result
    fusion = st.session_state.fusion_result
    patient = st.session_state.patient_data

    if ct is None:

        st.info(
            "Complete at least the CT analysis before "
            "using the assistant."
        )

    else:

        assistant_case = {

            "patient_id":
                patient.get(
                    "patient_id",
                    ""
                ),

            "patient_name":
                patient.get(
                    "patient_name",
                    ""
                ),

            "age":
                patient.get(
                    "age",
                    ""
                ),

            "gender":
                patient.get(
                    "gender",
                    ""
                ),

            "prediction":
                ct.get(
                    "prediction",
                    ""
                ),

            "confidence":
                ct.get(
                    "confidence",
                    0
                ),

            "probabilities":
                ct.get(
                    "probabilities",
                    {}
                ),

            "gradcam":
                "Grad-CAM visualization available.",

            "lesion_analysis":
                (
                    "Grad-CAM highlights regions that "
                    "contributed to the model prediction. "
                    "This does not constitute definitive "
                    "lesion localization."
                )
        }

        if facial:

            assistant_case.update({

                "facial_assessment":
                    facial.get(
                        "facial_assessment",
                        ""
                    ),

                "facial_asymmetry":
                    facial.get(
                        "asymmetry_percentage",
                        0
                    )
            })

        if clinical:

            assistant_case.update({

                "clinical_risk_probability":
                    clinical.get(
                        "clinical_risk_probability",
                        0
                    ),

                "clinical_inputs":
                    patient
            })

        if fusion:

            assistant_case.update({

                "overall_status":
                    fusion.get(
                        "overall_status",
                        ""
                    )
            })

        question = st.text_input(
            "Ask a question",
            placeholder=(
                "What does the CT result mean?"
            )
        )

        if st.button(
            "Ask Neuro.ai",
            type="primary"
        ):

            if question.strip():

                try:

                    response = assistant_response(
                        question,
                        assistant_case
                    )

                    st.markdown(
                        '<div class="card">',
                        unsafe_allow_html=True
                    )

                    st.subheader(
                        "Neuro.ai"
                    )

                    st.write(
                        response
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        f"Assistant error: {e}"
                    )

            else:

                st.warning(
                    "Please enter a question."
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <hr>

    Neuro.ai | Explainable Multimodal Stroke Assessment<br>
    For educational and research purposes only

    </div>
    """,
    unsafe_allow_html=True
)