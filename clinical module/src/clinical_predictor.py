# ============================================================
# Neuro.ai - Clinical Prediction Module
# ============================================================

import os
import joblib
import pandas as pd


# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "output",
    "clinical_random_forest.joblib"
)


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Clinical model not found at:\n{MODEL_PATH}"
    )


model = joblib.load(MODEL_PATH)

print("Clinical Random Forest model loaded successfully.")


# ============================================================
# 2. CLINICAL PREDICTION FUNCTION
# ============================================================

def predict_clinical_risk(patient_data):
    """
    Predict clinical stroke-risk indicator from patient data.

    Parameters
    ----------
    patient_data : dict
        Patient clinical information.

    Returns
    -------
    dict
        Prediction result.
    """

    # Convert dictionary to DataFrame
    input_data = pd.DataFrame(
        [patient_data]
    )

    # Make prediction
    prediction = model.predict(
        input_data
    )[0]

    # Get probability for stroke class
    probability = model.predict_proba(
        input_data
    )[0][1]

    # Convert to percentage
    risk_percentage = probability * 100


    # --------------------------------------------------------
    # Risk category
    # --------------------------------------------------------

    if risk_percentage < 20:

        risk_category = "Low"

    elif risk_percentage < 50:

        risk_category = "Moderate"

    else:

        risk_category = "High"


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {

        "prediction": int(prediction),

        "clinical_risk_probability": round(
            risk_percentage,
            2
        ),

        "risk_category": risk_category,

        "interpretation":
            "Clinical risk indicator based on the provided "
            "demographic, medical, and lifestyle information."
    }


    return result


# ============================================================
# 3. TEST THE MODULE
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("TESTING CLINICAL PREDICTION MODULE")
    print("=" * 60)


    # Example patient
    patient = {

        "gender": "Male",

        "age": 65,

        "hypertension": 1,

        "heart_disease": 0,

        "ever_married": "Yes",

        "work_type": "Private",

        "Residence_type": "Urban",

        "avg_glucose_level": 180.0,

        "bmi": 27.5,

        "smoking_status": "formerly smoked"
    }


    print("\nPatient information:")

    for key, value in patient.items():

        print(
            f"{key}: {value}"
        )


    # Run prediction
    result = predict_clinical_risk(
        patient
    )


    print("\n" + "=" * 60)
    print("CLINICAL PREDICTION")
    print("=" * 60)

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    print(
        f"Clinical Risk Probability: "
        f"{result['clinical_risk_probability']}%"
    )

    print(
        f"Risk Category: "
        f"{result['risk_category']}"
    )

    print(
        f"\nInterpretation:\n"
        f"{result['interpretation']}"
    )

    print("\n" + "=" * 60)
    print("PREDICTION COMPLETED")
    print("=" * 60)