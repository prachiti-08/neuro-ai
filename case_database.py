# ============================================================
# NEURO.AI - CASE DATABASE
# ============================================================

import sqlite3
from datetime import datetime
from pathlib import Path


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = (
    BASE_DIR / "neurovision_cases.db"
)


# ============================================================
# CREATE DATABASE AND TABLE
# ============================================================

def create_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cases (

            case_id INTEGER PRIMARY KEY AUTOINCREMENT,

            patient_id TEXT NOT NULL,
            patient_name TEXT,

            age INTEGER,
            gender TEXT,

            -- ------------------------------------------------
            -- CT INFORMATION
            -- ------------------------------------------------

            ct_file TEXT,
            ct_metadata TEXT,

            prediction TEXT,
            confidence REAL,

            ischemic_probability REAL,
            hemorrhagic_probability REAL,
            normal_probability REAL,

            -- ------------------------------------------------
            -- FACIAL PALSY INFORMATION
            -- ------------------------------------------------

            facial_assessment TEXT,
            facial_asymmetry REAL,

            -- ------------------------------------------------
            -- CLINICAL INFORMATION
            -- ------------------------------------------------

            clinical_risk_probability REAL,
            clinical_inputs TEXT,

            -- ------------------------------------------------
            -- EXPLAINABILITY
            -- ------------------------------------------------

            gradcam_info TEXT,
            lesion_analysis TEXT,

            -- ------------------------------------------------
            -- FINAL FUSION
            -- ------------------------------------------------

            overall_status TEXT,

            -- ------------------------------------------------
            -- REPORT / CASE INFORMATION
            -- ------------------------------------------------

            priority TEXT,
            report_file TEXT,

            timestamp TEXT
        )
    """)

    connection.commit()

    connection.close()

    print(
        "Database created successfully."
    )


# ============================================================
# DATABASE MIGRATION
# ============================================================

def migrate_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    # --------------------------------------------------------
    # Get existing columns
    # --------------------------------------------------------

    cursor.execute(
        "PRAGMA table_info(cases)"
    )

    existing_columns = {
        row[1]
        for row in cursor.fetchall()
    }

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    new_columns = {

        "facial_assessment":
            "TEXT",

        "facial_asymmetry":
            "REAL",

        "clinical_risk_probability":
            "REAL",

        "overall_status":
            "TEXT"
    }

    # --------------------------------------------------------
    # Add missing columns
    # --------------------------------------------------------

    for column_name, column_type in (
        new_columns.items()
    ):

        if column_name not in existing_columns:

            cursor.execute(
                f"""
                ALTER TABLE cases
                ADD COLUMN {column_name}
                {column_type}
                """
            )

            print(
                f"Added database column: "
                f"{column_name}"
            )

    connection.commit()

    connection.close()


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def initialize_database():

    create_database()

    migrate_database()


# ============================================================
# ADD NEW CASE
# ============================================================

def add_case(case):

    # --------------------------------------------------------
    # Connect to database
    # --------------------------------------------------------

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()


    # --------------------------------------------------------
    # CT confidence
    # --------------------------------------------------------

    confidence = float(
        case.get(
            "confidence",
            0
        )
    )


    # --------------------------------------------------------
    # Calculate application priority
    #
    # These are prototype application-level
    # priority thresholds, NOT clinical triage rules.
    # --------------------------------------------------------

    if confidence >= 90:

        priority = "High"

    elif confidence >= 70:

        priority = "Medium"

    else:

        priority = "Low"


    # --------------------------------------------------------
    # CT probabilities
    # --------------------------------------------------------

    probabilities = case.get(
        "probabilities",
        {}
    )


    ischemic_probability = probabilities.get(
        "Ischemic Stroke",
        probabilities.get(
            "ischemic",
            0
        )
    )


    hemorrhagic_probability = probabilities.get(
        "Hemorrhagic Stroke",
        probabilities.get(
            "hemorrhagic",
            0
        )
    )


    normal_probability = probabilities.get(
        "Normal",
        probabilities.get(
            "normal",
            0
        )
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )


    # --------------------------------------------------------
    # Insert complete case
    # --------------------------------------------------------

    cursor.execute(
        """
        INSERT INTO cases (

            patient_id,
            patient_name,

            age,
            gender,

            ct_file,
            ct_metadata,

            prediction,
            confidence,

            ischemic_probability,
            hemorrhagic_probability,
            normal_probability,

            facial_assessment,
            facial_asymmetry,

            clinical_risk_probability,
            clinical_inputs,

            gradcam_info,
            lesion_analysis,

            overall_status,

            priority,
            report_file,

            timestamp
        )

        VALUES (

            ?, ?, ?, ?,

            ?, ?,

            ?, ?,

            ?, ?, ?,

            ?, ?,

            ?, ?,

            ?, ?,

            ?,

            ?, ?,

            ?
        )
        """,
        (

            # ------------------------------------------------
            # Patient
            # ------------------------------------------------

            case.get(
                "patient_id"
            ),

            case.get(
                "patient_name"
            ),

            case.get(
                "age"
            ),

            case.get(
                "gender"
            ),


            # ------------------------------------------------
            # CT
            # ------------------------------------------------

            case.get(
                "ct_file"
            ),

            case.get(
                "ct_metadata"
            ),

            case.get(
                "prediction"
            ),

            confidence,

            ischemic_probability,

            hemorrhagic_probability,

            normal_probability,


            # ------------------------------------------------
            # Facial
            # ------------------------------------------------

            case.get(
                "facial_assessment"
            ),

            case.get(
                "facial_asymmetry",
                0
            ),


            # ------------------------------------------------
            # Clinical
            # ------------------------------------------------

            case.get(
                "clinical_risk_probability",
                0
            ),

            str(
                case.get(
                    "clinical_inputs",
                    {}
                )
            ),


            # ------------------------------------------------
            # Explainability
            # ------------------------------------------------

            str(
                case.get(
                    "gradcam",
                    ""
                )
            ),

            case.get(
                "lesion_analysis",
                ""
            ),


            # ------------------------------------------------
            # Fusion
            # ------------------------------------------------

            case.get(
                "overall_status",
                ""
            ),


            # ------------------------------------------------
            # Report
            # ------------------------------------------------

            priority,

            case.get(
                "report_file",
                ""
            ),


            # ------------------------------------------------
            # Timestamp
            # ------------------------------------------------

            timestamp
        )
    )


    # --------------------------------------------------------
    # Commit
    # --------------------------------------------------------

    connection.commit()


    # --------------------------------------------------------
    # Get SQLite-generated Case ID
    # --------------------------------------------------------

    case_id = cursor.lastrowid


    # --------------------------------------------------------
    # Close connection
    # --------------------------------------------------------

    connection.close()


    print(
        "Case added successfully."
    )


    return case_id


# ============================================================
# GET ALL CASES
# ============================================================

def get_all_cases():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        SELECT

            case_id,
            patient_id,
            patient_name,

            age,
            gender,

            prediction,
            confidence,

            facial_assessment,
            facial_asymmetry,

            clinical_risk_probability,

            overall_status,

            priority,
            timestamp

        FROM cases

        ORDER BY case_id DESC
    """)

    cases = cursor.fetchall()

    connection.close()

    return cases


# ============================================================
# DISPLAY CASES
# ============================================================

def display_cases():

    cases = get_all_cases()

    print(
        "\n=============================================="
    )

    print(
        "          NEURO.AI - CASE DATABASE"
    )

    print(
        "=============================================="
    )


    if not cases:

        print(
            "No cases found."
        )

        return


    for case in cases:

        print(
            "\n----------------------------------------------"
        )

        print(
            f"Case ID              : {case[0]}"
        )

        print(
            f"Patient ID           : {case[1]}"
        )

        print(
            f"Patient Name         : {case[2]}"
        )

        print(
            f"Age                  : {case[3]}"
        )

        print(
            f"Gender               : {case[4]}"
        )

        print(
            f"CT Prediction        : {case[5]}"
        )

        print(
            f"CT Confidence        : "
            f"{case[6]:.2f}%"
        )

        print(
            f"Facial Assessment    : {case[7]}"
        )

        print(
            f"Facial Asymmetry     : "
            f"{case[8] or 0:.2f}%"
        )

        print(
            f"Clinical Risk        : "
            f"{case[9] or 0:.2f}%"
        )

        print(
            f"Overall Status       : {case[10]}"
        )

        print(
            f"Priority             : {case[11]}"
        )

        print(
            f"Timestamp            : {case[12]}"
        )


    print(
        "\n=============================================="
    )


# ============================================================
# SEARCH CASE BY PATIENT ID
# ============================================================

def search_case(
    patient_id
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM cases
        WHERE patient_id = ?
        ORDER BY case_id DESC
        """,
        (
            patient_id,
        )
    )

    case = cursor.fetchone()

    connection.close()

    return case


# ============================================================
# SEARCH CASE BY CASE ID
# ============================================================

def search_case_by_id(
    case_id
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM cases
        WHERE case_id = ?
        """,
        (
            case_id,
        )
    )

    case = cursor.fetchone()

    connection.close()

    return case


# ============================================================
# SAMPLE CASE FOR TESTING
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Initialize database
    # --------------------------------------------------------

    initialize_database()


    # --------------------------------------------------------
    # Sample Neuro.ai case
    # --------------------------------------------------------

    sample_case = {

        "patient_id":
            "NV001",

        "patient_name":
            "Sample Patient",

        "age":
            62,

        "gender":
            "Male",

        "ct_file":
            "sample_ct_scan.png",

        "ct_metadata":
            "CT brain scan - sample image",

        "prediction":
            "Ischemic Stroke",

        "confidence":
            94.5,

        "probabilities": {

            "Ischemic Stroke":
                94.5,

            "Hemorrhagic Stroke":
                3.0,

            "Normal":
                2.5
        },

        "facial_assessment":
            "Moderate asymmetry",

        "facial_asymmetry":
            14.0,

        "clinical_risk_probability":
            61.8,

        "clinical_inputs": {

            "Blood Pressure":
                "150/95 mmHg",

            "Symptoms":
                "Sudden weakness and speech difficulty",

            "Medical History":
                "Hypertension"
        },

        "gradcam":
            "Grad-CAM visualization available.",

        "lesion_analysis":
            (
                "Potential abnormal region identified. "
                "Further clinical evaluation is required."
            ),

        "overall_status":
            "Stroke pattern detected on CT",

        "report_file":
            "neurovision_stroke_report.pdf"
    }


    # --------------------------------------------------------
    # Add sample case
    # --------------------------------------------------------

    case_id = add_case(
        sample_case
    )


    print(
        f"\nSample Case ID: {case_id}"
    )


    # --------------------------------------------------------
    # Display cases
    # --------------------------------------------------------

    display_cases()