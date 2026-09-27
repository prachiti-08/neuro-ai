# ============================================================
# Neuro.ai - Clinical Indicators Module
# Logistic Regression Training
# ============================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "healthcare-dataset-stroke-data.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 60)
print("LOADING CLINICAL DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 3. BASIC DATA INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nStroke distribution:")
print(df["stroke"].value_counts())

print("\nStroke percentage:")
print(df["stroke"].value_counts(normalize=True) * 100)


# ============================================================
# 4. REMOVE UNNECESSARY COLUMN
# ============================================================

# Patient ID does not contain useful clinical information
if "id" in df.columns:
    df = df.drop(columns=["id"])


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

TARGET = "stroke"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = [
    "age",
    "avg_glucose_level",
    "bmi"
]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

# Binary clinical indicators
binary_features = [
    "hypertension",
    "heart_disease"
]

# Add binary features to numerical features
numeric_features = numeric_features + binary_features


print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

# First: 80% training, 20% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Second: divide temporary set equally
# Final = 80% train, 10% validation, 10% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)


print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(f"Training samples   : {len(X_train)}")
print(f"Validation samples : {len(X_val)}")
print(f"Testing samples    : {len(X_test)}")

print("\nStroke distribution:")

print("\nTrain:")
print(y_train.value_counts())

print("\nValidation:")
print(y_val.value_counts())

print("\nTest:")
print(y_test.value_counts())


# ============================================================
# 8. PREPROCESSING
# ============================================================

# Numerical preprocessing:
# - Missing values replaced by median
# - Features standardized

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing:
# - Missing values replaced with most frequent value
# - One-hot encoding

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 9. CREATE LOGISTIC REGRESSION MODEL
# ============================================================

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)


# ============================================================
# 10. COMPLETE PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            model
        )
    ]
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CLINICAL MODEL")
print("=" * 60)

pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# 12. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_pipeline, X_data, y_data, dataset_name):

    predictions = model_pipeline.predict(X_data)

    probabilities = model_pipeline.predict_proba(X_data)[:, 1]

    accuracy = accuracy_score(
        y_data,
        predictions
    )

    precision = precision_score(
        y_data,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_data,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_data,
        predictions,
        zero_division=0
    )

    balanced_acc = balanced_accuracy_score(
        y_data,
        predictions
    )

    roc_auc = roc_auc_score(
        y_data,
        probabilities
    )

    pr_auc = average_precision_score(
        y_data,
        probabilities
    )

    cm = confusion_matrix(
        y_data,
        predictions
    )

    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} RESULTS")
    print("=" * 60)

    print(f"Accuracy            : {accuracy:.4f}")
    print(f"Precision           : {precision:.4f}")
    print(f"Recall              : {recall:.4f}")
    print(f"F1 Score            : {f1:.4f}")
    print(f"Balanced Accuracy   : {balanced_acc:.4f}")
    print(f"ROC-AUC             : {roc_auc:.4f}")
    print(f"PR-AUC              : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_data,
            predictions,
            target_names=[
                "No Stroke",
                "Stroke"
            ],
            zero_division=0
        )
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_acc),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# 13. VALIDATION EVALUATION
# ============================================================

val_results = evaluate_model(
    pipeline,
    X_val,
    y_val,
    "Validation"
)


# ============================================================
# 14. FINAL TEST EVALUATION
# ============================================================

test_results = evaluate_model(
    pipeline,
    X_test,
    y_test,
    "Test"
)


# ============================================================
# 15. SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "clinical_logistic_regression.joblib"
)

joblib.dump(
    pipeline,
    model_path
)

print("\nModel saved to:")
print(model_path)


# ============================================================
# 16. SAVE METRICS
# ============================================================

metrics = {
    "model": "Logistic Regression",
    "dataset": "Stroke Prediction Dataset",
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "validation": val_results,
    "test": test_results
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "clinical_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


print("\nMetrics saved to:")
print(metrics_path)


# ============================================================
# 17. SAVE TEST PREDICTIONS
# ============================================================

test_output = X_test.copy()

test_output["actual_stroke"] = y_test.values
test_output["predicted_stroke"] = pipeline.predict(X_test)
test_output["stroke_probability"] = pipeline.predict_proba(X_test)[:, 1]

predictions_path = os.path.join(
    OUTPUT_DIR,
    "clinical_test_predictions.csv"
)

test_output.to_csv(
    predictions_path,
    index=False
)

print("\nTest predictions saved to:")
print(predictions_path)


# ============================================================
# 18. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("CLINICAL MODEL TRAINING COMPLETED")
print("=" * 60)