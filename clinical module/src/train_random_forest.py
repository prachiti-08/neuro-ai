# ============================================================
# Neuro.ai - Clinical Indicators Module
# Random Forest Training
# ============================================================

import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

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
# 2. LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING CLINICAL DATASET")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 3. REMOVE ID
# ============================================================

if "id" in df.columns:
    df = df.drop(columns=["id"])


# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

TARGET = "stroke"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 5. FEATURE TYPES
# ============================================================

numeric_features = [
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease"
]

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

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


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

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
# 8. RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=5,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# ============================================================
# 9. COMPLETE PIPELINE
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
# 10. TRAIN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST")
print("=" * 60)

pipeline.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_pipeline,
    X_data,
    y_data,
    dataset_name
):

    predictions = model_pipeline.predict(X_data)

    probabilities = model_pipeline.predict_proba(
        X_data
    )[:, 1]

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
# 12. VALIDATION
# ============================================================

val_results = evaluate_model(
    pipeline,
    X_val,
    y_val,
    "Validation"
)


# ============================================================
# 13. TEST
# ============================================================

test_results = evaluate_model(
    pipeline,
    X_test,
    y_test,
    "Test"
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "clinical_random_forest.joblib"
)

joblib.dump(
    pipeline,
    model_path
)

print("\nModel saved to:")
print(model_path)


# ============================================================
# 15. SAVE METRICS
# ============================================================

metrics = {
    "model": "Random Forest",
    "dataset": "Stroke Prediction Dataset",
    "training_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "validation": val_results,
    "test": test_results
}

metrics_path = os.path.join(
    OUTPUT_DIR,
    "clinical_random_forest_metrics.json"
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
# 16. SAVE TEST PREDICTIONS
# ============================================================

test_output = X_test.copy()

test_output["actual_stroke"] = y_test.values

test_output["predicted_stroke"] = pipeline.predict(
    X_test
)

test_output["stroke_probability"] = (
    pipeline.predict_proba(X_test)[:, 1]
)

predictions_path = os.path.join(
    OUTPUT_DIR,
    "clinical_random_forest_predictions.csv"
)

test_output.to_csv(
    predictions_path,
    index=False
)

print("\nTest predictions saved to:")
print(predictions_path)


# ============================================================
# 17. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("RANDOM FOREST TRAINING COMPLETED")
print("=" * 60)