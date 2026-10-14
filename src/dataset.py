"""
FACIAL PALSY DATASET PREPARATION
================================

Leakage-safe, case-group-aware, severity-stratified splitting.

IMPORTANT DATASET RULE
----------------------
Numeric IDs are NOT necessarily unique clinical cases.

Therefore:

    numeric sample ID + severity = case group

Example:

    10_Mi_eye1.bmp
    10_M_eye1.bmp

share numeric ID 10, but represent different severity cases.

Therefore they MUST NOT automatically be placed in the
same train/validation/test split.

Rules
-----
1. Same numeric ID + same severity -> same case group.
2. Eye / Eyebrow / Mouth belonging to the same case group
   stay together.
3. TRAIN:
      original + pre-generated augmentations
4. VALIDATION:
      original images only
5. TEST:
      original images only
6. Images without a reliable numeric ID:
      TRAIN ONLY
7. Every severity class must appear in:
      TRAIN / VALIDATION / TEST

Severity classes
----------------
Mild
Moderate
Moderate Severe
Severe
"""

from __future__ import annotations

import re
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "archive"
OUTPUT_DIR = BASE_DIR / "outputs"
DIAGNOSTIC_DIR = OUTPUT_DIR / "diagnostics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

REGIONS = {
    "eye": DATA_DIR / "Eye" / "Eye",
    "eyebrow": DATA_DIR / "Eyebrow" / "Eyebrow",
    "mouth": DATA_DIR / "Mouth" / "Mouth",
}

SEVERITIES = [
    "Mild",
    "Moderate",
    "Moderate Severe",
    "Severe",
]

SEVERITY_TO_INDEX = {
    "Mild": 0,
    "Moderate": 1,
    "Moderate Severe": 2,
    "Severe": 3,
}

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(text: str) -> str:

    text = text.lower()
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================================
# SEVERITY EXTRACTION
# ============================================================================

def extract_severity(folder_name: str) -> str | None:

    normalized = normalize_text(folder_name)

    if "moderate severe" in normalized:
        return "Moderate Severe"

    if normalized.startswith("mild"):
        return "Mild"

    if normalized.startswith("moderate"):
        return "Moderate"

    if normalized.startswith("severe"):
        return "Severe"

    return None


# ============================================================================
# NUMERIC SAMPLE ID
# ============================================================================

def extract_sample_id(filename: str) -> int | None:
    """
    Extract the numeric ID from filenames such as:

        14_Mi_eye1.bmp
        10_M_eye1.bmp
        15_M.S_eye1.bmp
        3_S_eye1.bmp

        rotated_min1014_Mi_eye1.bmp
        translated_minus10_Mi_eyebrow1.bmp
        noisy0.5_10_Mi_eyebrow1.bmp
        cropped200_9_Mi_eyebrow1.bmp

    Numeric-only files such as:

        1276.bmp

    are treated as unknown because there is no reliable
    severity marker connecting the numeric value to a case.
    """

    stem = Path(filename).stem

    patterns = [

        # Moderate Severe
        r"(?P<id>\d+)_M\.S_",

        # Mild
        r"(?P<id>\d+)_Mi_",

        # Moderate
        r"(?P<id>\d+)_M_",

        # Severe
        r"(?P<id>\d+)_S_",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            stem,
            flags=re.IGNORECASE,
        )

        if match:
            return int(match.group("id"))

    return None


# ============================================================================
# CASE GROUP
# ============================================================================

def create_case_group(
    sample_id,
    severity,
):
    """
    Critical fix.

    A numeric ID can correspond to multiple severity cases.

    Therefore:

        case_group = numeric_id + severity

    Example:

        ID 10 + Mild
        ID 10 + Moderate

    are two different case groups.
    """

    if pd.isna(sample_id):

        return None

    return (
        f"{int(sample_id)}__"
        f"{severity.replace(' ', '_')}"
    )


# ============================================================================
# AUGMENTATION
# ============================================================================

def detect_augmentation(filename: str) -> str:

    stem = Path(filename).stem.lower()

    if "rotated" in stem or "rotate" in stem:
        return "rotated"

    if "cropped" in stem or "crop" in stem:
        return "cropped"

    if "noisy" in stem or "noise" in stem:
        return "noisy"

    if "translated" in stem or "translate" in stem:
        return "translated"

    return "original"


# ============================================================================
# FILENAME PATTERN
# ============================================================================

def detect_filename_pattern(filename: str) -> str:

    stem = Path(filename).stem

    if re.fullmatch(r"\d+", stem):
        return "numeric_only"

    lower = stem.lower()

    if "rotated" in lower or "rotate" in lower:
        return "augmented_rotated"

    if "cropped" in lower or "crop" in lower:
        return "augmented_cropped"

    if "noisy" in lower or "noise" in lower:
        return "augmented_noisy"

    if "translated" in lower or "translate" in lower:
        return "augmented_translated"

    if re.search(r"\d+_M\.S_", stem, re.I):
        return "standard_moderate_severe"

    if re.search(r"\d+_Mi_", stem, re.I):
        return "standard_mild"

    if re.search(r"\d+_M_", stem, re.I):
        return "standard_moderate"

    if re.search(r"\d+_S_", stem, re.I):
        return "standard_severe"

    return "other"


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def get_image_files(folder: Path) -> list[Path]:

    if not folder.exists():
        return []

    return sorted(
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    )


# ============================================================================
# PROCESS REGION
# ============================================================================

def process_region(
    region: str,
    folder: Path,
) -> list[dict]:

    print()
    print("=" * 70)
    print(f"SCANNING {region.upper()}")
    print("=" * 70)

    files = get_image_files(folder)

    print(
        f"Images found: {len(files):,}"
    )

    records = []

    for image_path in files:

        severity_folder = image_path.parent.name

        severity = extract_severity(
            severity_folder
        )

        if severity is None:
            continue

        sample_id = extract_sample_id(
            image_path.name
        )

        augmentation = detect_augmentation(
            image_path.name
        )

        pattern = detect_filename_pattern(
            image_path.name
        )

        case_group = create_case_group(
            sample_id,
            severity,
        )

        records.append({
            "image_path": str(image_path),
            "filename": image_path.name,
            "region": region,
            "severity": severity,
            "sample_id": sample_id,
            "case_group": case_group,
            "augmentation": augmentation,
            "filename_pattern": pattern,
        })

    print(
        f"Records created: {len(records):,}"
    )

    return records


# ============================================================================
# BUILD DATASET
# ============================================================================

def build_dataset() -> pd.DataFrame:

    print()
    print("=" * 70)
    print("FACIAL PALSY DATASET PREPARATION")
    print("=" * 70)

    print()
    print("DATA DIRECTORY:")
    print(DATA_DIR)

    all_records = []

    for region, folder in REGIONS.items():

        records = process_region(
            region,
            folder,
        )

        all_records.extend(records)

    df = pd.DataFrame(
        all_records
    )

    if df.empty:

        raise RuntimeError(
            "No images found. "
            "Check data/archive structure."
        )

    return df


# ============================================================================
# DATASET DISCOVERY
# ============================================================================

def print_discovery(
    df: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("DATASET DISCOVERY")
    print("=" * 70)

    print()
    print(
        f"Total images discovered: "
        f"{len(df):,}"
    )

    print()
    print("Images by region:")
    print(
        df["region"].value_counts()
    )

    print()
    print("Images by severity:")
    print(
        df["severity"]
        .value_counts()
        .reindex(
            SEVERITIES,
            fill_value=0,
        )
    )

    print()
    print("Images by augmentation:")
    print(
        df["augmentation"]
        .value_counts()
    )

    known = df[
        df["sample_id"].notna()
    ]

    unknown = df[
        df["sample_id"].isna()
    ]

    print()
    print(
        f"Images with embedded sample ID: "
        f"{len(known):,}"
    )

    print(
        f"Images without reliable sample ID: "
        f"{len(unknown):,}"
    )

    print()
    print(
        f"Unique numeric IDs: "
        f"{known['sample_id'].nunique():,}"
    )

    print(
        f"Unique case groups: "
        f"{known['case_group'].nunique():,}"
    )


# ============================================================================
# CASE GROUP DIAGNOSTIC
# ============================================================================

def print_case_group_diagnostic(
    df: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("CASE GROUP DIAGNOSTIC")
    print("=" * 70)

    known = df[
        df["case_group"].notna()
    ].copy()

    print()
    print(
        f"Images with extracted sample ID: "
        f"{len(known):,}"
    )

    print(
        f"Unique numeric IDs: "
        f"{known['sample_id'].nunique():,}"
    )

    print(
        f"Unique case groups: "
        f"{known['case_group'].nunique():,}"
    )

    print()
    print("NUMERIC ID → SEVERITY STRUCTURE")

    id_severity = (
        known.groupby(
            "sample_id"
        )["severity"]
        .unique()
    )

    for sid in sorted(
        id_severity.index
    ):

        severities = list(
            id_severity.loc[sid]
        )

        if len(severities) > 1:

            print(
                f"  ID {int(sid):>2}: "
                f"{severities}"
            )

    print()
    print("CASE GROUPS BY SEVERITY:")

    case_counts = (
        known.groupby(
            "severity"
        )["case_group"]
        .nunique()
        .reindex(
            SEVERITIES,
            fill_value=0,
        )
    )

    print(
        case_counts
    )

    print()
    print("REGION COVERAGE PER CASE GROUP:")

    region_counts = (
        known.groupby(
            "case_group"
        )["region"]
        .nunique()
        .value_counts()
        .sort_index()
    )

    print(
        region_counts
    )


# ============================================================================
# CREATE CASE GROUP TABLE
# ============================================================================

def create_case_group_table(
    df: pd.DataFrame,
) -> pd.DataFrame:

    known = df[
        df["case_group"].notna()
    ].copy()

    rows = []

    for case_group, group in known.groupby(
        "case_group"
    ):

        severity_values = (
            group["severity"]
            .unique()
        )

        if len(severity_values) != 1:

            raise RuntimeError(
                f"Case group {case_group} "
                f"contains multiple severities."
            )

        severity = severity_values[0]

        sample_id = int(
            group["sample_id"].iloc[0]
        )

        row = {
            "case_group": case_group,
            "sample_id": sample_id,
            "severity": severity,
            "total_images": len(group),
            "original_images": int(
                (
                    group["augmentation"]
                    == "original"
                ).sum()
            ),
            "region_count": group[
                "region"
            ].nunique(),
        }

        for region in REGIONS:

            row[
                f"{region}_images"
            ] = int(
                (
                    group["region"]
                    == region
                ).sum()
            )

        rows.append(row)

    case_df = pd.DataFrame(
        rows
    )

    if case_df.empty:

        raise RuntimeError(
            "No valid case groups found."
        )

    return case_df


# ============================================================================
# DETERMINISTIC STRATIFIED SPLIT
# ============================================================================

def create_case_group_split(
    case_df: pd.DataFrame,
) -> dict[str, str]:

    """
    Create a severity-stratified case-group split.

    This intentionally does NOT attempt to hit exact image-level
    70/15/15 ratios because leakage-safe case grouping is more
    important.

    Per severity:

        n >= 3:
            train / validation / test

        Severe has only 4 groups:
            2 train / 1 validation / 1 test

    Larger classes are distributed approximately 70/15/15.
    """

    print()
    print("=" * 70)
    print("CREATING LEAKAGE-SAFE CASE-GROUP SPLIT")
    print("=" * 70)

    rng = random.Random(
        RANDOM_SEED
    )

    assignments = {}

    for severity in SEVERITIES:

        subset = case_df[
            case_df["severity"]
            == severity
        ].copy()

        groups = (
            subset["case_group"]
            .tolist()
        )

        rng.shuffle(groups)

        n = len(groups)

        if n < 3:

            raise RuntimeError(
                f"Severity '{severity}' has only "
                f"{n} case groups. At least 3 "
                f"are required."
            )

        # ------------------------------------------------------------
        # Calculate counts.
        # ------------------------------------------------------------

        if severity == "Severe" and n == 4:

            n_test = 1
            n_val = 1
            n_train = 2

        else:

            n_test = max(
                1,
                round(n * TEST_RATIO),
            )

            n_val = max(
                1,
                round(n * VAL_RATIO),
            )

            n_train = (
                n
                - n_val
                - n_test
            )

            if n_train < 1:

                n_train = 1

                if n_val > 1:
                    n_val -= 1

                elif n_test > 1:
                    n_test -= 1

        # ------------------------------------------------------------
        # Safety correction.
        # ------------------------------------------------------------

        while (
            n_train
            + n_val
            + n_test
            < n
        ):

            n_train += 1

        while (
            n_train
            + n_val
            + n_test
            > n
        ):

            if n_train > n_val:
                n_train -= 1

            elif n_val > n_test:
                n_val -= 1

            else:
                n_test -= 1

        # ------------------------------------------------------------
        # Assign.
        # ------------------------------------------------------------

        train_groups = groups[
            :n_train
        ]

        val_groups = groups[
            n_train:
            n_train + n_val
        ]

        test_groups = groups[
            n_train + n_val:
        ]

        for case_group in train_groups:
            assignments[
                case_group
            ] = "train"

        for case_group in val_groups:
            assignments[
                case_group
            ] = "validation"

        for case_group in test_groups:
            assignments[
                case_group
            ] = "test"

        print()
        print(
            f"{severity}: "
            f"{n} case groups"
        )

        print(
            f"  Train:      {len(train_groups)}"
        )

        print(
            f"  Validation: {len(val_groups)}"
        )

        print(
            f"  Test:       {len(test_groups)}"
        )

    # ------------------------------------------------------------------------
    # Verify all groups assigned exactly once.
    # ------------------------------------------------------------------------

    expected = set(
        case_df["case_group"]
    )

    actual = set(
        assignments.keys()
    )

    if expected != actual:

        missing = expected - actual
        extra = actual - expected

        raise RuntimeError(
            "Case-group assignment mismatch.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    # ------------------------------------------------------------------------
    # Verify all splits present.
    # ------------------------------------------------------------------------

    required = {
        "train",
        "validation",
        "test",
    }

    if set(
        assignments.values()
    ) != required:

        raise RuntimeError(
            "Not all train/validation/test "
            "splits were created."
        )

    return assignments


# ============================================================================
# ASSIGN IMAGE SPLITS
# ============================================================================

def assign_splits(
    df: pd.DataFrame,
    assignments: dict[str, str],
) -> pd.DataFrame:

    df = df.copy()

    def get_split(case_group):

        # Unknown IDs -> TRAIN ONLY.
        if pd.isna(case_group):

            return "train"

        if case_group not in assignments:

            raise RuntimeError(
                f"Case group {case_group} "
                f"has no split assignment."
            )

        return assignments[
            case_group
        ]

    df["split"] = df[
        "case_group"
    ].apply(get_split)

    return df


# ============================================================================
# REMOVE AUGMENTED EVALUATION IMAGES
# ============================================================================

def enforce_original_only_eval(
    df: pd.DataFrame,
):

    eval_mask = df[
        "split"
    ].isin(
        [
            "validation",
            "test",
        ]
    )

    augmented_mask = (
        df["augmentation"]
        != "original"
    )

    remove_mask = (
        eval_mask
        & augmented_mask
    )

    removed = df[
        remove_mask
    ].copy()

    removed[
        "excluded_reason"
    ] = (
        "pre_generated_augmentation_removed_from_eval"
    )

    clean = df[
        ~remove_mask
    ].copy()

    return clean, removed


# ============================================================================
# LEAKAGE CHECK
# ============================================================================

def check_case_group_leakage(
    df: pd.DataFrame,
):

    known = df[
        df["case_group"].notna()
    ]

    split_counts = (
        known.groupby(
            "case_group"
        )["split"]
        .nunique()
    )

    leaked = split_counts[
        split_counts > 1
    ]

    if not leaked.empty:

        print()
        print(
            "!!! CASE-GROUP LEAKAGE DETECTED !!!"
        )

        print(
            leaked
        )

        return False

    print(
        "✓ No case-group leakage detected."
    )

    return True


# ============================================================================
# NUMERIC ID DIAGNOSTIC
# ============================================================================

def check_numeric_id_behavior(
    df: pd.DataFrame,
):

    """
    Numeric IDs are allowed to occur in multiple splits
    ONLY when they correspond to different severity case groups.

    This is intentional.
    """

    known = df[
        df["sample_id"].notna()
    ]

    problems = []

    for sample_id, group in known.groupby(
        "sample_id"
    ):

        for severity, sev_group in group.groupby(
            "severity"
        ):

            splits = (
                sev_group["split"]
                .unique()
            )

            if len(splits) > 1:

                problems.append({
                    "sample_id": int(sample_id),
                    "severity": severity,
                    "splits": list(splits),
                })

    if problems:

        print()
        print(
            "!!! SAME ID + SAME SEVERITY "
            "APPEARS IN MULTIPLE SPLITS !!!"
        )

        print(
            pd.DataFrame(problems)
        )

        return False

    print(
        "✓ Same numeric ID + severity "
        "remains in one split."
    )

    return True


# ============================================================================
# CROSS-REGION CHECK
# ============================================================================

def check_cross_region_consistency(
    df: pd.DataFrame,
):

    known = df[
        df["case_group"].notna()
    ]

    problems = []

    for case_group, group in known.groupby(
        "case_group"
    ):

        splits = (
            group["split"]
            .unique()
        )

        if len(splits) != 1:

            problems.append({
                "case_group": case_group,
                "splits": list(splits),
            })

    if problems:

        print(
            "!!! CROSS-REGION SPLIT ERROR !!!"
        )

        print(
            pd.DataFrame(problems)
        )

        return False

    print(
        "✓ Eye/Eyebrow/Mouth images belonging "
        "to the same case group remain together."
    )

    return True


# ============================================================================
# EVALUATION PROTOCOL CHECK
# ============================================================================

def check_evaluation_protocol(
    df: pd.DataFrame,
):

    problems = df[
        df["split"].isin(
            [
                "validation",
                "test",
            ]
        )
        &
        (
            df["augmentation"]
            != "original"
        )
    ]

    if not problems.empty:

        print(
            "!!! EVALUATION PROTOCOL ERROR !!!"
        )

        return False

    print(
        "✓ Validation/test contain "
        "original images only."
    )

    return True


# ============================================================================
# UNKNOWN ID CHECK
# ============================================================================

def check_unknown_ids(
    df: pd.DataFrame,
):

    unknown = df[
        df["sample_id"].isna()
    ]

    if unknown.empty:

        print(
            "✓ No unknown sample IDs."
        )

        return True

    invalid = unknown[
        unknown["split"] != "train"
    ]

    if not invalid.empty:

        print(
            "!!! UNKNOWN-ID IMAGE "
            "OUTSIDE TRAIN !!!"
        )

        print(
            invalid
        )

        return False

    print(
        f"✓ {len(unknown):,} images without "
        f"reliable sample IDs are TRAIN ONLY."
    )

    return True


# ============================================================================
# CLASS PRESENCE
# ============================================================================

def check_all_classes_present(
    df: pd.DataFrame,
):

    print()
    print(
        "FINAL CLASS-PRESENCE CHECK"
    )

    valid = True

    for split in [
        "train",
        "validation",
        "test",
    ]:

        subset = df[
            df["split"] == split
        ]

        counts = (
            subset["severity"]
            .value_counts()
            .reindex(
                SEVERITIES,
                fill_value=0,
            )
        )

        print()
        print(
            split.upper()
        )

        for severity in SEVERITIES:

            count = int(
                counts[severity]
            )

            status = (
                "✓"
                if count > 0
                else "✗"
            )

            print(
                f"  {status} "
                f"{severity:<18} "
                f"{count}"
            )

            if count == 0:
                valid = False

    if not valid:

        raise RuntimeError(
            "At least one split is missing "
            "a severity class."
        )

    print()
    print(
        "✓ ALL FOUR CLASSES ARE PRESENT "
        "IN TRAIN, VALIDATION AND TEST."
    )


# ============================================================================
# SPLIT SUMMARY
# ============================================================================

def print_split_summary(
    df: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("FINAL SPLIT SUMMARY")
    print("=" * 70)

    print()
    print("IMAGES BY SPLIT:")

    print(
        df["split"]
        .value_counts()
    )

    print()
    print("SPLIT × SEVERITY:")

    table = (
        pd.crosstab(
            df["split"],
            df["severity"],
        )
        .reindex(
            index=[
                "train",
                "validation",
                "test",
            ],
            columns=SEVERITIES,
            fill_value=0,
        )
    )

    print(
        table
    )

    print()
    print("SPLIT × REGION:")

    print(
        pd.crosstab(
            df["split"],
            df["region"],
        )
    )

    print()
    print("SPLIT × AUGMENTATION:")

    print(
        pd.crosstab(
            df["split"],
            df["augmentation"],
        )
    )

    print()
    print("CASE GROUPS BY SPLIT:")

    known = df[
        df["case_group"].notna()
    ]

    case_split = (
        known.groupby(
            "split"
        )["case_group"]
        .nunique()
    )

    print(
        case_split
    )


# ============================================================================
# SAVE DIAGNOSTICS
# ============================================================================

def save_diagnostics(
    df: pd.DataFrame,
    case_df: pd.DataFrame,
    removed_eval: pd.DataFrame,
):

    all_path = (
        DIAGNOSTIC_DIR
        / "all_images_diagnostic.csv"
    )

    df.to_csv(
        all_path,
        index=False,
    )

    case_path = (
        DIAGNOSTIC_DIR
        / "case_group_summary.csv"
    )

    case_df.to_csv(
        case_path,
        index=False,
    )

    removed_path = (
        DIAGNOSTIC_DIR
        / "removed_eval_augmentations.csv"
    )

    removed_eval.to_csv(
        removed_path,
        index=False,
    )

    print()
    print(
        "Diagnostics saved:"
    )

    print(
        all_path
    )

    print(
        case_path
    )

    print(
        removed_path
    )


# ============================================================================
# SAVE OUTPUT CSV
# ============================================================================

def save_outputs(
    df: pd.DataFrame,
):

    train = df[
        df["split"] == "train"
    ].copy()

    val = df[
        df["split"] == "validation"
    ].copy()

    test = df[
        df["split"] == "test"
    ].copy()

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    val.to_csv(
        OUTPUT_DIR / "val.csv",
        index=False,
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    df.to_csv(
        OUTPUT_DIR
        / "clean_dataset_metadata.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print()
    print(
        f"Train:\n"
        f"{OUTPUT_DIR / 'train.csv'}"
    )

    print()
    print(
        f"Validation:\n"
        f"{OUTPUT_DIR / 'val.csv'}"
    )

    print()
    print(
        f"Test:\n"
        f"{OUTPUT_DIR / 'test.csv'}"
    )

    print()
    print(
        f"Metadata:\n"
        f"{OUTPUT_DIR / 'clean_dataset_metadata.csv'}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    # ------------------------------------------------------------------------
    # 1. DISCOVER
    # ------------------------------------------------------------------------

    df = build_dataset()

    print_discovery(
        df
    )

    print_case_group_diagnostic(
        df
    )

    # ------------------------------------------------------------------------
    # 2. CREATE CASE GROUP TABLE
    # ------------------------------------------------------------------------

    case_df = create_case_group_table(
        df
    )

    print()
    print("=" * 70)
    print("CASE GROUP ANALYSIS")
    print("=" * 70)

    print()
    print(
        f"Unique numeric IDs: "
        f"{df['sample_id'].nunique():,}"
    )

    print(
        f"Unique case groups: "
        f"{len(case_df):,}"
    )

    print()
    print("Case groups by severity:")

    print(
        case_df[
            "severity"
        ]
        .value_counts()
        .reindex(
            SEVERITIES,
            fill_value=0,
        )
    )

    print()
    print(
        "Case groups containing multiple regions:"
    )

    multi_region = (
        case_df[
            "region_count"
        ] > 1
    ).sum()

    print(
        f"{multi_region:,}"
    )

    # ------------------------------------------------------------------------
    # 3. CREATE STRATIFIED CASE SPLIT
    # ------------------------------------------------------------------------

    assignments = create_case_group_split(
        case_df
    )

    # ------------------------------------------------------------------------
    # 4. ASSIGN IMAGES
    # ------------------------------------------------------------------------

    df = assign_splits(
        df,
        assignments,
    )

    # ------------------------------------------------------------------------
    # 5. REMOVE AUGMENTATIONS FROM EVAL
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION IMAGE CLEANING")
    print("=" * 70)

    df, removed_eval = (
        enforce_original_only_eval(
            df
        )
    )

    print()
    print(
        "Pre-generated augmented images "
        "removed from validation/test: "
        f"{len(removed_eval):,}"
    )

    # ------------------------------------------------------------------------
    # 6. SAFETY CHECKS
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SAFETY CHECKS")
    print("=" * 70)

    if not check_case_group_leakage(
        df
    ):

        raise RuntimeError(
            "CASE-GROUP LEAKAGE DETECTED."
        )

    if not check_numeric_id_behavior(
        df
    ):

        raise RuntimeError(
            "Same numeric ID + severity "
            "appears in multiple splits."
        )

    if not check_cross_region_consistency(
        df
    ):

        raise RuntimeError(
            "CROSS-REGION SPLIT ERROR."
        )

    if not check_evaluation_protocol(
        df
    ):

        raise RuntimeError(
            "Validation/test contain "
            "augmented images."
        )

    if not check_unknown_ids(
        df
    ):

        raise RuntimeError(
            "Images without sample IDs "
            "were assigned outside TRAIN."
        )

    check_all_classes_present(
        df
    )

    # ------------------------------------------------------------------------
    # 7. SUMMARY
    # ------------------------------------------------------------------------

    print_split_summary(
        df
    )

    # ------------------------------------------------------------------------
    # 8. SAVE
    # ------------------------------------------------------------------------

    save_diagnostics(
        df,
        case_df,
        removed_eval,
    )

    save_outputs(
        df
    )

    # ------------------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print()
    print(
        "✓ Same case_group stays in one split"
    )

    print(
        "✓ Numeric ID + severity defines case"
    )

    print(
        "✓ Cross-region leakage prevented"
    )

    print(
        "✓ Missing-ID images restricted to TRAIN"
    )

    print(
        "✓ Augmented images retained in TRAIN"
    )

    print(
        "✓ Validation contains ORIGINAL images only"
    )

    print(
        "✓ Test contains ORIGINAL images only"
    )

    print(
        "✓ Mild present in all splits"
    )

    print(
        "✓ Moderate present in all splits"
    )

    print(
        "✓ Moderate Severe present in all splits"
    )

    print(
        "✓ Severe present in all splits"
    )

    print(
        "✓ Leakage checks passed"
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Run:"
    )

    print(
        "python src\\train.py"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()