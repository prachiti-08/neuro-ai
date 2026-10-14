"""
Facial Palsy Dataset Audit
==========================

Audits the complete YFP dataset before model training.

The audit intentionally uses:

    1. Folder structure -> region + severity
    2. Filename -> sample identifier
    3. File extension -> image type

We do NOT depend on the severity abbreviation in the filename,
because the dataset contains variations such as:

    Mi
    M
    Mo
    MS
    S

The parent folder is treated as the authoritative severity label.

Example:

    Eye/Eye/Mild eye/rotated_min1014_Mi_eye1.bmp

becomes:

    region       = eye
    severity     = Mild
    sample_id    = 1014

Example:

    Mouth/Mouth/Moderate mouth/14_M_mouth1.bmp

becomes:

    region       = mouth
    severity     = Moderate
    sample_id    = 14
"""

from pathlib import Path
from collections import Counter
import re

import pandas as pd

from config import (
    REGIONS,
    CLASSES,
    SUPPORTED_EXTENSIONS,
    OUTPUT_DIR,
)


# ============================================================
# SEVERITY FOLDER MAPPING
# ============================================================

SEVERITY_FOLDER_MAPPING = {
    "mild": "Mild",
    "moderate": "Moderate",
    "moderate severe": "Moderate Severe",
    "moderately severe": "Moderate Severe",
    "severe": "Severe",
}


# ============================================================
# EXTRACT SEVERITY FROM FOLDER
# ============================================================

def get_severity_from_folder(folder_name):
    """
    Determine severity using the dataset folder name.

    Examples:

        Mild eye
        Moderate eye
        Moderate severe eye
        Severe eye

    or:

        Mild eyebrow
        Moderate eyebrow
        Moderate severe eyebrow
        Severe eyebrow
    """

    normalized = (
        folder_name
        .lower()
        .strip()
    )

    # Remove region name
    for region in [
        "eye",
        "eyebrow",
        "mouth",
    ]:

        normalized = normalized.replace(
            region,
            ""
        )

    normalized = (
        normalized
        .strip()
    )

    return SEVERITY_FOLDER_MAPPING.get(
        normalized
    )


# ============================================================
# EXTRACT SAMPLE ID
# ============================================================

def extract_sample_id(filename, region):
    """
    Extract the dataset sample identifier from the filename.

    We look for:

        <number>_<severity>_<region><image_number>

    anywhere inside the filename.

    Examples:

        14_M_mouth1.bmp
        cropped_26_Mi_eye1.bmp
        noisy_18dB_14_Mi_mouth1.bmp
        rotated_min1014_Mi_eye1.bmp
        translated_minus10_Mi_eyebrow1.bmp

    The augmentation prefix is ignored.

    IMPORTANT:
        This function extracts the candidate sample ID.
        It does NOT determine severity.
    """

    stem = Path(filename).stem

    # --------------------------------------------------------
    # Main pattern
    # --------------------------------------------------------

    pattern = (
        rf"(?:^|_)"
        rf"(\d+)_"
        rf"(?:Mi|M|Mo|MS|S)_"
        rf"{re.escape(region)}"
        rf"\d+$"
    )

    match = re.search(
        pattern,
        stem,
        re.IGNORECASE
    )

    if match:

        return int(
            match.group(1)
        )

    # --------------------------------------------------------
    # More permissive fallback
    # --------------------------------------------------------
    #
    # This handles unexpected severity abbreviations while
    # still finding:
    #
    # number + region + image number
    #
    # --------------------------------------------------------

    fallback_pattern = (
        rf"(?:^|_)"
        rf"(\d+)_"
        rf"[A-Za-z]+_"
        rf"{re.escape(region)}"
        rf"\d+$"
    )

    match = re.search(
        fallback_pattern,
        stem,
        re.IGNORECASE
    )

    if match:

        return int(
            match.group(1)
        )

    return None


# ============================================================
# DETECT AUGMENTATION TYPE
# ============================================================

def detect_augmentation(filename):
    """
    Identify the augmentation prefix.

    This is descriptive metadata only.

    Examples:

        original
        cropped
        noisy
        translated
        rotated
        unknown
    """

    stem = Path(filename).stem.lower()

    if stem.startswith("cropped"):
        return "cropped"

    if stem.startswith("noisy"):
        return "noisy"

    if stem.startswith("translated"):
        return "translated"

    if stem.startswith("rotated"):
        return "rotated"

    # No obvious augmentation prefix
    return "original_or_other"


# ============================================================
# FIND ALL IMAGES
# ============================================================

def find_images(folder):
    """
    Recursively find all supported images.
    """

    return sorted(
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and
            path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    )


# ============================================================
# AUDIT ONE REGION
# ============================================================

def audit_region(region):
    """
    Audit one facial region.
    """

    region_folder = REGIONS[
        region
    ]["folder"]

    print()
    print("=" * 70)
    print(
        f"AUDITING {region.upper()}"
    )
    print("=" * 70)

    images = find_images(
        region_folder
    )

    print(
        f"Images found: "
        f"{len(images):,}"
    )

    records = []

    invalid_severity = []
    invalid_sample_id = []

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for image_path in images:

        # ----------------------------------------------------
        # Severity comes from parent folder
        # ----------------------------------------------------

        severity = (
            get_severity_from_folder(
                image_path.parent.name
            )
        )

        if severity is None:

            invalid_severity.append(
                str(image_path)
            )

            continue

        # ----------------------------------------------------
        # Sample ID comes from filename
        # ----------------------------------------------------

        sample_id = extract_sample_id(
            image_path.name,
            region
        )

        if sample_id is None:

            invalid_sample_id.append(
                str(image_path)
            )

            continue

        # ----------------------------------------------------
        # Augmentation type
        # ----------------------------------------------------

        augmentation = (
            detect_augmentation(
                image_path.name
            )
        )

        records.append({

            "path":
                str(
                    image_path.resolve()
                ),

            "filename":
                image_path.name,

            "region":
                region,

            "severity":
                severity,

            "sample_id":
                sample_id,

            "augmentation":
                augmentation,

            "extension":
                image_path.suffix.lower(),

            "folder":
                image_path.parent.name,

        })

    # --------------------------------------------------------
    # Report problems
    # --------------------------------------------------------

    if invalid_severity:

        print(
            f"\nInvalid severity folders: "
            f"{len(invalid_severity):,}"
        )

        for path in invalid_severity[:10]:

            print(
                f"  {path}"
            )

    if invalid_sample_id:

        print(
            f"\nCould not extract sample ID: "
            f"{len(invalid_sample_id):,}"
        )

        for path in invalid_sample_id[:10]:

            print(
                f"  {path}"
            )

    print(
        f"\nSuccessfully audited: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# BUILD COMPLETE AUDIT DATAFRAME
# ============================================================

def build_audit_dataframe():
    """
    Audit Eye, Eyebrow and Mouth.
    """

    all_records = []

    for region in REGIONS:

        records = audit_region(
            region
        )

        all_records.extend(
            records
        )

    if not all_records:

        raise RuntimeError(
            "No valid dataset records were found."
        )

    return pd.DataFrame(
        all_records
    )


# ============================================================
# PRINT DATASET SUMMARY
# ============================================================

def print_summary(df):
    """
    Print complete dataset summary.
    """

    print()
    print("=" * 70)
    print("COMPLETE DATASET SUMMARY")
    print("=" * 70)

    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    print(
        f"\nTotal images audited: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Region
    # --------------------------------------------------------

    print(
        "\nImages by region:"
    )

    region_counts = (
        df
        .groupby("region")
        .size()
    )

    for region, count in region_counts.items():

        print(
            f"  {region:<10} "
            f"{count:>6,}"
        )

    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    print(
        "\nImages by severity:"
    )

    severity_counts = (
        df["severity"]
        .value_counts()
        .reindex(
            CLASSES,
            fill_value=0
        )
    )

    for severity, count in severity_counts.items():

        print(
            f"  {severity:<20} "
            f"{count:>6,}"
        )

    # --------------------------------------------------------
    # Region + severity
    # --------------------------------------------------------

    print(
        "\nImages by region and severity:"
    )

    region_severity = (
        df
        .groupby(
            [
                "region",
                "severity"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    region_severity = (
        region_severity
        .reindex(
            columns=CLASSES,
            fill_value=0
        )
    )

    print(
        region_severity.to_string()
    )

    # --------------------------------------------------------
    # Unique samples
    # --------------------------------------------------------

    print(
        "\nUnique sample IDs:"
    )

    print(
        df["sample_id"]
        .nunique()
    )

    # --------------------------------------------------------
    # Sample IDs
    # --------------------------------------------------------

    sample_ids = sorted(
        df["sample_id"]
        .unique()
    )

    print(
        "\nSample IDs:"
    )

    print(
        sample_ids
    )

    # --------------------------------------------------------
    # Samples by region
    # --------------------------------------------------------

    print(
        "\nUnique samples by region:"
    )

    samples_by_region = (
        df
        .groupby("region")[
            "sample_id"
        ]
        .nunique()
    )

    print(
        samples_by_region.to_string()
    )

    # --------------------------------------------------------
    # Samples by region + severity
    # --------------------------------------------------------

    print(
        "\nUnique samples by region and severity:"
    )

    sample_severity = (
        df
        .groupby(
            [
                "region",
                "severity"
            ]
        )["sample_id"]
        .nunique()
        .unstack(
            fill_value=0
        )
    )

    sample_severity = (
        sample_severity
        .reindex(
            columns=CLASSES,
            fill_value=0
        )
    )

    print(
        sample_severity.to_string()
    )

    # --------------------------------------------------------
    # Augmentation distribution
    # --------------------------------------------------------

    print(
        "\nAugmentation distribution:"
    )

    augmentation_counts = (
        df["augmentation"]
        .value_counts()
    )

    for augmentation, count in (
        augmentation_counts.items()
    ):

        print(
            f"  {augmentation:<20} "
            f"{count:>6,}"
        )

    # --------------------------------------------------------
    # File extensions
    # --------------------------------------------------------

    print(
        "\nFile extensions:"
    )

    extension_counts = (
        df["extension"]
        .value_counts()
    )

    for extension, count in (
        extension_counts.items()
    ):

        print(
            f"  {extension:<10} "
            f"{count:>6,}"
        )

    print()
    print("=" * 70)


# ============================================================
# CHECK SAMPLE / SEVERITY CONSISTENCY
# ============================================================

def check_consistency(df):
    """
    Check whether each region/sample combination has a single
    severity label.
    """

    print()
    print("=" * 70)
    print("SAMPLE CONSISTENCY CHECK")
    print("=" * 70)

    grouped = (
        df
        .groupby(
            [
                "region",
                "sample_id"
            ]
        )["severity"]
        .nunique()
    )

    inconsistent = grouped[
        grouped > 1
    ]

    if inconsistent.empty:

        print(
            "\n✓ All region/sample combinations "
            "have consistent severity labels."
        )

    else:

        print(
            "\n⚠ Inconsistent labels detected:"
        )

        print(
            inconsistent.to_string()
        )

    print()
    print("=" * 70)


# ============================================================
# FIND MISSING SEVERITY CLASSES
# ============================================================

def check_class_coverage(df):
    """
    Check whether every region contains every severity class.
    """

    print()
    print("=" * 70)
    print("CLASS COVERAGE CHECK")
    print("=" * 70)

    coverage = (
        df
        .groupby(
            [
                "region",
                "severity"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    coverage = coverage.reindex(
        columns=CLASSES,
        fill_value=0
    )

    print()

    for region in coverage.index:

        print(
            f"{region.upper()}:"
        )

        for severity in CLASSES:

            count = coverage.loc[
                region,
                severity
            ]

            if count > 0:

                print(
                    f"  ✓ {severity:<20}"
                    f"{count:>6,} images"
                )

            else:

                print(
                    f"  ✗ {severity:<20}"
                    f"0 images"
                )

        print()


# ============================================================
# SAVE AUDIT
# ============================================================

def save_audit(df):
    """
    Save complete metadata to CSV.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR
        / "dataset_audit.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nAudit CSV saved to:"
    )

    print(
        output_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("FACIAL PALSY DATASET AUDIT")
    print("=" * 70)

    print(
        "\nDataset audit started..."
    )

    # --------------------------------------------------------
    # Build metadata
    # --------------------------------------------------------

    df = build_audit_dataframe()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        df
    )

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    check_consistency(
        df
    )

    # --------------------------------------------------------
    # Class coverage
    # --------------------------------------------------------

    check_class_coverage(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_audit(
        df
    )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)