from pathlib import Path
import pandas as pd

from config import REGIONS


METADATA = Path("outputs/dataset_metadata.csv")

df = pd.read_csv(METADATA)

TARGET_IDS = [9, 14]


print("=" * 80)
print("INVESTIGATING AMBIGUOUS SAMPLE IDs")
print("=" * 80)


for region in ["eye", "eyebrow", "mouth"]:

    print()
    print("=" * 80)
    print(region.upper())
    print("=" * 80)

    region_df = df[
        (df["region"] == region) &
        (df["sample_id"].isin(TARGET_IDS))
    ]

    if region_df.empty:

        print("No records found.")
        continue

    for sample_id in TARGET_IDS:

        sample_df = region_df[
            region_df["sample_id"] == sample_id
        ]

        if sample_df.empty:
            continue

        print()
        print("-" * 80)
        print(f"SAMPLE ID: {sample_id}")
        print("-" * 80)

        print(
            "\nSeverity distribution:"
        )

        print(
            sample_df["severity"]
            .value_counts()
            .to_string()
        )

        print(
            "\nAugmentation distribution:"
        )

        print(
            sample_df["augmentation"]
            .value_counts()
            .to_string()
        )

        print(
            "\nExample filenames:"
        )

        for filename in (
            sample_df["filename"]
            .head(30)
        ):

            print(
                " ",
                filename
            )