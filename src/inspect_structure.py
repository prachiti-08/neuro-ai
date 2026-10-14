from pathlib import Path
from collections import Counter

from config import REGIONS


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".png",
}


def inspect_region(region):

    root = REGIONS[region]["folder"]

    print()
    print("=" * 80)
    print(f"{region.upper()} FOLDER STRUCTURE")
    print("=" * 80)

    # ---------------------------------------------------------
    # Exact immediate subfolders
    # ---------------------------------------------------------

    folders = sorted(
        {
            path.name
            for path in root.rglob("*")
            if path.is_dir()
        }
    )

    print("\nALL FOLDERS:")
    print("-" * 80)

    for folder in folders:
        print(repr(folder))

    # ---------------------------------------------------------
    # Count images per folder
    # ---------------------------------------------------------

    print("\nIMAGE COUNT PER LEAF FOLDER:")
    print("-" * 80)

    folder_counts = Counter()

    for path in root.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            folder_counts[
                path.parent.name
            ] += 1

    for folder, count in sorted(
        folder_counts.items()
    ):

        print(
            f"{folder:<30} "
            f"{count:>6}"
        )

    # ---------------------------------------------------------
    # Show filename examples
    # ---------------------------------------------------------

    print("\nFILENAME EXAMPLES:")
    print("-" * 80)

    examples = {}

    for path in root.rglob("*"):

        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            folder = path.parent.name

            if folder not in examples:

                examples[folder] = []

            if len(examples[folder]) < 8:

                examples[folder].append(
                    path.name
                )

    for folder in sorted(examples):

        print(f"\n[{folder}]")

        for filename in examples[folder]:

            print(
                f"  {filename}"
            )


def main():

    print("=" * 80)
    print("FACIAL PALSY DATASET STRUCTURE INSPECTOR")
    print("=" * 80)

    for region in REGIONS:

        inspect_region(
            region
        )

    print()
    print("=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()