import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = Path("data/train")


# ============================================================
# Load data
# ============================================================

s1 = pd.read_csv(
    "data/student_resource/dataset/train/train_source1.tsv",
    sep="\t"
)

s2 = pd.read_csv(
    "data/student_resource/dataset/train/train_source2.tsv",
    sep="\t"
)

s3 = pd.read_csv(
    "data/student_resource/dataset/train/train_source3.tsv",
    sep="\t"
)


# ============================================================
# Helper
# ============================================================

def overlap_stats(source1, source_other, source_name):

    print("\n" + "=" * 70)
    print(f"S1 vs {source_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Raw exact name overlap
    # --------------------------------------------------------

    s1_names = set(
        source1["business_name"]
        .dropna()
        .astype(str)
    )

    other_names = set(
        source_other["business_name"]
        .dropna()
        .astype(str)
    )

    name_overlap = s1_names & other_names

    print(
        f"Unique exact business names shared: "
        f"{len(name_overlap):,}"
    )

    # --------------------------------------------------------
    # Raw exact address overlap
    # --------------------------------------------------------

    s1_addresses = set(
        source1["business_address"]
        .dropna()
        .astype(str)
    )

    other_addresses = set(
        source_other["business_address"]
        .dropna()
        .astype(str)
    )

    address_overlap = s1_addresses & other_addresses

    print(
        f"Unique exact addresses shared: "
        f"{len(address_overlap):,}"
    )

    # --------------------------------------------------------
    # Raw exact name + address overlap
    # --------------------------------------------------------

    s1_pairs = set(
        zip(
            source1["business_name"].fillna("").astype(str),
            source1["business_address"].fillna("").astype(str)
        )
    )

    other_pairs = set(
        zip(
            source_other["business_name"].fillna("").astype(str),
            source_other["business_address"].fillna("").astype(str)
        )
    )

    pair_overlap = s1_pairs & other_pairs

    print(
        f"Unique exact name + address pairs shared: "
        f"{len(pair_overlap):,}"
    )


# ============================================================
# Run analysis
# ============================================================

overlap_stats(s1, s2, "Source 2")
overlap_stats(s1, s3, "Source 3")