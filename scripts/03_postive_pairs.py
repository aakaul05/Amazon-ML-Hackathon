import pandas as pd
from pathlib import Path


# ============================================================
# Paths
# ============================================================

TRAIN_DIR = Path("data/student_resource/dataset/train")


# ============================================================
# Load only required columns
# ============================================================

print("Loading data...")

s1 = pd.read_csv(
    TRAIN_DIR / "train_source1.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ],
)

s2 = pd.read_csv(
    TRAIN_DIR / "train_source2.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ],
)

s3 = pd.read_csv(
    TRAIN_DIR / "train_source3.tsv",
    sep="\t",
    usecols=[
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ],
)

ground_truth = pd.read_csv(
    TRAIN_DIR / "train_ground_truth.tsv",
    sep="\t",
    usecols=[
        "source1_entity_id",
        "matched_entity_ids",
    ],
)


# ============================================================
# Prepare Source 1
# ============================================================

s1 = s1.rename(
    columns={
        "entity_id": "s1_id",
        "business_name": "s1_name",
        "business_address": "s1_address",
        "country": "s1_country",
    }
)

s1 = s1.set_index("s1_id")


# ============================================================
# Prepare ground truth
# ============================================================

print("Preparing ground truth...")

gt = ground_truth.rename(
    columns={
        "source1_entity_id": "s1_id"
    }
)

# Missing means zero matches
gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .fillna("")
)

# Convert:
#
# S2-123,S2-456,S3-789
#
# into a Python list inside each row
gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .str.split(",")
)

# Turn each matched ID into its own row
gt = gt.explode(
    "matched_entity_ids",
    ignore_index=True
)

# Remove empty matches
gt["matched_entity_ids"] = (
    gt["matched_entity_ids"]
    .str.strip()
)

gt = gt[
    gt["matched_entity_ids"] != ""
].copy()

print(
    f"Positive pairs: {len(gt):,}"
)


# ============================================================
# Analyze one source at a time
# ============================================================

def analyze_source(
    gt,
    s1,
    source_df,
    source_name,
):

    print("\n" + "=" * 70)
    print(f"ANALYZING {source_name} POSITIVE PAIRS")
    print("=" * 70)

    # --------------------------------------------------------
    # Select only matches belonging to this source
    # --------------------------------------------------------

    prefix = source_name + "-"

    pairs = gt[
        gt["matched_entity_ids"].str.startswith(prefix)
    ].copy()

    print(
        f"Positive {source_name} pairs: "
        f"{len(pairs):,}"
    )

    # Rename target ID
    pairs = pairs.rename(
        columns={
            "matched_entity_ids": "other_id"
        }
    )

    # --------------------------------------------------------
    # Load source columns
    # --------------------------------------------------------

    other = source_df.rename(
        columns={
            "entity_id": "other_id",
            "business_name": "other_name",
            "business_address": "other_address",
            "country": "other_country",
        }
    )

    other = other.set_index("other_id")

    # --------------------------------------------------------
    # Join S1 information
    # --------------------------------------------------------

    pairs = pairs.join(
        s1[
            [
                "s1_name",
                "s1_address",
                "s1_country",
            ]
        ],
        on="s1_id",
        how="left",
    )

    # --------------------------------------------------------
    # Join target-source information
    # --------------------------------------------------------

    pairs = pairs.join(
        other[
            [
                "other_name",
                "other_address",
                "other_country",
            ]
        ],
        on="other_id",
        how="left",
    )

    # --------------------------------------------------------
    # Exact comparisons
    # --------------------------------------------------------

    pairs["name_exact"] = (
        pairs["s1_name"]
        == pairs["other_name"]
    )

    pairs["address_exact"] = (
        pairs["s1_address"]
        == pairs["other_address"]
    )

    pairs["country_exact"] = (
        pairs["s1_country"]
        == pairs["other_country"]
    )

    # --------------------------------------------------------
    # Missingness
    # --------------------------------------------------------

    pairs["other_name_missing"] = (
        pairs["other_name"].isna()
    )

    pairs["other_address_missing"] = (
        pairs["other_address"].isna()
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\nExact signals among TRUE matches")
    print("-" * 50)

    print(
        f"Name exact: "
        f"{pairs['name_exact'].mean() * 100:.2f}%"
    )

    print(
        f"Address exact: "
        f"{pairs['address_exact'].mean() * 100:.2f}%"
    )

    print(
        f"Country exact: "
        f"{pairs['country_exact'].mean() * 100:.2f}%"
    )

    print("\nMissing values among TRUE matches")
    print("-" * 50)

    print(
        f"Name missing: "
        f"{pairs['other_name_missing'].mean() * 100:.4f}%"
    )

    print(
        f"Address missing: "
        f"{pairs['other_address_missing'].mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Save compact analysis result
    # --------------------------------------------------------

    output_columns = [
        "s1_id",
        "other_id",
        "s1_name",
        "other_name",
        "s1_address",
        "other_address",
        "s1_country",
        "other_country",
        "name_exact",
        "address_exact",
        "country_exact",
    ]

    output = pairs[output_columns]

    output_path = (
        TRAIN_DIR.parent.parent
        / "outputs"
        / "eda"
        / f"positive_pairs_{source_name.lower()}.parquet"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nSaved: {output_path}"
    )

    return pairs


# ============================================================
# Analyze S2
# ============================================================

s2_pairs = analyze_source(
    gt=gt,
    s1=s1,
    source_df=s2,
    source_name="S2",
)

del s2_pairs


# ============================================================
# Analyze S3
# ============================================================

s3_pairs = analyze_source(
    gt=gt,
    s1=s1,
    source_df=s3,
    source_name="S3",
)

del s3_pairs


# ============================================================
# Final
# ============================================================

print("\n" + "=" * 70)
print("POSITIVE PAIR ANALYSIS COMPLETE")
print("=" * 70)