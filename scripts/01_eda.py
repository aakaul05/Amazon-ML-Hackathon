import pandas as pd

train_s1 = pd.read_csv(
    "data/student_resource/dataset/train/train_source1.tsv",
    sep="\t"
)

train_s2 = pd.read_csv(
    "data/student_resource/dataset/train/train_source2.tsv",
    sep="\t"
)

train_s3 = pd.read_csv(
    "data/student_resource/dataset/train/train_source3.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "data/student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t"
)
# ============================================================
# Basic dataset information
# ============================================================

datasets = {
    "Source 1": train_s1,
    "Source 2": train_s2,
    "Source 3": train_s3,
    "Ground Truth": ground_truth,
}


for name, df in datasets.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nDtypes:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isna().sum())


# ============================================================
# Duplicate ID checks
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE ID CHECK")
print("=" * 70)

for name, df in {
    "Source 1": train_s1,
    "Source 2": train_s2,
    "Source 3": train_s3,
}.items():

    duplicate_count = df["entity_id"].duplicated().sum()

    print(
        f"{name}: "
        f"{duplicate_count} duplicate entity_id values"
    )


# ============================================================
# Country distribution
# ============================================================

print("\n" + "=" * 70)
print("COUNTRY DISTRIBUTION")
print("=" * 70)

for name, df in {
    "Source 1": train_s1,
    "Source 2": train_s2,
    "Source 3": train_s3,
}.items():

    print(f"\n{name}")
    print(df["country"].value_counts(dropna=False))


# ============================================================
# Name statistics
# ============================================================

print("\n" + "=" * 70)
print("BUSINESS NAME STATISTICS")
print("=" * 70)

for name, df in {
    "Source 1": train_s1,
    "Source 2": train_s2,
    "Source 3": train_s3,
}.items():

    name_lengths = df["business_name"].fillna("").astype(str).str.len()

    print(f"\n{name}")

    print("Min length :", name_lengths.min())
    print("Mean length:", round(name_lengths.mean(), 2))
    print("Median     :", name_lengths.median())
    print("Max length :", name_lengths.max())


# ============================================================
# Address statistics
# ============================================================

print("\n" + "=" * 70)
print("ADDRESS STATISTICS")
print("=" * 70)

for name, df in {
    "Source 1": train_s1,
    "Source 2": train_s2,
    "Source 3": train_s3,
}.items():

    address_lengths = (
        df["business_address"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    print(f"\n{name}")

    print("Min length :", address_lengths.min())
    print("Mean length:", round(address_lengths.mean(), 2))
    print("Median     :", address_lengths.median())
    print("Max length :", address_lengths.max())


# ============================================================
# Ground truth inspection
# ============================================================

print("\n" + "=" * 70)
print("GROUND TRUTH")
print("=" * 70)

print("\nColumns:")
print(ground_truth.columns.tolist())

print("\nShape:")
print(ground_truth.shape)

print("\nFirst 10 rows:")
print(ground_truth.head(10))


# ============================================================
# Match-count distribution
# ============================================================

# Adjust this column name after inspecting the actual file.
# This section assumes the ground truth has a column containing
# comma-separated matching IDs.

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

# ============================================================
# Ground Truth Match Count Analysis
# ============================================================

print("\n" + "=" * 70)
print("GROUND TRUTH MATCH COUNT ANALYSIS")
print("=" * 70)


def count_matches(value):
    if pd.isna(value):
        return 0

    value = str(value).strip()

    if value == "":
        return 0

    return len(value.split(","))


ground_truth["match_count"] = (
    ground_truth["matched_entity_ids"]
    .apply(count_matches)
)


print("\nMatch count distribution:")
print(
    ground_truth["match_count"]
    .value_counts()
    .sort_index()
)


print("\nMatch count statistics:")

print(
    ground_truth["match_count"]
    .describe()
)


print("\nNumber of singleton S1 entities:")

singleton_count = (
    ground_truth["match_count"] == 0
).sum()

print(singleton_count)


print("\nPercentage of singleton S1 entities:")

singleton_percentage = (
    singleton_count / len(ground_truth)
) * 100

print(f"{singleton_percentage:.2f}%")
# ============================================================
# Ground Truth: S2 vs S3 Match Counts
# ============================================================

print("\n" + "=" * 70)
print("S2 VS S3 MATCH DISTRIBUTION")
print("=" * 70)


def count_source_matches(value, prefix):

    if pd.isna(value):
        return 0

    value = str(value).strip()

    if value == "":
        return 0

    ids = value.split(",")

    return sum(
        entity_id.startswith(prefix)
        for entity_id in ids
    )


ground_truth["s2_match_count"] = (
    ground_truth["matched_entity_ids"]
    .apply(lambda x: count_source_matches(x, "S2-"))
)

ground_truth["s3_match_count"] = (
    ground_truth["matched_entity_ids"]
    .apply(lambda x: count_source_matches(x, "S3-"))
)


print("\nS2 matches per S1:")
print(
    ground_truth["s2_match_count"]
    .value_counts()
    .sort_index()
)


print("\nS3 matches per S1:")
print(
    ground_truth["s3_match_count"]
    .value_counts()
    .sort_index()
)