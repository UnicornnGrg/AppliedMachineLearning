"""
MASTER DATA PREPARATION CELL
============================

Goal:
  - Prepare ACS 2023 PUMS person-level data (psam_p36.csv) for health insurance
    prediction in a way that is:
      * Aligned with course lectures & exercises (Data wrangling, Insurance, Fairness)
      * Explicit about leakage, sensitive features, and engineered features
      * Flexible: you can later choose different feature subsets and encodings

What this cell does:
  1) Load raw PUMS data (person records).
  2) Restrict to a single state (already done in file, but kept flexible).
  3) Restrict to working-age adults (18–64).
  4) Recode target HICOV -> HICOV_bin (1=insured, 0=uninsured).
  5) Engineer multiple feature families:
       - Demographics: Age (continuous and binned), Sex, MaritalStatus
       - Education: EducationBucket
       - Income & poverty: Income variants + PovBand
       - Employment: ESR_group
       - Occupation & Industry: OccGroup, IndGroup (+ NA categories)
       - Ethnicity: EthnicitySimple
       - Optional leakage flags: PRIVCOV, PUBCOV, HINS1-7 (marked as LEAKAGE)
       - Optional sensitive attributes: Sex, Ethnicity
  6) Define:
       - One "clean baseline" feature set (no occupation, no sensitive)
       - One "extended" feature set including OccGroup
       - One "maximal" feature set including OccGroup + IndGroup (+ numeric income)
  7) Provide a small helper function `build_feature_df(...)` so you can easily
     experiment with different combinations later.

IMPORTANT:
  - This cell does NOT train any models and does NOT compute any metrics.
  - It only prepares data and documents the choices.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------
# 0. CONFIG
# ---------------------------

CSV_PATH = "/workspaces/AppliedMachineLearning/data/raw/psam_p38.csv"   # adjust if needed

AGE_MIN = 18
AGE_MAX = 64

# ---------------------------
# 1. LOAD RAW DATA
# ---------------------------

# Low_memory=False to avoid dtype guessing issues
df_raw = pd.read_csv(CSV_PATH, low_memory=False)

# Optional: if you ever have multiple states, you can filter here, e.g.:
# df_raw = df_raw[df_raw["STATE"] == 36].copy()

# ---------------------------
# 2. BASIC TARGET & CORE COLUMNS
# ---------------------------

# We keep a reasonably rich subset of columns that are relevant or might be useful.
# This is "more than needed", but everything beyond modeling is clearly marked.

core_cols = [
    # ID-like / structure
    "SERIALNO", "SPORDER", "PWGTP",   # person weights (PWGTP) may be useful for weighted stats later

    # Target
    "HICOV",                          # 1 = with coverage, 2 = without

    # Coverage *components* (LEAKAGE if used in prediction)
    "HINS1", "HINS2", "HINS3", "HINS4", "HINS5", "HINS6", "HINS7",
    "PRIVCOV", "PUBCOV",

    # Demographic basics
    "AGEP", "SEX", "MAR",

    # Education
    "SCHL", "SCH",                    # SCH = current enrollment, SCHL = highest degree

    # Employment / labor
    "ESR",                            # Employment status recode

    # Income and poverty
    "PINCP", "WAGP", "POVPIP",

    # Occupation / Industry
    "OCCP", "INDP",

    # Disability & difficulties (sensitive, optional)
    "DIS", "DEAR", "DEYE",

    # Ethnicity / Race / Citizenship (for fairness & sensitive attributes)
    "HISP", "RAC1P", "CIT", "NATIVITY",

    # Geography (we keep for reference but do not use in modeling by default)
    "STATE", "PUMA",
]

# Some columns may not exist in every extract; filter gracefully:
available = [c for c in core_cols if c in df_raw.columns]
df = df_raw[available].copy()

# ---------------------------
# 3. RESTRICT TO WORKING-AGE ADULTS (18–64)
# ---------------------------

df = df[(df["AGEP"] >= AGE_MIN) & (df["AGEP"] <= AGE_MAX)].copy()

# ---------------------------
# 4. TARGET RECODE: HICOV -> HICOV_bin
# ---------------------------
# Course & exercise convention:
#   HICOV: 1 = with coverage, 2 = without
#   HICOV_bin: 1 = insured, 0 = uninsured

df["HICOV_bin"] = df["HICOV"].map({1: 1, 2: 0}).astype("Int64")

# ---------------------------
# 5. DEMOGRAPHIC FEATURES
# ---------------------------

# 5.1 Age: continuous + binned variants
df["Age"] = df["AGEP"].astype("Int64")

# AgeBand5: 18–25, 26–35, 36–45, 46–55, 56–64 (as used in our diagnostics)
df["AgeBand5"] = pd.cut(
    df["Age"],
    bins=[18, 25, 35, 45, 55, 64],
    labels=["18–25", "26–35", "36–45", "46–55", "56–64"],
    include_lowest=True
).astype("category")

# Alternative age band (if you want coarser grouping later)
df["AgeBand3"] = pd.cut(
    df["Age"],
    bins=[18, 35, 55, 64],
    labels=["18–34", "35–54", "55–64"],
    include_lowest=True
).astype("category")

# 5.2 Sex (also used as sensitive attribute)
sex_map = {1: "Male", 2: "Female"}
df["Sex"] = df["SEX"].map(sex_map).astype("category")

# 5.3 Marital status
mar_map = {
    1: "Married",
    2: "Widowed",
    3: "Divorced",
    4: "Separated",
    5: "Never married"
}
df["MaritalStatus"] = df["MAR"].map(mar_map).astype("category")

# ---------------------------
# 6. EDUCATION FEATURES
# ---------------------------

# SCHL is coded 1..24 (ACS)
# Course-consistent bucket:
def edu_bucket(schl):
    """
    Map detailed SCHL codes to coarser education groups.
    This follows the logic used in the AML exercises:
      ≤HS, HS/GED, Some college, Associate, Bachelor, Graduate+.
    """
    if pd.isna(schl):
        return pd.NA
    x = int(schl)
    if x <= 15:                  # Less than HS, some HS
        return "≤HS"
    if x in (16, 17):            # HS diploma / GED
        return "HS/GED"
    if x in (18, 19):            # Some college, no degree
        return "Some college"
    if x == 20:                  # Associate's degree
        return "Associate"
    if x == 21:                  # Bachelor's degree
        return "Bachelor"
    # 22–24: Master's, professional, doctorate
    return "Graduate+"

df["EducationBucket"] = df["SCHL"].apply(edu_bucket).astype("category")

# (Optionally keep SCHL raw as ordinal integer)
df["SCHL_int"] = df["SCHL"].astype("Int64")

# ---------------------------
# 7. INCOME / POVERTY FEATURES
# ---------------------------

# Raw personal income (PINCP) and wage income (WAGP)
df["Income_PINCP"] = df["PINCP"] if "PINCP" in df.columns else pd.NA
df["Income_WAGP"] = df["WAGP"] if "WAGP" in df.columns else pd.NA

# Robust log transform (clip negatives at 0)
df["Income_PINCP_log1p"] = np.log1p(df["Income_PINCP"].clip(lower=0))
df["Income_WAGP_log1p"]  = np.log1p(df["Income_WAGP"].clip(lower=0))

# Poverty ratio POVPIP: ratio to poverty threshold
# Course-aligned poverty bands:
df["PovBand"] = pd.cut(
    df["POVPIP"],
    bins=[-0.1, 100, 200, 400, 1000],
    labels=["<100%", "100–199%", "200–399%", "400%+"]
).astype("object")

# Missing POVPIP is *informative* (mostly young adults / some not-in-labor-force):
df.loc[df["PovBand"].isna(), "PovBand"] = "POV_unknown"
df["PovBand"] = df["PovBand"].astype("category")

# Alternative: continuous poverty ratio (for later experiments)
df["PovRatio"] = df["POVPIP"].astype("Float64")  # may contain NaNs

# ---------------------------
# 8. EMPLOYMENT FEATURES (ESR)
# ---------------------------

# ESR (Employment status recode). Map to 4 groups:
#   1,2 -> Employed
#   3   -> Unemployed
#   6   -> Not in labor force
#   4,5 -> Armed forces
def esr_group(esr):
    if pd.isna(esr):
        return pd.NA
    x = int(esr)
    if x in (1, 2):
        return "Employed"
    if x == 3:
        return "Unemployed"
    if x == 6:
        return "NotLF"
    if x in (4, 5):
        return "Armed"
    return "Other"

df["ESR_group"] = df["ESR"].apply(esr_group).astype("category")

# Optional: keep ESR raw
df["ESR_int"] = df["ESR"].astype("Int64")

# ---------------------------
# 9. OCCUPATION & INDUSTRY FEATURES (OCCP, INDP)
# ---------------------------

# The ACS occupation / industry codes are 4-digit numeric codes.
# We aggregate them to "2-digit" groups (first two digits) and create categories.
# Missing OCCP/INDP is not random: for NotLF, codes are NA by design -> we keep a special "-NA" group.

def occ_ind_two_digit(val):
    if pd.isna(val):
        return pd.NA
    try:
        return int(val) // 100
    except Exception:
        return pd.NA

# 9.1 Occupation
df["Occ2d"] = df["OCCP"].apply(occ_ind_two_digit).astype("Int64")
df["OccGroup"] = df["Occ2d"].apply(
    lambda x: f"Occ{int(x):02d}" if pd.notna(x) else "Occ-NA"
).astype("category")

# 9.2 Industry
df["Ind2d"] = df["INDP"].apply(occ_ind_two_digit).astype("Int64")
df["IndGroup"] = df["Ind2d"].apply(
    lambda x: f"Ind{int(x):02d}" if pd.notna(x) else "Ind-NA"
).astype("category")

# Note:
#   - "Occ-NA" and "Ind-NA" are structurally missing codes (mostly NotLF).
#   - Use them as proper categories; do NOT impute with random values.

# ---------------------------
# 10. ETHNICITY & RACE FEATURES
# ---------------------------

# HISP: 1 = Not Hispanic, >1 = Hispanic (various sub-codes)
# RAC1P: 1 White, 2 Black, 6 Asian, others = various groups
def eth_simple(hisp, rac1p):
    """
    Reduce detailed HISP + RAC1P to a simple, mutually exclusive set:
      - Hispanic
      - NH White
      - NH Black
      - NH Asian
      - NH Other
    Used for fairness analysis and as optional predictive feature.
    """
    if pd.isna(hisp) or pd.isna(rac1p):
        return pd.NA
    if int(hisp) != 1:
        return "Hispanic"
    r = int(rac1p)
    if r == 1:
        return "NH White"
    if r == 2:
        return "NH Black"
    if r == 6:
        return "NH Asian"
    return "NH Other"

df["EthnicitySimple"] = df.apply(lambda row: eth_simple(row.get("HISP", pd.NA),
                                                        row.get("RAC1P", pd.NA)),
                                 axis=1).astype("category")

# ---------------------------
# 11. DISABILITY (OPTIONAL / SENSITIVE)
# ---------------------------
# These are potentially sensitive. We keep them as engineered booleans for
# descriptive / fairness analysis, but **do not include** them in default feature sets.

def bool_from_acs_12(x):
    # 1 = Yes, 2 = No in many ACS binary questions
    if pd.isna(x):
        return pd.NA
    return True if int(x) == 1 else False

df["Disability"]  = df["DIS"].apply(bool_from_acs_12)
df["HearingDiff"] = df["DEAR"].apply(bool_from_acs_12)
df["VisionDiff"]  = df["DEYE"].apply(bool_from_acs_12)

# ---------------------------
# 12. LEAKAGE FEATURES (do NOT use for prediction)
# ---------------------------

# Private/public coverage flags and HINS1–HINS7 are *definitions* of HICOV.
# They are kept only so that you can inspect them, but never used in X.
leakage_cols = []
for col in ["PRIVCOV", "PUBCOV", "HINS1", "HINS2", "HINS3", "HINS4", "HINS5", "HINS6", "HINS7"]:
    if col in df.columns:
        leakage_cols.append(col)

# You can check df[leakage_cols] later if you like.

# ---------------------------
# 13. SENSITIVE FEATURES (for FAIRNESS, not necessarily for prediction)
# ---------------------------

sensitive_cols = ["Sex", "EthnicitySimple"]

# ---------------------------
# 14. DEFINE FEATURE BLOCKS (for flexible modeling later)
# ---------------------------

# These blocks are logical groups of columns you can mix & match.
feature_blocks = {
    "demographic_basic": [
        "AgeBand5",
        "MaritalStatus",
    ],
    "demographic_extra": [
        "Age",           # continuous age
        "AgeBand3",      # alternative coarser band
    ],
    "education": [
        "EducationBucket",
        "SCHL_int",
    ],
    "poverty": [
        "PovBand",
        "PovRatio",
        "Income_PINCP_log1p",
        "Income_WAGP_log1p",
    ],
    "employment": [
        "ESR_group",
        "ESR_int",
    ],
    "occupation": [
        "OccGroup",
        "Occ2d",
    ],
    "industry": [
        "IndGroup",
        "Ind2d",
    ],
    "sensitive": sensitive_cols,
    "disability_sensitive": [
        "Disability",
        "HearingDiff",
        "VisionDiff",
    ],
    "leakage": leakage_cols,
    "target": ["HICOV_bin"],
}

# ---------------------------
# 15. HELPER: BUILD A FEATURE DATAFRAME FROM BLOCKS
# ---------------------------

def build_feature_df(
    df_source: pd.DataFrame,
    blocks_to_use,
    include_sensitive: bool = False,
    include_leakage: bool = False,
    drop_raw_ids: bool = True,
) -> pd.DataFrame:
    """
    Build a modeling-ready DataFrame X + y from df_source
    by selecting feature blocks.

    Parameters
    ----------
    df_source : pd.DataFrame
        The full prepared DataFrame (df in this cell).
    blocks_to_use : list of str
        Names of feature_blocks keys to include, e.g.:
            ["demographic_basic", "education", "poverty", "employment"]
    include_sensitive : bool
        If True, also include "sensitive" block (Sex, EthnicitySimple).
    include_leakage : bool
        If True, also include leakage columns (NOT recommended for real models).
    drop_raw_ids : bool
        If True, drop SERIALNO/SPORDER/PWGTP if present.

    Returns
    -------
    df_out : pd.DataFrame
        DataFrame containing selected feature columns and target HICOV_bin.
    """
    cols = []

    for block in blocks_to_use:
        if block not in feature_blocks:
            raise ValueError(f"Unknown feature block: {block}")
        cols.extend(feature_blocks[block])

    if include_sensitive:
        cols.extend(feature_blocks["sensitive"])

    if include_leakage:
        cols.extend(feature_blocks["leakage"])

    # Always ensure target present:
    cols.extend(feature_blocks["target"])

    # Deduplicate while preserving order:
    seen = set()
    ordered_cols = []
    for c in cols:
        if c not in seen and c in df_source.columns:
            seen.add(c)
            ordered_cols.append(c)

    df_out = df_source[ordered_cols].copy()

    if drop_raw_ids:
        for id_col in ["SERIALNO", "SPORDER", "PWGTP"]:
            if id_col in df_out.columns:
                df_out = df_out.drop(columns=id_col)

    return df_out

# ---------------------------
# 16. EXAMPLE PRE-BUILT DATAFRAMES (NO MODELS, JUST PREP)
# ---------------------------

# 16.1 Clean baseline (course-like): demographics + education + poverty + employment
baseline_blocks = [
    "demographic_basic",   # AgeBand5, MaritalStatus
    "education",
    "poverty",
    "employment",
]
df_baseline = build_feature_df(
    df_source=df,
    blocks_to_use=baseline_blocks,
    include_sensitive=False,   # we keep Sex & Ethnicity aside for fairness
    include_leakage=False,
)

# 16.2 Extended baseline including occupation (as in our later diagnostics)
ext_blocks_occ = baseline_blocks + ["occupation"]
df_ext_occ = build_feature_df(
    df_source=df,
    blocks_to_use=ext_blocks_occ,
    include_sensitive=False,
    include_leakage=False,
)

# 16.3 Maximal experimental feature set: add industry + sensitive attributes
max_blocks = baseline_blocks + ["occupation", "industry"]
df_maximal = build_feature_df(
    df_source=df,
    blocks_to_use=max_blocks,
    include_sensitive=True,    # include Sex & EthnicitySimple
    include_leakage=False,     # keep leakage off by default
)

# At this point you have:
#   - df         : full prepared working-age ACS person data with many engineered columns
#   - df_baseline: "clean", course-aligned baseline feature set
#   - df_ext_occ : baseline + OccGroup (stronger but more complex)
#   - df_maximal : extended set including OccGroup, IndGroup, sensitive attributes
#
# You can now:
#   - Choose one of these for modeling
#   - Or call `build_feature_df(...)` with any combination of blocks
#   - And then perform train/test split, encoding, models, fairness analysis, etc., in later cells.

print("Data preparation complete.")
print("df shape:", df.shape)
print("df_baseline shape:", df_baseline.shape)
print("df_ext_occ shape:", df_ext_occ.shape)
print("df_maximal shape:", df_maximal.shape)