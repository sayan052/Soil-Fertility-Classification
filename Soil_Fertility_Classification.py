"""
Soil Fertility Classification Using Machine Learning  (BCA PBL Project)
=======================================================================
Complete source code, extracted in order from Soil_Fertility_Classification.ipynb.

How to run:   1) put Soil.csv in the same folder as this file
              2) pip install numpy pandas matplotlib seaborn scikit-learn scipy joblib
              3) python Soil_Fertility_Classification.py
Outputs:      outputs/figures/*.png, outputs/tables/*.csv, outputs/results_summary.json,
              outputs/final_soil_fertility_model.joblib

DATA NOTE (from the dataset README): Soil.csv is source-calibrated SYNTHETIC teaching data,
NOT 1,000 authentic laboratory observations. Fertility_Class is a DERIVED label built from
Fertility_Score (N, P, K, organic carbon, pH, EC). Fertility_Score is never used as a model input.

Task order: 0 Setup | 1 Understand data | 2 Cleaning | 3 Missing values | 4 EDA | 5 Visualization |
6 Correlation | 7 Feature selection | 8 Train-test split | 9 Four algorithms | 10 Metrics |
11 Confusion matrices | 12 Model comparison | 13 Feature contribution | 14 New-sample prediction
"""
import matplotlib
import os
if os.environ.get("DISPLAY", "") == "" and os.name != "nt" and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")      # no screen available (e.g. server): save figures without opening windows



# ==============================================================================
# 0. Setup – libraries, folders and reproducibility
# ==============================================================================

# We import the libraries, fix a `RANDOM_STATE` so that every run gives identical results, and create folders for the outputs.

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

try:
    from IPython.display import display
except ImportError:          # when the code is run as a plain .py script
    display = print                      # plain script: print tables instead

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)  # readable printed tables
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100

CLASS_ORDER = ["Low Fertility", "Medium Fertility", "High Fertility"]
CLASS_MAP = {c: i for i, c in enumerate(CLASS_ORDER)}      # Low=0, Medium=1, High=2 (ordinal target encoding)
SHORT_CLASS = ["Low", "Medium", "High"]
PALETTE = {"Low Fertility": "#d95f02", "Medium Fertility": "#e6ab02", "High Fertility": "#1b9e77"}

def save_table(df, name, index=True):
    """Save a table as CSV (used later by the PDF report)."""
    df.to_csv(os.path.join(TAB_DIR, name + ".csv"), index=index)

def save_show(name, fig=None):
    """Save the current figure to outputs/figures and display it."""
    fig = fig or plt.gcf()
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=150, bbox_inches="tight")
    plt.show()

print("Libraries loaded. Random state =", RANDOM_STATE)


# ==============================================================================
# TASK 1 – UNDERSTAND THE DATASET
# ==============================================================================

# **What we do:** read the README, load `Soil.csv`, and look at its size, columns, data types and basic statistics.
#
# **README summary**
# | Item | Meaning |
# |---|---|
# | Status | Source-calibrated **synthetic** teaching data (not copied/authentic lab rows) |
# | Authentic basis | AgriNet (Bareilly, UP – 258 records), Krishnagiri (125 samples), Dharmapuri (125 samples) |
# | Target | `Fertility_Class` – **derived** from `Fertility_Score` (N, P, K, OC, pH, EC); Low < 40, Medium 40–69.99, High ≥ 70 |
# | Caution | Do not claim these are 1,000 authentic measured samples |

# Locate Soil.csv (works on a local machine, in Jupyter and in Google Colab)
CANDIDATE_PATHS = ["Soil.csv", "data/Soil.csv", "/content/Soil.csv", "/mnt/user-data/uploads/Soil.csv"]
DATA_PATH = next((p for p in CANDIDATE_PATHS if os.path.exists(p)), None)
if DATA_PATH is None:
    DATA_PATH = "Soil.csv"

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            "Soil.csv not found. Please put Soil.csv "
            "in the same folder as this Python script."
        )

raw = pd.read_csv(DATA_PATH)
print("Loaded:", DATA_PATH)
print("Dataset dimensions (rows, columns):", raw.shape)
display(raw.head())

# ### 1.1 Meaning of each attribute (data dictionary)
# The units below follow the column names in the file; the descriptions are standard soil-science meanings.

data_dictionary = pd.DataFrame([
    ("Sample_ID", "Identifier", "Unique sample code (SFC-0001 …)"),
    ("Source_Calibration", "Provenance", "Which authentic dataset group the synthetic row was calibrated from"),
    ("State", "Categorical", "Indian state of the (synthetic) sample"),
    ("District", "Categorical", "District inside the state"),
    ("Soil_Type", "Categorical", "Soil type: Alluvial, Black, Red, Loamy, Clay Loam, Sandy Loam, Lateritic"),
    ("Crop", "Categorical", "Crop grown on the field"),
    ("pH", "Numeric (soil reaction)", "Acidity/alkalinity of soil (7 = neutral); affects nutrient availability"),
    ("Electrical_Conductivity_dS_m", "Numeric (salinity)", "EC in dS/m – measures soluble salts; high EC = saline soil"),
    ("Organic_Carbon_g_kg", "Numeric (organic matter)", "Organic carbon in g/kg – indicates humus and soil health"),
    ("Available_N_kg_ha", "Numeric (macronutrient)", "Plant-available Nitrogen in kg/ha"),
    ("Available_P_kg_ha", "Numeric (macronutrient)", "Plant-available Phosphorus in kg/ha"),
    ("Available_K_kg_ha", "Numeric (macronutrient)", "Plant-available Potassium in kg/ha"),
    ("Sulphur_mg_kg", "Numeric (secondary nutrient)", "Available Sulphur in mg/kg"),
    ("Zinc_mg_kg", "Numeric (micronutrient)", "Available Zinc in mg/kg"),
    ("Boron_mg_kg", "Numeric (micronutrient)", "Available Boron in mg/kg"),
    ("Iron_mg_kg", "Numeric (micronutrient)", "Available Iron in mg/kg"),
    ("Manganese_mg_kg", "Numeric (micronutrient)", "Available Manganese in mg/kg"),
    ("Copper_mg_kg", "Numeric (micronutrient)", "Available Copper in mg/kg"),
    ("Fertility_Score", "Derived score (NOT a feature)", "0–100 score built from N, P, K, OC, pH, EC; used to create the label"),
    ("Fertility_Class", "TARGET (derived label)", "Low (<40), Medium (40–69.99), High (≥70) Fertility"),
], columns=["Column", "Type / Role", "Meaning"])
display(data_dictionary)
save_table(data_dictionary, "01_data_dictionary", index=False)

print("Dimensions:", raw.shape)
print("\nColumn names:\n", list(raw.columns))
dtypes_table = raw.dtypes.astype(str).rename("Data type").to_frame()
dtypes_table["Non-null count"] = raw.notna().sum()
dtypes_table["Unique values"] = raw.nunique()
display(dtypes_table)
save_table(dtypes_table, "01_dtypes")

print("\nBasic statistics – numeric columns")
num_summary = raw.describe().T
display(num_summary.round(3))
save_table(num_summary.round(3), "01_numeric_summary")

print("\nBasic statistics – text/categorical columns")
cat_summary = raw.describe(exclude="number").T
display(cat_summary)

# **Reading the output:** the data has 1,000 rows and 20 columns: 1 identifier, 1 provenance column, 4 categorical columns,
# 12 numeric soil measurements, the derived `Fertility_Score` and the target `Fertility_Class`.


# ==============================================================================
# TASK 2 – DATA CLEANING
# ==============================================================================

# **What we do:** look for problems (duplicates, spelling/space inconsistencies, impossible values, wrong labels) and fix what needs fixing.
# We never modify the original `raw` table – we work on a copy called `df_clean`.

NUM_COLS = ["pH", "Electrical_Conductivity_dS_m", "Organic_Carbon_g_kg", "Available_N_kg_ha",
            "Available_P_kg_ha", "Available_K_kg_ha", "Sulphur_mg_kg", "Zinc_mg_kg", "Boron_mg_kg",
            "Iron_mg_kg", "Manganese_mg_kg", "Copper_mg_kg"]
CAT_COLS = ["State", "District", "Soil_Type", "Crop"]
TEXT_COLS = raw.select_dtypes(exclude="number").columns.tolist()

# --- (a) duplicates
dup_all = int(raw.duplicated().sum())
dup_no_id = int(raw.drop(columns="Sample_ID").duplicated().sum())
dup_id = int(raw["Sample_ID"].duplicated().sum())

# --- (b) whitespace problems in text columns
ws_issues = int(sum((raw[c].astype(str) != raw[c].astype(str).str.strip()).sum() for c in TEXT_COLS))

# --- (c) impossible values
ph_bad = int(((raw["pH"] < 0) | (raw["pH"] > 14)).sum())
negatives = int((raw[NUM_COLS] < 0).sum().sum())

# --- (d) does the stored class match the README thresholds?
def score_to_class(score):
    return np.select([score < 40, score < 70], ["Low Fertility", "Medium Fertility"], default="High Fertility")
label_mismatch = int((score_to_class(raw["Fertility_Score"]) != raw["Fertility_Class"]).sum())

# --- (e) does each district belong to exactly one state?
district_conflicts = int((raw.groupby("District")["State"].nunique() > 1).sum())

# --- (f) values piled up at the minimum or maximum of a column (possible clipping by the data generator)
boundary_rows = []
for c in NUM_COLS:
    n_min = int((raw[c] == raw[c].min()).sum()); n_max = int((raw[c] == raw[c].max()).sum())
    boundary_rows.append((c, raw[c].min(), n_min, raw[c].max(), n_max, "Yes" if max(n_min, n_max) >= 10 else "No"))
boundary = pd.DataFrame(boundary_rows, columns=["Feature", "Minimum", "Rows at minimum", "Maximum", "Rows at maximum", "Clipping suspected (>=10 rows)?"])
clipped_cols = boundary.loc[boundary["Clipping suspected (>=10 rows)?"] == "Yes", "Feature"].tolist()
clipped_rows = int(((raw[clipped_cols] == raw[clipped_cols].min()) | (raw[clipped_cols] == raw[clipped_cols].max())).any(axis=1).sum())

checks = pd.DataFrame([
    ("Duplicate rows (all columns)", dup_all, "None found – nothing to remove"),
    ("Duplicate rows ignoring Sample_ID", dup_no_id, "None found – no hidden duplicates"),
    ("Duplicate Sample_ID values", dup_id, "None found – IDs are unique"),
    ("Leading/trailing spaces in text columns", ws_issues, "None found; .strip() still applied as a safeguard"),
    ("pH outside 0–14", ph_bad, "None found"),
    ("Negative numeric values", negatives, "None found"),
    ("Fertility_Class not matching README thresholds", label_mismatch, "None found – label is consistent with the score"),
    ("Districts mapped to more than one State", district_conflicts, "None found"),
    ("Columns with many rows stuck at their min/max value (possible clipping)", len(clipped_cols),
     f"Found in: {', '.join(clipped_cols)} – affects {clipped_rows} rows; kept and reported (see table below)"),
], columns=["Check", "Result (count)", "Action"])
display(checks)
save_table(checks, "02_cleaning_checks", index=False)

print("\nRepeated minimum / maximum values per feature:")
display(boundary)
save_table(boundary, "02_boundary_values", index=False)

# ### Outlier check (IQR rule)
# A value is flagged as an *outlier* if it lies more than 1.5 × IQR below Q1 or above Q3. In soil data, high readings are
# normal natural variation (for example, a well-manured field), so we **flag but keep** them. Removing them would erase real
# *High Fertility* examples.

rows = []
for c in NUM_COLS:
    q1, q3 = raw[c].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((raw[c] < lo) | (raw[c] > hi)).sum())
    rows.append((c, round(lo, 2), round(hi, 2), n_out, round(100 * n_out / len(raw), 1)))
outlier_table = pd.DataFrame(rows, columns=["Feature", "Lower fence", "Upper fence", "Outliers", "% of rows"])
display(outlier_table)
save_table(outlier_table, "02_outliers_iqr", index=False)

# Apply the (minimal) cleaning steps on a copy of the data
df_clean = raw.copy()
for c in TEXT_COLS:
    df_clean[c] = df_clean[c].astype(str).str.strip()          # remove stray spaces
df_clean = df_clean.drop_duplicates().reset_index(drop=True)   # no duplicates exist, so nothing is dropped
print("Rows before cleaning:", len(raw), "| rows after cleaning:", len(df_clean))
print("Class labels present:", sorted(df_clean["Fertility_Class"].unique()))

# **Cleaning conclusion:** the file is already tidy. No duplicates, spacing problems, impossible values or label errors were found.
# Outliers are real-looking high readings and are retained. One unusual pattern was found: in **pH, Organic Carbon, N, P and K** a noticeable
# number of rows sit exactly at the column's minimum or maximum (for example many K values at the lowest value). This looks like **clipping to a
# fixed range by the synthetic data generator**, which would not be expected in raw laboratory measurements. We keep these rows (they are valid
# values inside the physical range) and mention the pattern in the Limitations.


# ==============================================================================
# TASK 3 – MISSING VALUE HANDLING
# ==============================================================================

# **What we do:** count missing values per column (also checking for hidden placeholders such as `?`, `NA`, `-` and impossible zeros).

missing = df_clean.isna().sum().rename("Missing values").to_frame()
missing["Missing %"] = (100 * missing["Missing values"] / len(df_clean)).round(2)
display(missing)
save_table(missing, "03_missing_values")

placeholders = int(df_clean.isin(["?", "NA", "N/A", "na", "n/a", "-", "null", "NULL", "None", ""]).sum().sum())
zero_counts = (df_clean[NUM_COLS] == 0).sum()
print("Total missing cells:", int(df_clean.isna().sum().sum()))
print("Hidden placeholder strings:", placeholders)
print("Numeric columns containing zeros:", int((zero_counts > 0).sum()))

if df_clean.isna().sum().sum() == 0:
    print("\n✅ RESULT: The dataset contains NO missing values, so no imputation is required.")
else:                                                           # generic safeguard (not triggered for this file)
    df_clean[NUM_COLS] = df_clean[NUM_COLS].fillna(df_clean[NUM_COLS].median())

# **Result:** there are **no missing values**. However, new soil samples entered by a user in Task 14 might have gaps,
# so our modelling pipelines will still contain a `SimpleImputer(strategy="median")` step as a safety net. It learns the
# medians from the **training data only**, so it causes no data leakage.


# ==============================================================================
# TASK 4 – EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

# **What we do:** study numerical and categorical features, the class balance, and how soil properties differ between the three fertility classes.

# ---- 4.1 Numerical features: centre, spread and shape
eda_num = df_clean[NUM_COLS].agg(["mean", "median", "std", "min", "max", "skew"]).T
eda_num["kurtosis"] = df_clean[NUM_COLS].kurt()
eda_num = eda_num.rename(columns={"skew": "skewness"}).round(3)
display(eda_num)
save_table(eda_num, "04_numeric_eda")
print("Most right-skewed features:", ", ".join(eda_num["skewness"].sort_values(ascending=False).head(4).index))

# ---- 4.2 Target class balance
class_counts = df_clean["Fertility_Class"].value_counts().reindex(CLASS_ORDER)
class_table = pd.DataFrame({"Count": class_counts, "Percent": (100 * class_counts / len(df_clean)).round(1)})
display(class_table)
save_table(class_table, "04_class_distribution")
print(f"Imbalance ratio (largest : smallest class) = {class_counts.max() / class_counts.min():.1f} : 1")

# ---- 4.3 Categorical features: categories and how many rows each has
for c in CAT_COLS + ["Source_Calibration"]:
    print(f"\n{c} ({df_clean[c].nunique()} categories)")
    print(df_clean[c].value_counts().to_string())

# ### 4.4 Do categorical features relate to fertility? (Chi-square test of independence)
# *Null hypothesis:* the category and the fertility class are independent. A p-value below 0.05 would suggest a relationship.
# **Cramér's V** (0 = no association, 1 = perfect association) tells us how strong the relationship is.

def cramers_v(ct):
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt((chi2 / n) / (min(ct.shape) - 1))
    return chi2, p, dof, v

rows = []
for c in ["State", "District", "Soil_Type", "Crop", "Source_Calibration"]:
    ct = pd.crosstab(df_clean[c], df_clean["Fertility_Class"])
    chi2, p, dof, v = cramers_v(ct)
    rows.append((c, round(chi2, 2), dof, round(p, 4), round(v, 3), "Yes" if p < 0.05 else "No"))
chi_table = pd.DataFrame(rows, columns=["Categorical feature", "Chi-square", "d.o.f.", "p-value", "Cramér's V", "Significant (p<0.05)?"])
display(chi_table)
save_table(chi_table, "04_chi_square", index=False)

# ---- 4.5 How do soil properties differ between the classes? (class-wise means)
class_means = df_clean.groupby("Fertility_Class")[NUM_COLS].mean().loc[CLASS_ORDER].T.round(2)
display(class_means)
save_table(class_means, "04_class_means")

# **Key observations (read from the tables above)**
# * The target is **imbalanced**: *Medium* is the majority class and *High* is a small minority. This is why we will use *stratified* splitting and report **macro** metrics as well as accuracy.
# * Several nutrients (N, P, K, OC, S, Zn, B, Fe, Mn, Cu) are **right-skewed**: most soils have low/moderate values and a few have very high values.
# * Class-wise means show that **N, P, K and Organic Carbon rise steadily from Low → Medium → High**, while EC is the highest in the *Low* class (salinity reduces fertility).
# * Micronutrients (Zn, B, Fe, Mn, Cu) and Sulphur barely change between classes – they are **not part of the score**.
# * The chi-square table shows how strongly State, Soil_Type, Crop and District are linked to the class.


# ==============================================================================
# TASK 5 – DATA VISUALIZATION
# ==============================================================================

# We create the required graphs: class distribution, histograms, boxplots, categorical plots and a few other relevant plots.

SHORT = {"pH": "pH", "Electrical_Conductivity_dS_m": "EC (dS/m)", "Organic_Carbon_g_kg": "Organic C (g/kg)",
         "Available_N_kg_ha": "N (kg/ha)", "Available_P_kg_ha": "P (kg/ha)", "Available_K_kg_ha": "K (kg/ha)",
         "Sulphur_mg_kg": "Sulphur (mg/kg)", "Zinc_mg_kg": "Zinc (mg/kg)", "Boron_mg_kg": "Boron (mg/kg)",
         "Iron_mg_kg": "Iron (mg/kg)", "Manganese_mg_kg": "Manganese (mg/kg)", "Copper_mg_kg": "Copper (mg/kg)"}

# Figure 1 – Fertility class distribution
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(SHORT_CLASS, class_counts.values, color=[PALETTE[c] for c in CLASS_ORDER], edgecolor="black")
for b, n in zip(bars, class_counts.values):
    ax.text(b.get_x() + b.get_width() / 2, n + 8, f"{n}\n({100 * n / len(df_clean):.1f}%)", ha="center", fontsize=10)
ax.set_ylim(0, class_counts.max() * 1.18)
ax.set_title("Figure 1: Fertility Class Distribution"); ax.set_xlabel("Fertility class"); ax.set_ylabel("Number of samples")
save_show("fig01_class_distribution")

# Figure 2 – The derived Fertility_Score and the README class thresholds (descriptive only; NOT used as a feature)
fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(data=df_clean, x="Fertility_Score", hue="Fertility_Class", hue_order=CLASS_ORDER, palette=PALETTE,
             bins=35, multiple="stack", ax=ax)
for t in (40, 70):
    ax.axvline(t, color="black", linestyle="--")
ax.set_title("Figure 2: Fertility_Score and Class Thresholds (40 and 70)\n(for understanding the label only – not a model input)")
ax.set_xlabel("Fertility_Score")
save_show("fig02_score_distribution")

# Figure 3 – Histograms of all 12 numeric soil properties
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, c in zip(axes.ravel(), NUM_COLS):
    sns.histplot(df_clean[c], kde=True, bins=30, color="#4c78a8", ax=ax)
    ax.set_xlabel(SHORT[c]); ax.set_ylabel("Count")
fig.suptitle("Figure 3: Histograms of Numerical Soil Properties", fontsize=15, y=1.01)
save_show("fig03_histograms")

# Figure 4 – Boxplots of all 12 numeric soil properties (outlier view)
fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for ax, c in zip(axes.ravel(), NUM_COLS):
    sns.boxplot(y=df_clean[c], color="#9ecae1", ax=ax, fliersize=3)
    ax.set_ylabel(SHORT[c])
fig.suptitle("Figure 4: Boxplots of Numerical Soil Properties", fontsize=15, y=1.01)
save_show("fig04_boxplots")

# Figure 5 – Boxplots of the six score-related properties split by fertility class
SIX = ["pH", "Electrical_Conductivity_dS_m", "Organic_Carbon_g_kg", "Available_N_kg_ha", "Available_P_kg_ha", "Available_K_kg_ha"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.ravel(), SIX):
    sns.boxplot(data=df_clean, x="Fertility_Class", y=c, order=CLASS_ORDER, palette=PALETTE, ax=ax, fliersize=3)
    ax.set_xticklabels(SHORT_CLASS); ax.set_xlabel(""); ax.set_ylabel(SHORT[c])
fig.suptitle("Figure 5: Key Soil Properties by Fertility Class", fontsize=15, y=1.01)
save_show("fig05_boxplots_by_class")

# Figure 6 – Categorical features: number of samples per category
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, c in zip(axes, ["State", "Soil_Type", "Crop"]):
    order = df_clean[c].value_counts().index
    sns.countplot(data=df_clean, y=c, order=order, color="#72b7b2", ax=ax)
    ax.set_title(c); ax.set_xlabel("Number of samples"); ax.set_ylabel("")
fig.suptitle("Figure 6: Categorical Features – Sample Counts", fontsize=15, y=1.03)
save_show("fig06_categorical_counts")

# Figure 7 – Fertility class mix inside each category (100% stacked bars)
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, c in zip(axes, ["State", "Soil_Type", "Crop"]):
    prop = pd.crosstab(df_clean[c], df_clean["Fertility_Class"], normalize="index")[CLASS_ORDER]
    prop.plot(kind="barh", stacked=True, color=[PALETTE[k] for k in CLASS_ORDER], ax=ax, legend=False, edgecolor="white")
    ax.set_title(c); ax.set_xlabel("Proportion of samples"); ax.set_ylabel("")
axes[-1].legend(SHORT_CLASS, title="Fertility", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
fig.suptitle("Figure 7: Fertility Class Mix within Each Category", fontsize=15, y=1.03)
save_show("fig07_class_by_category")

# Figure 8 – Pair plot of the main nutrients (N, P, K, Organic Carbon)
pp = sns.pairplot(df_clean, vars=["Available_N_kg_ha", "Available_P_kg_ha", "Available_K_kg_ha", "Organic_Carbon_g_kg"],
                  hue="Fertility_Class", hue_order=CLASS_ORDER, palette=PALETTE, corner=True,
                  plot_kws={"s": 14, "alpha": 0.6}, diag_kind="kde", height=2.3)
pp.fig.suptitle("Figure 8: Pair Plot of N, P, K and Organic Carbon by Fertility Class", y=1.02, fontsize=13)
save_show("fig08_pairplot", fig=pp.fig)

# **How to read the graphs**
# * **Fig. 1–2:** *Medium* is the biggest class; the score is split by two vertical cut-offs at 40 and 70.
# * **Fig. 3–4:** most nutrients are right-skewed with some high outliers (natural in soil data).
# * **Fig. 5:** N, P, K and Organic Carbon clearly shift upward from Low to High; EC is higher in Low-fertility soil; pH overlaps strongly between classes.
# * **Fig. 6–7:** the categorical features have fairly even class mixes, so they are not strong predictors.
# * **Fig. 8:** classes overlap in every pair of two nutrients – no single pair separates them perfectly, so a model must combine several features.


# ==============================================================================
# TASK 6 – CORRELATION ANALYSIS
# ==============================================================================

# **What we do:** measure the linear relationship (Pearson *r*, from −1 to +1) between every pair of numeric soil properties and draw a heatmap.
# `Fertility_Score` is excluded because it is a derived target-related field.

corr = df_clean[NUM_COLS].corr(method="pearson")
corr_pretty = corr.rename(index=SHORT, columns=SHORT)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_pretty, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Figure 9: Correlation Heatmap of Numerical Soil Properties (Pearson)")
save_show("fig09_correlation_heatmap")
save_table(corr.round(3), "06_correlation_matrix")

# strongest pairs
pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().rename("r").reset_index())
pairs.columns = ["Feature 1", "Feature 2", "r"]
pairs["|r|"] = pairs["r"].abs()
top_pairs = pairs.sort_values("|r|", ascending=False).head(5).round(3)
display(top_pairs)
save_table(top_pairs, "06_top_correlated_pairs", index=False)
print(f"Largest absolute correlation between two features = {pairs['|r|'].max():.3f}")

# Figure 10 – How strongly is each feature related to the fertility class? (Spearman rank correlation with the ordered class 0/1/2)
y_ordinal = df_clean["Fertility_Class"].map(CLASS_MAP)
target_corr = df_clean[NUM_COLS].apply(lambda s: s.corr(y_ordinal, method="spearman")).sort_values()
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.barh([SHORT[c] for c in target_corr.index], target_corr.values,
        color=["#d95f02" if v < 0 else "#1b9e77" for v in target_corr.values], edgecolor="black")
ax.axvline(0, color="black"); ax.set_xlabel("Spearman correlation with fertility class (Low=0, Medium=1, High=2)")
ax.set_title("Figure 10: Feature Correlation with Fertility Class")
save_show("fig10_target_correlation")
save_table(target_corr.rename("Spearman_with_class").round(3).to_frame(), "06_target_correlation")
display(target_corr.round(3).to_frame("Spearman with class").T)

# Evidence for a U-shaped pH effect: average ordinal fertility class (0=Low, 1=Medium, 2=High) inside pH bands
ph_bins = pd.cut(df_clean["pH"], bins=[5, 6, 6.5, 6.8, 7.2, 7.5, 8, 9])
ph_effect = df_clean.groupby(ph_bins, observed=True).apply(
    lambda g: pd.Series({"Samples": len(g), "Mean fertility class (0-2)": y_ordinal.loc[g.index].mean(),
                         "% Low": 100 * (g["Fertility_Class"] == "Low Fertility").mean(),
                         "% High": 100 * (g["Fertility_Class"] == "High Fertility").mean()})).round(2)
ph_effect.index = ph_effect.index.astype(str)
display(ph_effect)
save_table(ph_effect, "06_ph_effect")

# **Findings**
# * The soil properties are almost **uncorrelated with each other** (all pairwise |r| are tiny). This is typical of synthetic data generated independently, and it means there is **no multicollinearity problem**.
# * N, P, K and Organic Carbon have a clear **positive** relationship with the class; EC has a **negative** one (more salt → lower fertility).
# * pH has almost no *linear/rank* correlation with the class. In the score, pH is best near the neutral range, so both very low and very high pH hurt – a "U-shaped" effect that a straight-line correlation cannot capture. Tree-based and kernel models can learn such patterns.


# ==============================================================================
# TASK 7 – FEATURE SELECTION
# ==============================================================================

# **Goal:** choose input features that are (1) relevant, (2) available when a farmer tests a new soil sample, and (3) free from **target leakage**.
#
# **Step A – Remove columns that must not be inputs**
# | Column | Decision | Reason |
# |---|---|---|
# | `Fertility_Score` | ❌ Exclude | The class was *calculated* from it → direct **target leakage** |
# | `Fertility_Class` | ❌ Target | This is what we predict |
# | `Sample_ID` | ❌ Exclude | Just an identifier, no soil information |
# | `Source_Calibration` | ❌ Exclude | Describes how the synthetic data was made, not a soil property |
# | `District` | ❌ Exclude | Highly detailed (18 values) and repeats the information in `State` |
#
# **Step B – Statistical evidence** (mutual information for numeric features, Cramér's V for categorical features from Task 4).
# **Step C – Domain knowledge from the README:** the label is derived from **N, P, K, Organic Carbon, pH and EC**.
# These are ordinary soil-test measurements, so using them is *not* leakage – the score itself is the leaked item, and it is excluded.
#
# > *Note:* the statistics in Step B use the full dataset for guidance only. The choice is confirmed again, using **training data only**, in Task 9.

mi = mutual_info_classif(df_clean[NUM_COLS], y_ordinal, random_state=RANDOM_STATE)
SCORE_COMPONENTS = SIX
sel_table = pd.DataFrame({
    "Feature": NUM_COLS,
    "Mutual information": mi.round(3),
    "Spearman with class": [round(df_clean[c].corr(y_ordinal, method="spearman"), 3) for c in NUM_COLS],
    "In README score formula?": ["Yes" if c in SCORE_COMPONENTS else "No" for c in NUM_COLS],
})
sel_table["Decision"] = np.where(sel_table["In README score formula?"] == "Yes", "SELECT", "Drop")
sel_table = sel_table.sort_values("Mutual information", ascending=False).reset_index(drop=True)
display(sel_table)
save_table(sel_table, "07_feature_selection", index=False)

# Figure 11 – evidence used for feature selection
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
order = sel_table.sort_values("Mutual information")
axes[0].barh([SHORT[c] for c in order["Feature"]], order["Mutual information"],
             color=["#1b9e77" if d == "SELECT" else "#bdbdbd" for d in order["Decision"]], edgecolor="black")
axes[0].set_xlabel("Mutual information with fertility class"); axes[0].set_title("(a) Numeric features (green = selected)")
cv_ = chi_table[chi_table["Categorical feature"].isin(["State", "Soil_Type", "Crop"])]
axes[1].bar(cv_["Categorical feature"], cv_["Cramér's V"], color="#bdbdbd", edgecolor="black")
axes[1].set_ylim(0, 0.3); axes[1].set_ylabel("Cramér's V with fertility class")
axes[1].set_title("(b) Categorical features (all weak and dropped)")
for i, (v, p) in enumerate(zip(cv_["Cramér's V"], cv_["p-value"])):
    axes[1].text(i, v + 0.005, f"V={v}\np={p}", ha="center", fontsize=9)
fig.suptitle("Figure 11: Evidence Used for Feature Selection", fontsize=15, y=1.02)
save_show("fig11_feature_selection")

# **Interpretation**
# * **N, K, Organic Carbon and P** have the highest mutual information with the class.
# * **pH and EC** show low *univariate* mutual information. This does not mean they are useless: a single-feature filter cannot see an effect that only appears *together with* other features or that is U-shaped. The README states that they are part of the score, and Task 9 re-checks this with cross-validation.
# * **Sulphur and the five micronutrients** have near-zero information about the class and are not in the score formula → dropped.
# * **State, Soil_Type and Crop** are statistically weak (small Cramér's V) → dropped.
#
# ### Final selected features (6)
# `pH`, `Electrical_Conductivity_dS_m`, `Organic_Carbon_g_kg`, `Available_N_kg_ha`, `Available_P_kg_ha`, `Available_K_kg_ha`

FEATURES = SCORE_COMPONENTS.copy()
TARGET = "Fertility_Class"
print("Selected features:", FEATURES)
print("Leakage check – Fertility_Score in features?", "Fertility_Score" in FEATURES)
assert "Fertility_Score" not in FEATURES and TARGET not in FEATURES


# ==============================================================================
# TASK 8 – TRAIN-TEST SPLIT
# ==============================================================================

# **What we do:** split the data into **80 % training** and **20 % testing** data.
# * `stratify=y` keeps the same Low/Medium/High proportions in both parts (important because *High* is rare).
# * `random_state=42` makes the split identical every time.
# * The test set is kept untouched until the final evaluation, so the scores reflect performance on *unseen* soils.

X = df_clean[FEATURES].copy()
y = df_clean[TARGET].map(CLASS_MAP)             # target encoding: Low=0, Medium=1, High=2

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

print("Training set:", X_train.shape, "| Test set:", X_test.shape)
split_table = pd.DataFrame({
    "Full data": y.value_counts().sort_index(),
    "Train": y_train.value_counts().sort_index(),
    "Test": y_test.value_counts().sort_index(),
})
split_table.index = CLASS_ORDER
for col in ["Full data", "Train", "Test"]:
    split_table[col + " %"] = (100 * split_table[col] / split_table[col].sum()).round(1)
display(split_table)
save_table(split_table, "08_split_distribution")

# Figure 12 – class proportions in train and test
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(3); w = 0.38
ax.bar(x - w / 2, split_table["Train %"], w, label=f"Train (n={len(y_train)})", color="#4c78a8", edgecolor="black")
ax.bar(x + w / 2, split_table["Test %"], w, label=f"Test (n={len(y_test)})", color="#f58518", edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels(SHORT_CLASS); ax.set_ylabel("Percentage of samples"); ax.legend()
ax.set_title("Figure 12: Stratified Split – Class Proportions in Train and Test Sets")
save_show("fig12_split_distribution")


# ==============================================================================
# TASK 9 – IMPLEMENT AND COMPARE FOUR CLASSIFICATION ALGORITHMS
# ==============================================================================

# ### Preprocessing and encoding
# * **Target encoding:** Low = 0, Medium = 1, High = 2 (done in Task 8).
# * **Missing values:** `SimpleImputer(median)` inside the pipeline (nothing is missing now, but it protects new inputs).
# * **Feature scaling:** `StandardScaler` for **KNN and SVM**, because they depend on distances between samples and a feature like K (0–400) would otherwise dominate pH (5–8).
#   **Decision Tree and Random Forest** split on thresholds and are not affected by scale, so no scaling is applied to them.
# * **Categorical encoding:** the selected features are all numeric. For completeness, the helper below can one-hot encode categorical columns; it is used in the feature-set check further down.
# * **Pipelines:** every step lives inside a scikit-learn `Pipeline`, so the imputer/scaler learn only from training data (no leakage) and new samples are processed in exactly the same way.
#
# ### The four algorithms (simple settings, no heavy tuning)
# | Algorithm | Idea in one line | Settings |
# |---|---|---|
# | Decision Tree | A flow-chart of yes/no questions on feature values | `max_depth=5` (prevents memorising) |
# | Random Forest | Many trees vote together | `n_estimators=200` |
# | KNN | Look at the *k* most similar known soils and take a majority vote | `k=7`, scaled features |
# | SVM | Finds the best boundary between classes (RBF kernel curves the boundary) | `C=1`, `gamma='scale'`, scaled features |

def build_preprocessor(num_cols, cat_cols=(), scale=False):
    """Imputation + (optional) scaling for numeric columns, imputation + one-hot for categorical columns."""
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), list(num_cols))]
    if len(cat_cols) > 0:
        transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), list(cat_cols)))
    return ColumnTransformer(transformers)

MODEL_SPECS = {
    "Decision Tree": (DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), False),
    "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1), False),
    "KNN":           (KNeighborsClassifier(n_neighbors=7), True),
    "SVM":           (SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=RANDOM_STATE), True),
}

def make_pipeline(name, num_cols=FEATURES, cat_cols=()):
    model, scale = MODEL_SPECS[name]
    from sklearn.base import clone
    return Pipeline([("prep", build_preprocessor(num_cols, cat_cols, scale)), ("model", clone(model))])

pipelines = {name: make_pipeline(name) for name in MODEL_SPECS}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)               # fitted on TRAINING data only
    print(f"{name:14s} trained.")

# ### 9.1 Cross-validation on the training set (5-fold, stratified)
# Cross-validation repeatedly trains on 4/5 of the training data and validates on the remaining 1/5. It shows how *stable* each model is
# **without touching the test set**, so we can later use it to pick the final model fairly.

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_rows = []
for name, pipe in pipelines.items():
    res = cross_validate(make_pipeline(name), X_train, y_train, cv=skf,
                         scoring=["accuracy", "f1_macro", "f1_weighted"], n_jobs=1)
    cv_rows.append((name, res["test_accuracy"].mean(), res["test_accuracy"].std(),
                    res["test_f1_macro"].mean(), res["test_f1_macro"].std(), res["test_f1_weighted"].mean()))
cv_table = pd.DataFrame(cv_rows, columns=["Model", "CV Accuracy (mean)", "CV Accuracy (std)",
                                          "CV Macro F1 (mean)", "CV Macro F1 (std)", "CV Weighted F1 (mean)"]).set_index("Model").round(4)
display(cv_table)
save_table(cv_table, "09_cross_validation")

# ### 9.2 Re-checking feature selection with training data only
# We compare three feature sets using 5-fold CV on the **training** data (macro F1):
# * **A** – the 6 selected features (N, P, K, OC, pH, EC)
# * **B** – all 12 numeric soil properties
# * **C** – all 12 numeric + State, Soil_Type, Crop (categorical, one-hot encoded)
#
# If A is as good as or better than B and C, the extra columns only add noise.

CAT_FOR_CHECK = ["State", "Soil_Type", "Crop"]
sets = {"A: 6 selected": (FEATURES, []), "B: all 12 numeric": (NUM_COLS, []), "C: 12 numeric + 3 categorical": (NUM_COLS, CAT_FOR_CHECK)}
fs_rows = []
for set_name, (nc, cc) in sets.items():
    Xtr = df_clean.loc[X_train.index, nc + cc]
    row = {"Feature set": set_name, "No. of input columns": len(nc) + len(cc)}
    for name in MODEL_SPECS:
        s = cross_validate(make_pipeline(name, nc, cc), Xtr, y_train, cv=skf, scoring="f1_macro")["test_score"].mean()
        row[name] = round(s, 4)
    fs_rows.append(row)
fs_table = pd.DataFrame(fs_rows).set_index("Feature set")
display(fs_table)
save_table(fs_table, "09_feature_set_check")
print("Mean macro-F1 across the four models:\n", fs_table[list(MODEL_SPECS)].mean(axis=1).round(4).to_string())


# ==============================================================================
# TASK 10 – MODEL EVALUATION METRICS
# ==============================================================================

# All four trained pipelines now predict the **unseen test set** (20 % of the data).
# * **Accuracy** – share of all predictions that are correct.
# * **Precision** – of the soils predicted as a class, how many truly belong to it.
# * **Recall** – of the soils that truly belong to a class, how many were found.
# * **F1-score** – the balance (harmonic mean) of precision and recall.
# * **Macro average** – simple average over the 3 classes (treats the rare *High* class equally).
# * **Weighted average** – average weighted by how many test samples each class has.

predictions = {name: pipe.predict(X_test) for name, pipe in pipelines.items()}

metric_rows = []
for name, pred in predictions.items():
    metric_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision (macro)": precision_score(y_test, pred, average="macro", zero_division=0),
        "Recall (macro)": recall_score(y_test, pred, average="macro", zero_division=0),
        "F1 (macro)": f1_score(y_test, pred, average="macro", zero_division=0),
        "Precision (weighted)": precision_score(y_test, pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, pred, average="weighted", zero_division=0),
        "F1 (weighted)": f1_score(y_test, pred, average="weighted", zero_division=0),
    })
metrics = pd.DataFrame(metric_rows).set_index("Model").round(4)
display(metrics)
save_table(metrics, "10_test_metrics")

# Per-class precision / recall / F1 for each model
per_class_frames = []
for name, pred in predictions.items():
    rep = classification_report(y_test, pred, target_names=CLASS_ORDER, output_dict=True, zero_division=0)
    df_rep = pd.DataFrame(rep).T.loc[CLASS_ORDER + ["macro avg", "weighted avg"], ["precision", "recall", "f1-score", "support"]]
    df_rep.insert(0, "Model", name)
    per_class_frames.append(df_rep)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, pred, target_names=CLASS_ORDER, zero_division=0, digits=3))
per_class = pd.concat(per_class_frames).round(4)
save_table(per_class, "10_per_class_report")


# ==============================================================================
# TASK 11 – CONFUSION MATRICES
# ==============================================================================

# A confusion matrix is a table where **rows = true class** and **columns = predicted class**. Numbers on the diagonal are correct predictions; numbers
# off the diagonal are mistakes. The second figure shows the same matrices as row percentages (i.e. recall for each true class).

cms = {name: confusion_matrix(y_test, pred, labels=[0, 1, 2]) for name, pred in predictions.items()}

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (name, pred) in zip(axes.ravel(), predictions.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1, 2], display_labels=SHORT_CLASS, cmap="Blues",
                                            ax=ax, colorbar=False)
    ax.set_title(name); ax.grid(False)
fig.suptitle("Figure 13: Confusion Matrices (counts) – Test Set", fontsize=15, y=1.02)
save_show("fig13_confusion_matrices")

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, (name, pred) in zip(axes.ravel(), predictions.items()):
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1, 2], display_labels=SHORT_CLASS, cmap="Greens",
                                            normalize="true", values_format=".2f", ax=ax, colorbar=False)
    ax.set_title(name); ax.grid(False)
fig.suptitle("Figure 14: Confusion Matrices (row-normalised = recall per class)", fontsize=15, y=1.02)
save_show("fig14_confusion_matrices_normalized")

# Save as table + describe the mistakes in words
cm_rows = []
for name, cm in cms.items():
    for i, t in enumerate(SHORT_CLASS):
        for j, p in enumerate(SHORT_CLASS):
            cm_rows.append((name, t, p, int(cm[i, j])))
save_table(pd.DataFrame(cm_rows, columns=["Model", "True class", "Predicted class", "Count"]), "11_confusion_matrices", index=False)

print("Main error patterns on the test set:")
for name, cm in cms.items():
    errors = [(SHORT_CLASS[i], SHORT_CLASS[j], int(cm[i, j])) for i in range(3) for j in range(3) if i != j and cm[i, j] > 0]
    errors.sort(key=lambda t: -t[2])
    wrong = sum(e[2] for e in errors)
    txt = "; ".join(f"true {a} → predicted {b}: {n}" for a, b, n in errors) or "no errors"
    print(f"  {name:14s} total mistakes = {wrong:3d} of {cm.sum()}  |  {txt}")

# **Explanation of the matrices (test set = 200 soils: 64 Low, 119 Medium, 17 High)**
# * **SVM** makes the fewest mistakes (17 of 200), **Random Forest** 20, **KNN** 32 and **Decision Tree** 36.
# * Every model's mistakes are between **neighbouring classes** (Low↔Medium or Medium↔High) – a soil close to a cut-off score (40 or 70) can easily fall on the wrong side.
#   **No model ever confuses Low with High**, which would be the most serious kind of error.
# * The weak spot of every model is the rare **High** class. Its recall (the share of truly High soils that are found) is
#   0.35 for Decision Tree, 0.47 for KNN, 0.65 for Random Forest and 0.88 for SVM (15 of 17). Most missed High soils are predicted as Medium.
# * Because the High class has only 17 test samples, one extra mistake changes its recall by about 6 percentage points – so these class-level numbers should be read with care.


# ==============================================================================
# TASK 12 – MODEL COMPARISON
# ==============================================================================

# We put all results in one table and chart. **Selection rule:** the final model is the one with the best **macro-F1 on the unseen test set**
# (macro-F1 treats the rare *High* class equally to the large *Medium* class, so it is stricter than accuracy).
# We then check that the choice is supported by (a) the cross-validated scores from the training data and (b) a small gap between training and test accuracy
# (a large gap means the model has memorised the training data – *overfitting*).

comparison = metrics.join(cv_table[["CV Accuracy (mean)", "CV Macro F1 (mean)", "CV Macro F1 (std)"]])
comparison["Rank (test macro-F1)"] = comparison["F1 (macro)"].rank(ascending=False, method="min").astype(int)
comparison["Rank (CV macro-F1)"] = comparison["CV Macro F1 (mean)"].rank(ascending=False, method="min").astype(int)
display(comparison)
save_table(comparison, "12_model_comparison")

# Figure 15 – model performance chart
plot_cols = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "F1 (weighted)"]
ax = comparison[plot_cols].plot(kind="bar", figsize=(12, 5.5), edgecolor="black", width=0.8)
ax.set_ylim(max(0, comparison[plot_cols].min().min() - 0.1), 1.02)
ax.set_ylabel("Score"); ax.set_xlabel(""); plt.xticks(rotation=0)
ax.set_title("Figure 15: Model Performance Comparison on the Test Set")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5, fontsize=9)
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.2f", fontsize=7, padding=2)
save_show("fig15_model_comparison")

best_cv = comparison["CV Macro F1 (mean)"].idxmax()
best_test = comparison["F1 (macro)"].idxmax()
FINAL_MODEL_NAME = best_test
print(f"Best by test-set macro-F1 (selection rule)        : {best_test}")
print(f"Best by cross-validated macro-F1 (training data)  : {best_cv}")
diff = comparison.loc[best_cv, "CV Macro F1 (mean)"] - comparison.loc[best_test, "CV Macro F1 (mean)"]
if best_cv != best_test:
    print(f"The CV macro-F1 difference between them is only {abs(diff):.4f}, smaller than the CV standard deviation "
          f"({comparison.loc[best_test, 'CV Macro F1 (std)']:.4f} / {comparison.loc[best_cv, 'CV Macro F1 (std)']:.4f}) -> statistically a tie.")
print(f"\n➡ FINAL MODEL SELECTED: {FINAL_MODEL_NAME}")
sel = comparison.loc[FINAL_MODEL_NAME]
print(f"   Test accuracy = {sel['Accuracy']:.4f} | test macro-F1 = {sel['F1 (macro)']:.4f} | CV macro-F1 = {sel['CV Macro F1 (mean)']:.4f}")
# Overfitting check: train accuracy vs test accuracy for every model
gap = pd.DataFrame({
    "Train accuracy": {n: accuracy_score(y_train, p.predict(X_train)) for n, p in pipelines.items()},
    "Test accuracy": {n: accuracy_score(y_test, predictions[n]) for n in pipelines},
}).round(4)
gap["Gap"] = (gap["Train accuracy"] - gap["Test accuracy"]).round(4)
display(gap)
save_table(gap, "12_train_vs_test_accuracy")

# ### Why this model? (justification of the final selection)
# * **Best test results on every headline metric:** SVM has the highest accuracy (0.915), macro-precision (0.933), macro-recall (0.904), macro-F1 (0.917) and weighted-F1 (0.915).
# * **Best on the rare class:** SVM finds 15 of the 17 *High* soils (recall 0.88) with no false alarms, while Random Forest finds 11 of 17.
# * **Generalises well:** SVM's training accuracy (0.916) and test accuracy (0.915) are almost identical (gap 0.001). Random Forest scores 1.000 on its own training data but 0.900 on the test set (gap 0.100) – a sign of memorising.
# * **Cross-validation agrees it is competitive:** SVM (0.823) and Random Forest (0.826) are statistically tied on CV macro-F1 (difference 0.003, far smaller than the fold-to-fold spread). Both clearly beat KNN (0.749) and the Decision Tree (0.691).
# * **Honest caveat:** the test set has only 200 rows (17 High), and the final choice used the test results, so the reported test scores are slightly optimistic. Random Forest is a very close and equally valid alternative.
# * **Overall ranking by test macro-F1:** SVM > Random Forest > KNN > Decision Tree.


# ==============================================================================
# TASK 13 – SOIL PARAMETER CONTRIBUTION ANALYSIS
# ==============================================================================

# Which soil parameters matter most for the prediction?
# * **Random Forest feature importance (Gini/impurity):** how much each feature helps the trees purify their splits (computed on training data).
# * **Permutation importance:** randomly shuffle one feature in the **test set** and see how much the macro-F1 drops. A bigger drop = a more important feature.
#   This method works for *any* model, so we compute it for all four.

rf_pipe = pipelines["Random Forest"]
rf_imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_,
                   index=[n.split("__")[-1] for n in rf_pipe.named_steps["prep"].get_feature_names_out()]).sort_values(ascending=False)

perm_tables, perm_std = {}, {}
for name, pipe in pipelines.items():
    r = permutation_importance(pipe, X_test, y_test, scoring="f1_macro", n_repeats=30, random_state=RANDOM_STATE, n_jobs=1)
    perm_tables[name] = pd.Series(r.importances_mean, index=FEATURES)
    perm_std[name] = pd.Series(r.importances_std, index=FEATURES)
perm_df = pd.DataFrame(perm_tables)
perm_df.insert(0, "RF impurity importance", rf_imp)
perm_df = perm_df.sort_values(FINAL_MODEL_NAME, ascending=False)
display(perm_df.round(4))
save_table(perm_df.round(4), "13_feature_importance")
save_table(pd.DataFrame(perm_std).round(4), "13_permutation_std")

# Figure 16 – feature importance
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
rf_sorted = rf_imp.sort_values()
axes[0].barh([SHORT[c] for c in rf_sorted.index], rf_sorted.values, color="#4c78a8", edgecolor="black")
axes[0].set_xlabel("Impurity-based importance"); axes[0].set_title("(a) Random Forest feature importance")
fp = perm_df[FINAL_MODEL_NAME].sort_values(); fs_ = perm_std[FINAL_MODEL_NAME].loc[fp.index]
axes[1].barh([SHORT[c] for c in fp.index], fp.values, xerr=fs_.values, color="#f58518", edgecolor="black", capsize=3)
axes[1].set_xlabel("Drop in test macro-F1 when feature is shuffled")
axes[1].set_title(f"(b) Permutation importance – selected model ({FINAL_MODEL_NAME})")
fig.suptitle("Figure 16: Contribution of Soil Parameters to Fertility Classification", fontsize=15, y=1.02)
save_show("fig16_feature_importance")

print("Ranking of soil parameters (selected model, permutation importance):")
for i, (f, v) in enumerate(perm_df[FINAL_MODEL_NAME].sort_values(ascending=False).items(), 1):
    print(f"  {i}. {f:32s} {v:.4f}")

# **What the importance results show**
# * **Four parameters do almost all the work:** Nitrogen, Potassium, Organic Carbon and Phosphorus. For the selected SVM, shuffling any one of them reduces the test macro-F1 by roughly 0.25 – 0.32.
#   Random Forest's own importance agrees: these four hold about 85 % of the total importance (K 0.26, OC 0.21, N 0.20, P 0.18).
# * **pH and EC matter much less.** Their permutation importance is about 0.09 (pH) and 0.03 (EC) for SVM. pH acts through a *U-shaped* effect (best near pH 6.5–7, worse when very acidic or alkaline – see Task 6) and EC only lowers fertility when it is high, so each contributes less than the nutrients.
# * The exact order of the top four differs between methods (e.g. N first for SVM permutation, K first for Random Forest), because their importances are close and the test set is small. The overall message is the same in every model.
# * The five micronutrients and Sulphur were removed in Task 7 – the Task 9 check showed that adding them *lowered* the cross-validated score.
# * Remember: the model is largely **rediscovering the weights of the README scoring formula** (see Limitations), so this is not new agronomic evidence.


# ==============================================================================
# TASK 14 – PREDICTION ON NEW / UNSEEN SOIL SAMPLES
# ==============================================================================

# We use the **selected model's complete pipeline** (imputer → scaler if needed → classifier, fitted on the training data).
# Because the preprocessing is *inside* the pipeline, we simply pass raw soil measurements and the same transformations are applied automatically.
#
# Below are five hand-made soil samples (not from the dataset): a nutrient-rich soil, a poor and salty soil, an average soil, a borderline soil and
# an average soil with a missing value (to show that the imputer works).

final_pipe = pipelines[FINAL_MODEL_NAME]
new_samples = pd.DataFrame([
    ["Sample A – rich, balanced soil",   6.9, 0.30, 15.0, 250.0, 75.0, 290.0],
    ["Sample B – poor, salty soil",      5.8, 1.30,  3.0, 105.0, 12.0,  70.0],
    ["Sample C – average soil",          7.0, 0.45,  6.5, 177.0, 36.0, 180.0],
    ["Sample D – borderline soil",       7.2, 0.50,  9.0, 195.0, 45.0, 215.0],
    ["Sample E – average, P missing",    6.8, 0.40,  7.0, 180.0, np.nan, 175.0],
], columns=["Sample"] + FEATURES).set_index("Sample")
display(new_samples)

assert list(new_samples.columns) == list(final_pipe.feature_names_in_), "Column names/order must match training"
pred_idx = final_pipe.predict(new_samples)
proba = pd.DataFrame(final_pipe.predict_proba(new_samples), index=new_samples.index, columns=SHORT_CLASS).round(3)
result = new_samples.copy()
result["Predicted class"] = [CLASS_ORDER[i] for i in pred_idx]
result["Confidence"] = proba.max(axis=1)
display(result[["Predicted class", "Confidence"]].join(proba))
save_table(result.join(proba), "14_new_sample_predictions")

# Do the other three models agree?
agree = pd.DataFrame({name: [CLASS_ORDER[i].replace(" Fertility", "") for i in pipe.predict(new_samples)]
                      for name, pipe in pipelines.items()}, index=new_samples.index)
display(agree)
save_table(agree, "14_all_models_on_new_samples")

# Figure 17 – predicted probabilities for the new samples
fig, ax = plt.subplots(figsize=(10, 4.8))
proba.plot(kind="barh", stacked=True, color=[PALETTE[c] for c in CLASS_ORDER], ax=ax, edgecolor="white")
ax.set_xlim(0, 1); ax.set_xlabel("Predicted probability"); ax.set_ylabel("")
ax.set_title(f"Figure 17: Predicted Class Probabilities for New Soil Samples ({FINAL_MODEL_NAME})")
ax.legend(title="Class", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.invert_yaxis()
save_show("fig17_new_sample_predictions")

# Reusable helper: type your own soil test values and get a prediction
def predict_soil_fertility(pH, EC, organic_carbon, N, P, K, model=final_pipe):
    """Predict fertility class for one soil sample using the SAME pipeline that was trained."""
    row = pd.DataFrame([[pH, EC, organic_carbon, N, P, K]], columns=FEATURES)
    cls = CLASS_ORDER[int(model.predict(row)[0])]
    probs = dict(zip(SHORT_CLASS, model.predict_proba(row)[0].round(3)))
    return cls, probs

print(predict_soil_fertility(pH=7.1, EC=0.35, organic_carbon=11.0, N=215, P=60, K=260))

# Save the final model so it can be reused later (e.g., in a web app)
import joblib
joblib.dump(final_pipe, os.path.join(OUT_DIR, "final_soil_fertility_model.joblib"))


# ==============================================================================
# Final Summary, Conclusion and Limitations
# ==============================================================================

summary = {
    "rows": int(len(df_clean)), "columns": int(raw.shape[1]), "missing_cells": int(df_clean.isna().sum().sum()),
    "duplicate_rows": dup_all, "train_size": int(len(X_train)), "test_size": int(len(X_test)),
    "random_state": RANDOM_STATE, "features": FEATURES, "final_model": FINAL_MODEL_NAME,
    "best_by_test_macro_f1": best_test,
    "test_metrics": metrics.to_dict(orient="index"),
    "cv_metrics": cv_table.to_dict(orient="index"),
}
with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=float)
print(json.dumps({k: summary[k] for k in ["rows", "train_size", "test_size", "features", "final_model"]}, indent=2))
display(comparison[["Accuracy", "F1 (macro)", "F1 (weighted)", "CV Macro F1 (mean)"]])

# ## Final Conclusion
# * We built a complete multiclass pipeline to classify soil as **Low, Medium or High Fertility** from six soil-test values (pH, EC, Organic Carbon, N, P, K).
# * The data had **no missing values and no duplicates**; cleaning mainly consisted of consistency checks. Soil properties are almost uncorrelated with each other, and the classes are imbalanced (59.6 % Medium, 31.9 % Low, 8.5 % High).
# * Using an 80:20 **stratified** split (random_state = 42), four algorithms were compared. On the unseen test set **SVM was best** (accuracy 0.915, macro-F1 0.917), followed by Random Forest (0.900 / 0.852), KNN (0.840 / 0.771) and Decision Tree (0.820 / 0.722).
# * **N, K, Organic Carbon and P** are the most influential parameters; pH and EC contribute less.
# * The final SVM pipeline predicted High, Low, Medium, Medium and Medium for five new hand-made soils (all four models agree on every sample, and the imputer handled one missing value). These samples have no ground-truth labels, so this demonstrates *how to use* the pipeline, not its accuracy. It can be reused through `predict_soil_fertility(...)`.
#
# ## Limitations (please read)
# 1. **Synthetic data:** the 1,000 rows are source-calibrated synthetic teaching data, not 1,000 authentic laboratory observations. Only the AgriNet (258), Krishnagiri (125) and Dharmapuri (125) records are authentic sources. Results must **not** be presented as real-world field accuracy.
# 2. **Derived label:** `Fertility_Class` was calculated from N, P, K, OC, pH and EC using a scoring formula. The model is therefore learning to *approximate that formula*, not discovering natural soil fertility. High accuracy partly reflects this; a label from field experts or crop yields would be a harder, more meaningful test.
# 3. **Why the accuracy is not ~100 %:** the exact score formula is not provided, includes non-linear parts (e.g. the pH optimum) and the cut-offs (40 and 70) fall in dense regions of the data, so soils near a cut-off are easily misclassified.
# 4. **Leakage handling:** `Fertility_Score` was excluded from the inputs. The six selected features are the raw inputs of the formula; they are legitimate soil-test measurements but make the task easier than a real-world one.
# 5. **Small minority class:** only 85 High samples (17 in the test set), so High-class metrics are uncertain. A single 80:20 split and 200 test rows give approximate, not exact, performance estimates.
# 6. **Light tuning:** the models use simple, sensible settings without hyper-parameter search, to keep the project at BCA level. Careful tuning might change the ranking of SVM and Random Forest slightly.
# 7. **Clipped values:** several score-related columns (pH, OC, N, P, K) have many rows stuck at their minimum or maximum, which points to range-clipping in the synthetic generator and is not typical of raw lab data.
# 8. **Generalisation:** the data covers 8 Indian states and 18 districts; it should not be applied to other regions or soil-testing methods without re-validation.
#
# ## References
# 1. Mendeley Data – AgriNet: Bareilly (Uttar Pradesh) soil dataset, DOI 10.17632/c7tr9j78td.1.
# 2. Mendeley Data – Krishnagiri raw soil dataset, DOI 10.17632/7bk36mh3tc.1.
# 3. Dharmapuri soil data article (as cited in the dataset README).
# 4. Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
# 5. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95.
# 6. Waskom, M. (2021). seaborn: statistical data visualization. *Journal of Open Source Software*, 6(60), 3021.
# 7. McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*.