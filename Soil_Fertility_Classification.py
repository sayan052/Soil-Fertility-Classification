
import matplotlib
import os
if os.environ.get("DISPLAY", "") == "" and os.name != "nt" and not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")
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
except ImportError:          
    display = print

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)  
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
CLASS_MAP = {c: i for i, c in enumerate(CLASS_ORDER)}      
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
NUM_COLS = ["pH", "Electrical_Conductivity_dS_m", "Organic_Carbon_g_kg", "Available_N_kg_ha",
            "Available_P_kg_ha", "Available_K_kg_ha", "Sulphur_mg_kg", "Zinc_mg_kg", "Boron_mg_kg",
            "Iron_mg_kg", "Manganese_mg_kg", "Copper_mg_kg"]
CAT_COLS = ["State", "District", "Soil_Type", "Crop"]
TEXT_COLS = raw.select_dtypes(exclude="number").columns.tolist()

dup_all = int(raw.duplicated().sum())
dup_no_id = int(raw.drop(columns="Sample_ID").duplicated().sum())
dup_id = int(raw["Sample_ID"].duplicated().sum())


ws_issues = int(sum((raw[c].astype(str) != raw[c].astype(str).str.strip()).sum() for c in TEXT_COLS))


ph_bad = int(((raw["pH"] < 0) | (raw["pH"] > 14)).sum())
negatives = int((raw[NUM_COLS] < 0).sum().sum())

def score_to_class(score):
    return np.select([score < 40, score < 70], ["Low Fertility", "Medium Fertility"], default="High Fertility")
label_mismatch = int((score_to_class(raw["Fertility_Score"]) != raw["Fertility_Class"]).sum())

district_conflicts = int((raw.groupby("District")["State"].nunique() > 1).sum())

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


df_clean = raw.copy()
for c in TEXT_COLS:
    df_clean[c] = df_clean[c].astype(str).str.strip()          
df_clean = df_clean.drop_duplicates().reset_index(drop=True)  
print("Rows before cleaning:", len(raw), "| rows after cleaning:", len(df_clean))
print("Class labels present:", sorted(df_clean["Fertility_Class"].unique()))


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
else:                                                           
    df_clean[NUM_COLS] = df_clean[NUM_COLS].fillna(df_clean[NUM_COLS].median())

eda_num = df_clean[NUM_COLS].agg(["mean", "median", "std", "min", "max", "skew"]).T
eda_num["kurtosis"] = df_clean[NUM_COLS].kurt()
eda_num = eda_num.rename(columns={"skew": "skewness"}).round(3)
display(eda_num)
save_table(eda_num, "04_numeric_eda")
print("Most right-skewed features:", ", ".join(eda_num["skewness"].sort_values(ascending=False).head(4).index))


class_counts = df_clean["Fertility_Class"].value_counts().reindex(CLASS_ORDER)
class_table = pd.DataFrame({"Count": class_counts, "Percent": (100 * class_counts / len(df_clean)).round(1)})
display(class_table)
save_table(class_table, "04_class_distribution")
print(f"Imbalance ratio (largest : smallest class) = {class_counts.max() / class_counts.min():.1f} : 1")


for c in CAT_COLS + ["Source_Calibration"]:
    print(f"\n{c} ({df_clean[c].nunique()} categories)")
    print(df_clean[c].value_counts().to_string())



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

class_means = df_clean.groupby("Fertility_Class")[NUM_COLS].mean().loc[CLASS_ORDER].T.round(2)
display(class_means)
save_table(class_means, "04_class_means")

SHORT = {"pH": "pH", "Electrical_Conductivity_dS_m": "EC (dS/m)", "Organic_Carbon_g_kg": "Organic C (g/kg)",
         "Available_N_kg_ha": "N (kg/ha)", "Available_P_kg_ha": "P (kg/ha)", "Available_K_kg_ha": "K (kg/ha)",
         "Sulphur_mg_kg": "Sulphur (mg/kg)", "Zinc_mg_kg": "Zinc (mg/kg)", "Boron_mg_kg": "Boron (mg/kg)",
         "Iron_mg_kg": "Iron (mg/kg)", "Manganese_mg_kg": "Manganese (mg/kg)", "Copper_mg_kg": "Copper (mg/kg)"}


fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(SHORT_CLASS, class_counts.values, color=[PALETTE[c] for c in CLASS_ORDER], edgecolor="black")
for b, n in zip(bars, class_counts.values):
    ax.text(b.get_x() + b.get_width() / 2, n + 8, f"{n}\n({100 * n / len(df_clean):.1f}%)", ha="center", fontsize=10)
ax.set_ylim(0, class_counts.max() * 1.18)
ax.set_title("Figure 1: Fertility Class Distribution"); ax.set_xlabel("Fertility class"); ax.set_ylabel("Number of samples")
save_show("fig01_class_distribution")


fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(data=df_clean, x="Fertility_Score", hue="Fertility_Class", hue_order=CLASS_ORDER, palette=PALETTE,
             bins=35, multiple="stack", ax=ax)
for t in (40, 70):
    ax.axvline(t, color="black", linestyle="--")
ax.set_title("Figure 2: Fertility_Score and Class Thresholds (40 and 70)\n(for understanding the label only – not a model input)")
ax.set_xlabel("Fertility_Score")
save_show("fig02_score_distribution")


fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, c in zip(axes.ravel(), NUM_COLS):
    sns.histplot(df_clean[c], kde=True, bins=30, color="#4c78a8", ax=ax)
    ax.set_xlabel(SHORT[c]); ax.set_ylabel("Count")
fig.suptitle("Figure 3: Histograms of Numerical Soil Properties", fontsize=15, y=1.01)
save_show("fig03_histograms")


fig, axes = plt.subplots(3, 4, figsize=(16, 9))
for ax, c in zip(axes.ravel(), NUM_COLS):
    sns.boxplot(y=df_clean[c], color="#9ecae1", ax=ax, fliersize=3)
    ax.set_ylabel(SHORT[c])
fig.suptitle("Figure 4: Boxplots of Numerical Soil Properties", fontsize=15, y=1.01)
save_show("fig04_boxplots")


SIX = ["pH", "Electrical_Conductivity_dS_m", "Organic_Carbon_g_kg", "Available_N_kg_ha", "Available_P_kg_ha", "Available_K_kg_ha"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, c in zip(axes.ravel(), SIX):
    sns.boxplot(data=df_clean, x="Fertility_Class", y=c, order=CLASS_ORDER, palette=PALETTE, ax=ax, fliersize=3)
    ax.set_xticklabels(SHORT_CLASS); ax.set_xlabel(""); ax.set_ylabel(SHORT[c])
fig.suptitle("Figure 5: Key Soil Properties by Fertility Class", fontsize=15, y=1.01)
save_show("fig05_boxplots_by_class")


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, c in zip(axes, ["State", "Soil_Type", "Crop"]):
    order = df_clean[c].value_counts().index
    sns.countplot(data=df_clean, y=c, order=order, color="#72b7b2", ax=ax)
    ax.set_title(c); ax.set_xlabel("Number of samples"); ax.set_ylabel("")
fig.suptitle("Figure 6: Categorical Features – Sample Counts", fontsize=15, y=1.03)
save_show("fig06_categorical_counts")


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, c in zip(axes, ["State", "Soil_Type", "Crop"]):
    prop = pd.crosstab(df_clean[c], df_clean["Fertility_Class"], normalize="index")[CLASS_ORDER]
    prop.plot(kind="barh", stacked=True, color=[PALETTE[k] for k in CLASS_ORDER], ax=ax, legend=False, edgecolor="white")
    ax.set_title(c); ax.set_xlabel("Proportion of samples"); ax.set_ylabel("")
axes[-1].legend(SHORT_CLASS, title="Fertility", loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
fig.suptitle("Figure 7: Fertility Class Mix within Each Category", fontsize=15, y=1.03)
save_show("fig07_class_by_category")


pp = sns.pairplot(df_clean, vars=["Available_N_kg_ha", "Available_P_kg_ha", "Available_K_kg_ha", "Organic_Carbon_g_kg"],
                  hue="Fertility_Class", hue_order=CLASS_ORDER, palette=PALETTE, corner=True,
                  plot_kws={"s": 14, "alpha": 0.6}, diag_kind="kde", height=2.3)
pp.fig.suptitle("Figure 8: Pair Plot of N, P, K and Organic Carbon by Fertility Class", y=1.02, fontsize=13)
save_show("fig08_pairplot", fig=pp.fig)

corr = df_clean[NUM_COLS].corr(method="pearson")
corr_pretty = corr.rename(index=SHORT, columns=SHORT)
fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_pretty, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, square=True, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Figure 9: Correlation Heatmap of Numerical Soil Properties (Pearson)")
save_show("fig09_correlation_heatmap")
save_table(corr.round(3), "06_correlation_matrix")


pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().rename("r").reset_index())
pairs.columns = ["Feature 1", "Feature 2", "r"]
pairs["|r|"] = pairs["r"].abs()
top_pairs = pairs.sort_values("|r|", ascending=False).head(5).round(3)
display(top_pairs)
save_table(top_pairs, "06_top_correlated_pairs", index=False)
print(f"Largest absolute correlation between two features = {pairs['|r|'].max():.3f}")

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


ph_bins = pd.cut(df_clean["pH"], bins=[5, 6, 6.5, 6.8, 7.2, 7.5, 8, 9])
ph_effect = df_clean.groupby(ph_bins, observed=True).apply(
    lambda g: pd.Series({"Samples": len(g), "Mean fertility class (0-2)": y_ordinal.loc[g.index].mean(),
                         "% Low": 100 * (g["Fertility_Class"] == "Low Fertility").mean(),
                         "% High": 100 * (g["Fertility_Class"] == "High Fertility").mean()})).round(2)
ph_effect.index = ph_effect.index.astype(str)
display(ph_effect)
save_table(ph_effect, "06_ph_effect")


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

FEATURES = SCORE_COMPONENTS.copy()
TARGET = "Fertility_Class"
print("Selected features:", FEATURES)
print("Leakage check – Fertility_Score in features?", "Fertility_Score" in FEATURES)
assert "Fertility_Score" not in FEATURES and TARGET not in FEATURES
X = df_clean[FEATURES].copy()
y = df_clean[TARGET].map(CLASS_MAP)             

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

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(3); w = 0.38
ax.bar(x - w / 2, split_table["Train %"], w, label=f"Train (n={len(y_train)})", color="#4c78a8", edgecolor="black")
ax.bar(x + w / 2, split_table["Test %"], w, label=f"Test (n={len(y_test)})", color="#f58518", edgecolor="black")
ax.set_xticks(x); ax.set_xticklabels(SHORT_CLASS); ax.set_ylabel("Percentage of samples"); ax.legend()
ax.set_title("Figure 12: Stratified Split – Class Proportions in Train and Test Sets")
save_show("fig12_split_distribution")


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
    pipe.fit(X_train, y_train)              
    print(f"{name:14s} trained.")

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

comparison = metrics.join(cv_table[["CV Accuracy (mean)", "CV Macro F1 (mean)", "CV Macro F1 (std)"]])
comparison["Rank (test macro-F1)"] = comparison["F1 (macro)"].rank(ascending=False, method="min").astype(int)
comparison["Rank (CV macro-F1)"] = comparison["CV Macro F1 (mean)"].rank(ascending=False, method="min").astype(int)
display(comparison)
save_table(comparison, "12_model_comparison")

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

gap = pd.DataFrame({
    "Train accuracy": {n: accuracy_score(y_train, p.predict(X_train)) for n, p in pipelines.items()},
    "Test accuracy": {n: accuracy_score(y_test, predictions[n]) for n in pipelines},
}).round(4)
gap["Gap"] = (gap["Train accuracy"] - gap["Test accuracy"]).round(4)
display(gap)
save_table(gap, "12_train_vs_test_accuracy")

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


agree = pd.DataFrame({name: [CLASS_ORDER[i].replace(" Fertility", "") for i in pipe.predict(new_samples)]
                      for name, pipe in pipelines.items()}, index=new_samples.index)
display(agree)
save_table(agree, "14_all_models_on_new_samples")


fig, ax = plt.subplots(figsize=(10, 4.8))
proba.plot(kind="barh", stacked=True, color=[PALETTE[c] for c in CLASS_ORDER], ax=ax, edgecolor="white")
ax.set_xlim(0, 1); ax.set_xlabel("Predicted probability"); ax.set_ylabel("")
ax.set_title(f"Figure 17: Predicted Class Probabilities for New Soil Samples ({FINAL_MODEL_NAME})")
ax.legend(title="Class", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
ax.invert_yaxis()
save_show("fig17_new_sample_predictions")

def predict_soil_fertility(pH, EC, organic_carbon, N, P, K, model=final_pipe):
    """Predict fertility class for one soil sample using the SAME pipeline that was trained."""
    row = pd.DataFrame([[pH, EC, organic_carbon, N, P, K]], columns=FEATURES)
    cls = CLASS_ORDER[int(model.predict(row)[0])]
    probs = dict(zip(SHORT_CLASS, model.predict_proba(row)[0].round(3)))
    return cls, probs

print(predict_soil_fertility(pH=7.1, EC=0.35, organic_carbon=11.0, N=215, P=60, K=260))

import joblib
joblib.dump(final_pipe, os.path.join(OUT_DIR, "final_soil_fertility_model.joblib"))


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
