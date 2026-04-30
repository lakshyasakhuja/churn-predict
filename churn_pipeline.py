"""
================================================================================
  CUSTOMER CHURN ANALYSIS AND PREDICTION — COMPLETE ML PIPELINE
  Telco Churn Benchmark
  Author: Lakshya Sakhuja
  Python 3.10+ | scikit-learn 1.8
================================================================================

Pipeline stages
  Task 1 — Data Preparation       : load, clean, type-cast, missing-value audit
  Task 2 — Train / Test Split     : stratified 80/20 with reproducible seed
  Task 3 — Feature Engineering    : domain features + encoding + scaling
  Task 4 — Model Selection        : 6 classifiers benchmarked on 5-fold CV
  Task 5 — Model Training         : best model + GridSearchCV hyperparameter tuning
  Task 6 — Model Evaluation       : accuracy, precision, recall, F1, ROC-AUC,
                                    confusion matrix, learning curves, PR curve
================================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, warnings, json, pickle, logging, time
from pathlib import Path
from collections import Counter

# ── data ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd

# ── viz ───────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ── modelling ─────────────────────────────────────────────────────────────────
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    cross_val_score, learning_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, confusion_matrix,
    classification_report, precision_recall_curve,
    average_precision_score, matthews_corrcoef
)

warnings.filterwarnings("ignore")

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("churn")

# ── paths ─────────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
VIZ    = BASE / "visualizations"
MDL    = BASE / "models"
OUT    = BASE / "outputs"
DATA   = BASE / "data.csv"
for p in (VIZ, MDL, OUT): p.mkdir(exist_ok=True)

# ── aesthetics ────────────────────────────────────────────────────────────────
C = dict(
    bg="#0D1117", panel="#161B22", border="#30363D",
    text="#E6EDF3", muted="#8B949E",
    red="#FF6B6B", teal="#00D4AA", gold="#FFB347",
    blue="#58A6FF", purple="#BC8CFF", green="#3FB950"
)
CHURN_PAL = [C["teal"], C["red"]]

def _style():
    plt.rcParams.update({
        "figure.facecolor":  C["bg"],
        "axes.facecolor":    C["panel"],
        "axes.edgecolor":    C["border"],
        "axes.labelcolor":   C["text"],
        "xtick.color":       C["muted"],
        "ytick.color":       C["muted"],
        "text.color":        C["text"],
        "grid.color":        C["border"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.5,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.titlepad":     12,
        "legend.facecolor":  C["panel"],
        "legend.edgecolor":  C["border"],
        "figure.dpi":        150,
        "savefig.bbox":      "tight",
        "savefig.facecolor": C["bg"],
    })

_style()

SEED = 42
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 1 — DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

log.info("=" * 72)
log.info("TASK 1 — DATA PREPARATION")
log.info("=" * 72)

df = pd.read_csv(DATA)
log.info(f"Raw dataset shape : {df.shape}")

# ── 1a. Type casting ──────────────────────────────────────────────────────────
#  TotalCharges arrives as float; some entries can be blank strings in raw CSVs
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

# ── 1b. Missing values ────────────────────────────────────────────────────────
n_missing = df.isnull().sum()
log.info(f"Missing value counts:\n{n_missing[n_missing > 0]}")

# Impute TotalCharges with tenure × MonthlyCharges (business-logic imputation)
mask_missing = df["TotalCharges"].isnull()
df.loc[mask_missing, "TotalCharges"] = (
    df.loc[mask_missing, "tenure"] * df.loc[mask_missing, "MonthlyCharges"]
)
log.info(f"  → Imputed {mask_missing.sum()} TotalCharges rows via tenure × MonthlyCharges")

# ── 1c. Drop customerID (non-informative identifier) ─────────────────────────
df.drop(columns=["customerID"], inplace=True)

# ── 1d. Binary-encode target ──────────────────────────────────────────────────
df["Churn"] = (df["Churn"] == "Yes").astype(int)
log.info(f"Churn distribution: {Counter(df['Churn'])}")
log.info(f"Churn rate         : {df['Churn'].mean()*100:.2f}%")

# ── Snapshot of clean dataset ─────────────────────────────────────────────────
log.info(f"Clean dataset shape: {df.shape}")
df_clean = df.copy()   # preserve for EDA section

# ══════════════════════════════════════════════════════════════════════════════
# EXPLORATORY DATA ANALYSIS  (visualisations saved to /visualizations)
# ══════════════════════════════════════════════════════════════════════════════

log.info("\n── Generating EDA visualisations ──")

# ── EDA-1: Churn distribution (donut) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])
counts = df["Churn"].value_counts()
labels = ["Retained", "Churned"]
wedges, texts, autotexts = ax.pie(
    counts, labels=labels, autopct="%1.1f%%",
    colors=CHURN_PAL, startangle=90,
    wedgeprops={"width": 0.55, "edgecolor": C["bg"], "linewidth": 3},
    textprops={"color": C["text"], "fontsize": 13}
)
for at in autotexts:
    at.set_fontsize(14); at.set_fontweight("bold"); at.set_color(C["bg"])
ax.set_title("Customer Churn Distribution\n(7,043 Customers)", pad=18, color=C["text"])
ax.text(0, 0, f"{counts[1]}\nChurned", ha="center", va="center",
        fontsize=14, color=C["red"], fontweight="bold")
plt.tight_layout()
plt.savefig(VIZ / "01_churn_distribution.png"); plt.close()
log.info("  ✓ 01_churn_distribution.png")

# ── EDA-2: Numeric distributions by churn status ──────────────────────────────
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, col in zip(axes, num_cols):
    for val, color, label in zip([0, 1], CHURN_PAL, ["Retained", "Churned"]):
        data = df[df["Churn"] == val][col]
        ax.hist(data, bins=35, alpha=0.75, color=color, label=label, edgecolor="none")
    ax.set_title(col); ax.set_xlabel(col); ax.set_ylabel("Count")
    ax.legend(framealpha=0.3)
    ax.grid(True, alpha=0.3)
fig.suptitle("Numeric Feature Distributions by Churn Status", y=1.02, fontsize=14, color=C["text"])
plt.tight_layout()
plt.savefig(VIZ / "02_numeric_distributions.png"); plt.close()
log.info("  ✓ 02_numeric_distributions.png")

# ── EDA-3: Churn rate by Contract type ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
contract_churn = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
bars = ax.bar(contract_churn.index, contract_churn.values,
              color=[C["red"], C["gold"], C["teal"]], width=0.5, edgecolor="none")
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title("Churn Rate by Contract Type"); ax.set_ylabel("Churn Rate (%)")
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
            f"{b.get_height():.1f}%", ha="center", va="bottom",
            fontsize=11, color=C["text"], fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ / "03_churn_by_contract.png"); plt.close()
log.info("  ✓ 03_churn_by_contract.png")

# ── EDA-4: Churn rate by Internet Service ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
inet_churn = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)*100
bars = ax.bar(inet_churn.index, inet_churn.values,
              color=[C["red"], C["blue"], C["teal"]], width=0.5)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.set_title("Churn Rate by Internet Service"); ax.set_ylabel("Churn Rate (%)")
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
            f"{b.get_height():.1f}%", ha="center", fontsize=11,
            color=C["text"], fontweight="bold")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ / "04_churn_by_internet.png"); plt.close()
log.info("  ✓ 04_churn_by_internet.png")

# ── EDA-5: Tenure vs MonthlyCharges scatter ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for val, color, label in zip([0, 1], CHURN_PAL, ["Retained", "Churned"]):
    sub = df[df["Churn"] == val]
    ax.scatter(sub["tenure"], sub["MonthlyCharges"], c=color, alpha=0.3,
               s=8, label=label, rasterized=True)
ax.set_xlabel("Tenure (months)"); ax.set_ylabel("Monthly Charges ($)")
ax.set_title("Tenure vs Monthly Charges  |  Colored by Churn Status")
ax.legend(markerscale=4, framealpha=0.4)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ / "05_tenure_vs_charges_scatter.png"); plt.close()
log.info("  ✓ 05_tenure_vs_charges_scatter.png")

# ── EDA-6: Correlation heatmap (numeric) ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
corr_df = df[["tenure","MonthlyCharges","TotalCharges","SeniorCitizen","Churn"]]
corr = corr_df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, square=True, ax=ax, mask=mask,
            linewidths=0.5, linecolor=C["border"],
            annot_kws={"size": 11, "weight": "bold"},
            cbar_kws={"shrink": 0.8})
ax.set_title("Correlation Matrix — Numeric Features")
plt.tight_layout()
plt.savefig(VIZ / "06_correlation_heatmap.png"); plt.close()
log.info("  ✓ 06_correlation_heatmap.png")

# ── EDA-7: Categorical churn overview ─────────────────────────────────────────
cat_cols = ["gender","Partner","Dependents","PhoneService",
            "PaperlessBilling","PaymentMethod"]
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for ax, col in zip(axes, cat_cols):
    ct = df.groupby(col)["Churn"].agg(["sum","count"])
    ct["rate"] = ct["sum"] / ct["count"] * 100
    ct = ct.sort_values("rate", ascending=False)
    bars = ax.bar(ct.index, ct["rate"], color=C["blue"], alpha=0.85, width=0.5)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(col); ax.set_ylabel("Churn %")
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                f"{b.get_height():.1f}%", ha="center", fontsize=9, color=C["text"])
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("Churn Rate Across Key Categorical Features", fontsize=14, color=C["text"])
plt.tight_layout()
plt.savefig(VIZ / "07_categorical_churn_rates.png"); plt.close()
log.info("  ✓ 07_categorical_churn_rates.png")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

log.info("\n" + "="*72)
log.info("TASK 3 — FEATURE ENGINEERING")
log.info("="*72)

df_fe = df_clean.copy()
df_fe["Churn"] = (df_fe["Churn"] == "Yes").astype(int) if df_fe["Churn"].dtype == object else df_fe["Churn"]

# ── Domain-driven engineered features ─────────────────────────────────────────
df_fe["charges_per_tenure"]     = df_fe["MonthlyCharges"] / (df_fe["tenure"] + 1)
df_fe["is_new_customer"]        = (df_fe["tenure"] <= 3).astype(int)
df_fe["is_long_term"]           = (df_fe["tenure"] >= 48).astype(int)
df_fe["has_fiber"]              = (df_fe["InternetService"] == "Fiber optic").astype(int)
df_fe["no_protection_bundle"]   = (
    (df_fe["OnlineSecurity"] == "No") &
    (df_fe["TechSupport"] == "No") &
    (df_fe["DeviceProtection"] == "No")
).astype(int)
df_fe["total_services"]         = (
    (df_fe["PhoneService"] == "Yes").astype(int) +
    (df_fe["MultipleLines"] == "Yes").astype(int) +
    (df_fe["InternetService"] != "No").astype(int) +
    (df_fe["OnlineSecurity"] == "Yes").astype(int) +
    (df_fe["OnlineBackup"] == "Yes").astype(int) +
    (df_fe["DeviceProtection"] == "Yes").astype(int) +
    (df_fe["TechSupport"] == "Yes").astype(int) +
    (df_fe["StreamingTV"] == "Yes").astype(int) +
    (df_fe["StreamingMovies"] == "Yes").astype(int)
)
df_fe["high_monthly_charges"]   = (df_fe["MonthlyCharges"] > 70).astype(int)
df_fe["month_to_month_flag"]    = (df_fe["Contract"] == "Month-to-month").astype(int)

engineered = ["charges_per_tenure","is_new_customer","is_long_term",
              "has_fiber","no_protection_bundle","total_services",
              "high_monthly_charges","month_to_month_flag"]
log.info(f"Engineered {len(engineered)} domain features: {engineered}")

# ── Encoding categorical variables ────────────────────────────────────────────
# Binary columns  → 0/1 map
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
binary_cols = ["gender","Partner","Dependents","PhoneService",
               "PaperlessBilling"]
for col in binary_cols:
    df_fe[col] = df_fe[col].map(binary_map)

# Multi-class columns → one-hot (drop_first avoids dummy trap)
multi_cat = ["MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
             "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
             "Contract","PaymentMethod"]
df_fe = pd.get_dummies(df_fe, columns=multi_cat, drop_first=True)
log.info(f"Post-encoding shape: {df_fe.shape}")

# ── Feature / Target split ────────────────────────────────────────────────────
TARGET = "Churn"
FEATURES = [c for c in df_fe.columns if c != TARGET]
X = df_fe[FEATURES].astype(float)
y = df_fe[TARGET].astype(int)
log.info(f"Features: {len(FEATURES)}  |  Samples: {len(y)}")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 2 — TRAIN / TEST SPLIT  (80/20 stratified)
# ══════════════════════════════════════════════════════════════════════════════

log.info("\n" + "="*72)
log.info("TASK 2 — TRAIN / TEST SPLIT")
log.info("="*72)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
log.info(f"Train : {X_train.shape}  |  Test : {X_test.shape}")
log.info(f"Train churn rate : {y_train.mean()*100:.2f}%")
log.info(f"Test  churn rate : {y_test.mean()*100:.2f}%")

# ── Feature scaling ───────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ══════════════════════════════════════════════════════════════════════════════
# TASK 4 — MODEL SELECTION  (benchmark 8 classifiers on 5-fold CV)
# ══════════════════════════════════════════════════════════════════════════════

log.info("\n" + "="*72)
log.info("TASK 4 — MODEL SELECTION (8 classifiers, 5-fold Stratified CV)")
log.info("="*72)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

CANDIDATES = {
    "Logistic Regression":      LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree":             DecisionTreeClassifier(random_state=SEED),
    "Random Forest":             RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "Gradient Boosting":         GradientBoostingClassifier(n_estimators=200, random_state=SEED),
    "AdaBoost":                  AdaBoostClassifier(n_estimators=100, random_state=SEED),
    "K-Nearest Neighbours":      KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes":               GaussianNB(),
    "SVM (RBF)":                 SVC(probability=True, random_state=SEED),
}

results = {}
for name, model in CANDIDATES.items():
    t0 = time.time()
    scores = cross_val_score(model, X_train_sc, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)
    elapsed = time.time() - t0
    results[name] = {
        "mean_auc": scores.mean(),
        "std_auc":  scores.std(),
        "scores":   scores.tolist(),
        "time_s":   round(elapsed, 2),
    }
    log.info(f"  {name:<28} ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}  ({elapsed:.1f}s)")

# rank by mean AUC
results_df = pd.DataFrame(results).T.sort_values("mean_auc", ascending=False)
BEST_NAME  = results_df.index[0]
log.info(f"\n  ★ Best model selected: {BEST_NAME}  (AUC={results_df.loc[BEST_NAME,'mean_auc']:.4f})")

# ── Model Comparison Chart ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
names  = results_df.index.tolist()
means  = [results[n]["mean_auc"] for n in names]
stds   = [results[n]["std_auc"]  for n in names]
colors = [C["gold"] if n == BEST_NAME else C["blue"] for n in names]

bars = ax.barh(names, means, xerr=stds, color=colors, height=0.55,
               error_kw={"ecolor": C["muted"], "capsize": 4})
ax.set_xlim(0.60, 1.0)
ax.set_xlabel("Mean ROC-AUC (5-fold Stratified CV)")
ax.set_title("Model Benchmarking — Cross-Validated ROC-AUC")
for bar, v, s in zip(bars, means, stds):
    ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
            f"{v:.4f} ± {s:.4f}", va="center", fontsize=9, color=C["text"])
ax.axvline(max(means), color=C["gold"], ls="--", lw=1.2, alpha=0.6)
ax.grid(axis="x", alpha=0.3)
gold_patch  = mpatches.Patch(color=C["gold"],  label="Best model")
blue_patch  = mpatches.Patch(color=C["blue"],  label="Other models")
ax.legend(handles=[gold_patch, blue_patch], loc="lower right")
plt.tight_layout()
plt.savefig(VIZ / "08_model_benchmarking.png"); plt.close()
log.info("  ✓ 08_model_benchmarking.png")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 5 — MODEL TRAINING + HYPERPARAMETER TUNING (GridSearchCV)
# ══════════════════════════════════════════════════════════════════════════════

log.info("\n" + "="*72)
log.info("TASK 5 — MODEL TRAINING + HYPERPARAMETER TUNING")
log.info("="*72)

# Hyperparameter grids for our top two contenders
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators":       [100, 200, 300],
        "max_depth":          [None, 10, 20],
        "min_samples_split":  [2, 5, 10],
        "min_samples_leaf":   [1, 2, 4],
        "max_features":       ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "n_estimators":       [100, 200],
        "learning_rate":      [0.05, 0.1, 0.2],
        "max_depth":          [3, 5, 7],
        "subsample":          [0.8, 1.0],
        "min_samples_split":  [2, 5],
    },
    "Logistic Regression": {
        "C":                  [0.01, 0.1, 1, 10, 100],
        "penalty":            ["l1", "l2"],
        "solver":             ["liblinear"],
    },
}

if BEST_NAME in PARAM_GRIDS:
    log.info(f"  Running GridSearchCV for {BEST_NAME} ...")
    grid = GridSearchCV(
        CANDIDATES[BEST_NAME],
        PARAM_GRIDS[BEST_NAME],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid.fit(X_train_sc, y_train)
    best_model = grid.best_estimator_
    log.info(f"  Best params  : {grid.best_params_}")
    log.info(f"  Best CV AUC  : {grid.best_score_:.4f}")
else:
    log.info(f"  No param grid for {BEST_NAME}; training with defaults.")
    best_model = CANDIDATES[BEST_NAME]
    best_model.fit(X_train_sc, y_train)

# Also train all candidates on full training set (for comparison plots)
trained_models = {}
for name, model in CANDIDATES.items():
    model.fit(X_train_sc, y_train)
    trained_models[name] = model

log.info("  ✓ All candidate models trained on full training set.")

# ══════════════════════════════════════════════════════════════════════════════
# TASK 6 — MODEL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

log.info("\n" + "="*72)
log.info("TASK 6 — MODEL EVALUATION")
log.info("="*72)

def full_metrics(model, X_tr, y_tr, X_te, y_te, name="Model"):
    y_pred  = model.predict(X_te)
    y_prob  = model.predict_proba(X_te)[:, 1]
    m = dict(
        name      = name,
        accuracy  = accuracy_score(y_te, y_pred),
        precision = precision_score(y_te, y_pred, zero_division=0),
        recall    = recall_score(y_te, y_pred, zero_division=0),
        f1        = f1_score(y_te, y_pred, zero_division=0),
        roc_auc   = roc_auc_score(y_te, y_prob),
        pr_auc    = average_precision_score(y_te, y_prob),
        mcc       = matthews_corrcoef(y_te, y_pred),
        train_acc = accuracy_score(y_tr, model.predict(X_tr)),
    )
    return m, y_pred, y_prob

# Evaluate best model
metrics, y_pred, y_prob = full_metrics(
    best_model, X_train_sc, y_train, X_test_sc, y_test, BEST_NAME
)
log.info(f"\n  ── {BEST_NAME} Test-Set Metrics ──")
for k, v in metrics.items():
    if isinstance(v, float): log.info(f"    {k:<14}: {v:.4f}")

log.info("\n  Classification Report:\n" + classification_report(y_test, y_pred,
    target_names=["Retained", "Churned"]))

# Evaluate all models for comparison table
all_metrics = []
for name, model in trained_models.items():
    m, _, _ = full_metrics(model, X_train_sc, y_train, X_test_sc, y_test, name)
    all_metrics.append(m)
metrics_df = pd.DataFrame(all_metrics).sort_values("roc_auc", ascending=False)
metrics_df.to_csv(OUT / "all_model_metrics.csv", index=False)
log.info(f"  ✓ all_model_metrics.csv saved.")

# ── VIZ-9: Confusion Matrix ────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Retained","Churned"],
            yticklabels=["Retained","Churned"],
            linewidths=1, linecolor=C["border"], ax=ax,
            annot_kws={"size": 16, "weight": "bold"})
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
ax.set_title(f"Confusion Matrix — {BEST_NAME}")
tn, fp, fn, tp = cm.ravel()
ax.text(1.02, 0.5, f"TN={tn}\nFP={fp}\nFN={fn}\nTP={tp}",
        transform=ax.transAxes, va="center", fontsize=10, color=C["muted"])
plt.tight_layout()
plt.savefig(VIZ / "09_confusion_matrix.png"); plt.close()
log.info("  ✓ 09_confusion_matrix.png")

# ── VIZ-10: ROC Curves (all models) ──────────────────────────────────────────
model_colors = [C["red"],C["blue"],C["gold"],C["teal"],C["purple"],C["green"],C["muted"],"#FF8C00"]
fig, ax = plt.subplots(figsize=(9, 7))
ax.plot([0,1],[0,1], ls="--", color=C["muted"], lw=1.2, label="Random Chance (AUC=0.50)")
for (name, model), color in zip(trained_models.items(), model_colors):
    prob = model.predict_proba(X_test_sc)[:,1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    lw = 2.5 if name == BEST_NAME else 1.2
    ls = "-"  if name == BEST_NAME else "--"
    ax.plot(fpr, tpr, color=color, lw=lw, ls=ls, label=f"{name}  (AUC={auc_val:.3f})")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models vs. Random Baseline")
ax.legend(loc="lower right", fontsize=8, framealpha=0.4)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ / "10_roc_curves_all.png"); plt.close()
log.info("  ✓ 10_roc_curves_all.png")

# ── VIZ-11: Precision-Recall Curve ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
baseline = y_test.mean()
ax.axhline(baseline, ls="--", color=C["muted"], lw=1.2, label=f"No Skill (P={baseline:.2f})")
for (name, model), color in zip(trained_models.items(), model_colors):
    prob = model.predict_proba(X_test_sc)[:,1]
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    lw = 2.5 if name == BEST_NAME else 1.2
    ls = "-"  if name == BEST_NAME else "--"
    ax.plot(rec, prec, color=color, lw=lw, ls=ls, label=f"{name}  (AP={ap:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — All Models")
ax.legend(fontsize=8, framealpha=0.4)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ / "11_precision_recall_curves.png"); plt.close()
log.info("  ✓ 11_precision_recall_curves.png")

# ── VIZ-12: Feature Importance (Random Forest) ───────────────────────────────
rf_model = trained_models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
top20 = importances.head(20)

fig, ax = plt.subplots(figsize=(9, 8))
colors_fi = [C["gold"] if i < 5 else C["blue"] for i in range(len(top20))]
ax.barh(top20.index[::-1], top20.values[::-1], color=colors_fi[::-1], height=0.65)
ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
ax.set_title("Top 20 Feature Importances — Random Forest")
ax.grid(axis="x", alpha=0.3)
gold_patch = mpatches.Patch(color=C["gold"], label="Top 5 features")
blue_patch = mpatches.Patch(color=C["blue"], label="Other features")
ax.legend(handles=[gold_patch, blue_patch])
plt.tight_layout()
plt.savefig(VIZ / "12_feature_importance.png"); plt.close()
log.info("  ✓ 12_feature_importance.png")

# ── VIZ-13: Learning Curve ────────────────────────────────────────────────────
train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_sc, y_train,
    cv=cv, scoring="roc_auc",
    train_sizes=np.linspace(0.10, 1.0, 10),
    n_jobs=-1
)
fig, ax = plt.subplots(figsize=(9, 6))
ax.fill_between(train_sizes, train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1), alpha=0.15, color=C["blue"])
ax.fill_between(train_sizes, val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1), alpha=0.15, color=C["teal"])
ax.plot(train_sizes, train_scores.mean(1), "o-", color=C["blue"], lw=2, label="Training AUC")
ax.plot(train_sizes, val_scores.mean(1),   "s-", color=C["teal"], lw=2, label="CV Validation AUC")
ax.set_xlabel("Training Set Size"); ax.set_ylabel("ROC-AUC Score")
ax.set_title(f"Learning Curve — {BEST_NAME}")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ / "13_learning_curve.png"); plt.close()
log.info("  ✓ 13_learning_curve.png")

# ── VIZ-14: Metrics Comparison Heatmap ───────────────────────────────────────
heat_df = metrics_df.set_index("name")[
    ["accuracy","precision","recall","f1","roc_auc","pr_auc","mcc"]
].astype(float)
fig, ax = plt.subplots(figsize=(12, 7))
sns.heatmap(heat_df, annot=True, fmt=".3f", cmap="YlGn",
            linewidths=0.5, linecolor=C["border"], ax=ax,
            annot_kws={"size": 10})
ax.set_title("All Models — Test-Set Metric Comparison")
plt.xticks(rotation=20, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(VIZ / "14_metrics_heatmap.png"); plt.close()
log.info("  ✓ 14_metrics_heatmap.png")

# ── VIZ-15: Churn Probability Histogram ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
proba_retained = y_prob[y_test == 0]
proba_churned  = y_prob[y_test == 1]
ax.hist(proba_retained, bins=40, alpha=0.75, color=C["teal"], label="Retained (actual)", density=True)
ax.hist(proba_churned,  bins=40, alpha=0.75, color=C["red"],  label="Churned (actual)",  density=True)
ax.axvline(0.5, color=C["gold"], ls="--", lw=1.5, label="Decision threshold (0.5)")
ax.set_xlabel("Predicted Churn Probability"); ax.set_ylabel("Density")
ax.set_title(f"Churn Probability Distribution — {BEST_NAME}")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ / "15_churn_probability_histogram.png"); plt.close()
log.info("  ✓ 15_churn_probability_histogram.png")

# ── VIZ-16: Threshold Sensitivity Analysis ────────────────────────────────────
thresholds = np.linspace(0.1, 0.9, 80)
f1_scores, prec_scores, rec_scores = [], [], []
for t in thresholds:
    preds = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y_test, preds, zero_division=0))
    prec_scores.append(precision_score(y_test, preds, zero_division=0))
    rec_scores.append(recall_score(y_test, preds, zero_division=0))

best_t = thresholds[np.argmax(f1_scores)]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, f1_scores,   color=C["gold"],  lw=2, label="F1 Score")
ax.plot(thresholds, prec_scores, color=C["blue"],  lw=2, label="Precision")
ax.plot(thresholds, rec_scores,  color=C["teal"],  lw=2, label="Recall")
ax.axvline(best_t, color=C["red"], ls="--", lw=1.5, label=f"Optimal threshold ({best_t:.2f})")
ax.axvline(0.5,    color=C["muted"], ls=":", lw=1.2, label="Default threshold (0.50)")
ax.set_xlabel("Classification Threshold"); ax.set_ylabel("Score")
ax.set_title("Threshold Sensitivity Analysis — Precision / Recall / F1")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(VIZ / "16_threshold_analysis.png"); plt.close()
log.info(f"  ✓ 16_threshold_analysis.png  |  Optimal threshold = {best_t:.2f}")

# ── Save artefacts ─────────────────────────────────────────────────────────────
with open(MDL / "best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open(MDL / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(MDL / "feature_names.json", "w") as f:
    json.dump(FEATURES, f, indent=2)
log.info("  ✓ Model artefacts saved to /models/")

# ── Final summary ──────────────────────────────────────────────────────────────
log.info("\n" + "="*72)
log.info("PIPELINE COMPLETE — SUMMARY")
log.info("="*72)
log.info(f"  Best Model      : {BEST_NAME}")
log.info(f"  Test Accuracy   : {metrics['accuracy']*100:.2f}%")
log.info(f"  Test ROC-AUC    : {metrics['roc_auc']:.4f}")
log.info(f"  Test F1 Score   : {metrics['f1']:.4f}")
log.info(f"  Test Recall     : {metrics['recall']:.4f}  (% churners caught)")
log.info(f"  Test Precision  : {metrics['precision']:.4f}")
log.info(f"  Optimal Thresh  : {best_t:.2f}")
log.info(f"  Visualisations  : {len(list(VIZ.glob('*.png')))} plots saved to /visualizations/")
log.info("="*72)

# Save metrics JSON for report generation
summary = {
    "best_model":      BEST_NAME,
    "test_accuracy":   round(metrics["accuracy"],  4),
    "test_roc_auc":    round(metrics["roc_auc"],   4),
    "test_f1":         round(metrics["f1"],        4),
    "test_recall":     round(metrics["recall"],    4),
    "test_precision":  round(metrics["precision"], 4),
    "test_pr_auc":     round(metrics["pr_auc"],    4),
    "mcc":             round(metrics["mcc"],       4),
    "train_accuracy":  round(metrics["train_acc"], 4),
    "optimal_threshold": round(float(best_t),      2),
    "n_features":      len(FEATURES),
    "dataset_rows":    int(len(df)),
    "churn_rate_pct":  round(float(df["Churn"].mean()*100), 2),
    "confusion_matrix": cm.tolist(),
    "all_models":      metrics_df[["name","accuracy","precision","recall",
                                   "f1","roc_auc"]].to_dict(orient="records"),
}
with open(OUT / "pipeline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
log.info("  ✓ pipeline_summary.json saved.")
print("\nAll done! ✓")
