import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dccm_features(base_dir: str) -> pd.DataFrame:
    features_path = os.path.join(base_dir, "code/3d_structure/dccm_features_analysis.csv")
    df = pd.read_csv(features_path)
    # Normalize key naming
    df = df.rename(columns={"state": "State", "protein": "Protein"})
    # Keep a curated subset of numeric features to avoid leakage from identifiers
    numeric_cols = [
        "mean_correlation",
        "std_correlation",
        "median_correlation",
        "max_correlation",
        "min_correlation",
        "network_density",
        "max_hub_score",
        "mean_hub_score",
        "hub_std",
    ]
    df = df[[*numeric_cols, "Protein", "State"]].copy()
    return df


def load_patterns(base_dir: str) -> pd.DataFrame:
    path = os.path.join(base_dir, "code/3d_structure/motor_tubulin_correlation_patterns.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["protein", "state", "overall_pattern"])  # optional
    patterns = pd.read_csv(path)
    patterns = patterns.rename(columns={"protein": "Protein", "state": "State"})
    return patterns[["Protein", "State", "overall_pattern"]]


def engineer_state_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot state-specific features into wide format and add deltas.

    Output index per protein with columns like feature|ADP, feature|ATP, feature|APO, and deltas.
    """
    numeric_cols = [c for c in df.columns if c not in ("Protein", "State")]
    wide = df.pivot_table(index="Protein", columns="State", values=numeric_cols)
    # Flatten MultiIndex columns
    wide.columns = [f"{c}|{s}" for c, s in wide.columns]
    wide = wide.reset_index()

    # Create useful deltas (ATP-ADP, APO-ADP)
    for feat in [
        "mean_correlation",
        "std_correlation",
        "network_density",
        "max_hub_score",
        "mean_hub_score",
        "hub_std",
    ]:
        for pair in [("ATP", "ADP"), ("APO", "ADP")]:
            a, b = pair
            col_a = f"{feat}|{a}"
            col_b = f"{feat}|{b}"
            if col_a in wide.columns and col_b in wide.columns:
                wide[f"delta_{feat}_{a}_minus_{b}"] = wide[col_a] - wide[col_b]
    # State-averaged aggregates
    for feat in [
        "mean_correlation",
        "std_correlation",
        "network_density",
        "max_hub_score",
        "mean_hub_score",
        "hub_std",
    ]:
        cols = [c for c in wide.columns if c.startswith(f"{feat}|")]
        if cols:
            wide[f"{feat}|meanAcrossStates"] = wide[cols].mean(axis=1)
            wide[f"{feat}|stdAcrossStates"] = wide[cols].std(axis=1)
    return wide


def integrate_patterns(wide: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    if patterns.empty:
        return wide
    # Aggregate pattern per protein across states (majority/string mode)
    agg = (
        patterns.groupby("Protein")["overall_pattern"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    return wide.merge(agg, on="Protein", how="left")


def load_behavior_labels(base_dir: str) -> pd.DataFrame:
    labels_path = os.path.join(base_dir, "paper/motor_clusters_analysis.csv")
    df = pd.read_csv(labels_path)
    # Collapse to per-protein label by majority vote across concentrations
    if "behavioral_class" not in df.columns:
        raise RuntimeError("Expected 'behavioral_class' in motor_clusters_analysis.csv")
    votes = (
        df.groupby("protein")["behavioral_class"]
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    votes = votes.rename(columns={"protein": "RawProtein", "behavioral_class": "behavioral_class"})
    return votes


def build_protein_mapping() -> Dict[str, str]:
    # Map paper protein names to DCCM protein keys
    mapping = {
        # Single-letter motors
        "A": "a",
        "B": "b",
        "C": "c",
        "D": "d",
        "E": "e",
        "F": "f",
        "G": "g",
        "H": "h",
        # Species / named backbones
        "ThTr": "thtr",
        "NaGr": "nagr",
        "DiPu": "dipu",
        "AcSu": "acsu",
        "AcSu2": "acsu2",
        "Kif5a": "kif5a",
        "HeAl": "heal",
        "Unc": "unc",
        "TiLa": "tila",
        "AdPa": "adpa",
        # Others that appear in paper CSV but not necessarily in DCCM set
        "BleSto": "blesto",  # if absent, will drop later
        "K401": "a",  # legacy alias for A
    }
    return mapping


def align_labels_to_features(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    mapping = build_protein_mapping()
    labels["Protein"] = labels["RawProtein"].map(mapping).fillna(labels["RawProtein"].str.lower())
    aligned = labels.merge(features, on="Protein", how="inner")
    # keep only rows with a label and at least some features
    aligned = aligned.dropna(subset=["behavioral_class"]) 
    return aligned


def train_cv_classifier(df: pd.DataFrame, out_dir: str) -> Dict[str, float]:
    y = df["behavioral_class"].astype(str).values
    # Candidate feature columns
    exclude = {"Protein", "behavioral_class", "RawProtein"}
    categorical_cols = [c for c in df.columns if c == "overall_pattern"]
    numeric_cols = [c for c in df.columns if c not in exclude and c not in categorical_cols]

    X = df[numeric_cols + categorical_cols].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="drop",
    )

    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_true, y_pred = [], []

    cm_labels = sorted(np.unique(y))
    cms = []
    for train_idx, test_idx in skf.split(X, y):
        pipe.fit(X.iloc[train_idx], y[train_idx])
        pred = pipe.predict(X.iloc[test_idx])
        y_true.extend(y[test_idx])
        y_pred.extend(pred)
        cms.append(confusion_matrix(y[test_idx], pred, labels=cm_labels))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    # Aggregate confusion matrices
    cm_sum = np.sum(cms, axis=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_sum, annot=True, fmt="d", cmap="Blues", xticklabels=cm_labels, yticklabels=cm_labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Behavior class CV confusion matrix")
    ensure_dir(out_dir)
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Permutation baseline
    rng = np.random.default_rng(42)
    perm_acc = []
    for _ in range(100):
        perm_y = rng.permutation(y)
        preds = []
        for train_idx, test_idx in skf.split(X, perm_y):
            pipe.fit(X.iloc[train_idx], perm_y[train_idx])
            preds.extend(pipe.predict(X.iloc[test_idx]))
        perm_acc.append(accuracy_score(perm_y, np.array(preds)))

    result = {
        "cv_accuracy": float(acc),
        "cv_macro_f1": float(f1),
        "perm_acc_mean": float(np.mean(perm_acc)),
        "perm_acc_std": float(np.std(perm_acc, ddof=1)),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "classes": cm_labels,
        "report": classification_report(y_true, y_pred, output_dict=False),
    }

    with open(os.path.join(out_dir, "cv_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Save a readable text report
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(result["report"])  # type: ignore[arg-type]

    return result


def correlate_kinetics(labels_csv: str, out_dir: str, feature_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    # Aggregate per protein metrics (median across conditions)
    agg = (
        df.groupby("protein")[
            [
                "mean_velocity",
                "max_velocity",
                "max_continuous_motion_h",
            ]
        ]
        .median()
        .reset_index()
    )
    mapping = build_protein_mapping()
    agg["Protein"] = agg["protein"].map(mapping).fillna(agg["protein"].str.lower())
    merged = agg.merge(feature_df, on="Protein", how="inner")

    # Choose a compact set of interpretable features for correlation
    feat_cols = [
        "mean_correlation|meanAcrossStates",
        "std_correlation|meanAcrossStates",
        "network_density|meanAcrossStates",
        "max_hub_score|meanAcrossStates",
        "mean_hub_score|meanAcrossStates",
        "hub_std|meanAcrossStates",
    ]
    records: List[Dict[str, float]] = []
    for target in ["mean_velocity", "max_velocity", "max_continuous_motion_h"]:
        for feat in feat_cols:
            if feat not in merged.columns:
                continue
            x = merged[feat].values
            y = merged[target].values
            if np.all(np.isnan(x)) or np.all(np.isnan(y)):
                continue
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 4:
                continue
            rho, p = stats.spearmanr(x[mask], y[mask])
            records.append({"feature": feat, "target": target, "rho": float(rho), "p": float(p)})
    corr_df = pd.DataFrame.from_records(records)
    ensure_dir(out_dir)
    corr_df.to_csv(os.path.join(out_dir, "spearman_correlations.csv"), index=False)
    return corr_df


def summarize_patterns(patterns: pd.DataFrame, out_dir: str) -> pd.DataFrame:
    if patterns.empty:
        return pd.DataFrame()
    counts = (
        patterns.groupby(["State", "overall_pattern"]).size().reset_index(name="count")
    )
    total_by_state = counts.groupby("State")["count"].sum().reset_index(name="total")
    freq = counts.merge(total_by_state, on="State")
    freq["freq"] = freq["count"] / freq["total"].replace(0, np.nan)

    plt.figure(figsize=(7, 4))
    sns.barplot(data=freq, x="State", y="freq", hue="overall_pattern")
    plt.ylabel("Frequency")
    plt.title("A/B–C/D coupling pattern frequency by state")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pattern_frequencies.png"), dpi=200)
    plt.close()
    freq.to_csv(os.path.join(out_dir, "pattern_frequencies.csv"), index=False)
    return freq


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base_dir, "paper", "validation")
    ensure_dir(out_dir)

    # Load
    dccm_df = load_dccm_features(base_dir)
    patterns = load_patterns(base_dir)

    # Engineer features
    features_wide = engineer_state_features(dccm_df)
    features_wide = integrate_patterns(features_wide, patterns)

    # Labels
    labels = load_behavior_labels(base_dir)
    aligned = align_labels_to_features(labels, features_wide)
    aligned.to_csv(os.path.join(out_dir, "aligned_features_labels.csv"), index=False)

    # Classifier
    metrics = train_cv_classifier(aligned, out_dir)

    # Correlations
    corr_df = correlate_kinetics(os.path.join(base_dir, "paper/motor_clusters_analysis.csv"), out_dir, features_wide)

    # Patterns summary
    summarize_patterns(patterns, out_dir)

    # Quick Markdown summary
    summary_md = os.path.join(out_dir, "SUMMARY.md")
    with open(summary_md, "w") as f:
        f.write("# DCCM Validation Summary\n\n")
        f.write(f"Samples used: {metrics['n_samples']} | Features: {metrics['n_features']}\n\n")
        f.write(f"CV accuracy: {metrics['cv_accuracy']:.3f}\n\n")
        f.write(f"CV macro-F1: {metrics['cv_macro_f1']:.3f}\n\n")
        f.write(
            f"Permutation acc mean ± sd: {metrics['perm_acc_mean']:.3f} ± {metrics['perm_acc_std']:.3f}\n\n"
        )
        if not corr_df.empty:
            top = (
                corr_df.reindex(corr_df["rho"].abs().sort_values(ascending=False).index)
                .head(8)
                .reset_index(drop=True)
            )
            f.write("Top Spearman correlations (abs):\n\n")
            f.write(top.to_markdown(index=False))


if __name__ == "__main__":
    main()


