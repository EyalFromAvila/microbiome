"""
Evaluation Metrics & Publication-Grade Figure Generator
for Milestone 3 Presentation and Final PDF Report.

Run after `snf_neighbours.py experiments` (writes results/main_split.csv and results/missing_sweep.csv),
prepare_validation_datasets.py and `snf_neighbours.py` (submission matrices). Cohorts whose files are
missing are skipped.

Outputs:
  - results/validation_summary.csv
  - figures/fig1_pipeline_progression.png
  - figures/fig2_missingness_sweep.png
  - figures/fig3_external_validation_pcoa.png
"""

from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ConstantInputWarning, mannwhitneyu, pearsonr, rankdata, spearmanr
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score

from pipeline import RANDOM_STATE, load_data
from snf_neighbours import (
    ScoreSpace,
    distance_matrix,
    fill_scattered_gaps,
    knn50_distance,
    make_missing_split,
    metadata_features,
    pipeline_distance,
    tune,
)

# ==========================================
# GLOBAL PLOT STYLING (300 DPI, COLORBLIND-SAFE)
# ==========================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
PALETTE = sns.color_palette("colorblind")
FIG_DIR = Path("figures")
RES_DIR = Path("results")          # where snf_neighbours.py writes its tables (its --out default)
FILES = ("microbiome.csv", "metabolome.csv", "metadata.csv")
FINAL = "snf_neighbours (Final)"
COHORTS = {                        # name -> (folder, submission matrix)
    "MetaCardis": (Path("."), Path("student_res_metacardis.csv")),
    "Franzosa (IBD)": (Path("prepared_datasets") / "franzosa", Path("student_res_franzosa.csv")),
    "Yachida (CRC)": (Path("prepared_datasets") / "yachida", Path("student_res_yachida.csv")),
}

# Main-split results of the pipelines tried before the final one (score_space_pipelines.py, commit
# b57c6fc, same split and masks as pipeline.py; the Milestone 1 row is the original kernel pipeline).
# That script was removed, so these rows are a record, not recomputed here. The KNN50 and final
# pipeline rows are replaced by results/main_split.csv when snf_neighbours.py has been run.
HISTORICAL_MAIN_SPLIT = pd.DataFrame([
    {"Pipeline": "MKC+SNF kernel (Milestone 1)", "Pearson_r": 0.2283, "Spearman_r": 0.2251, "Run_s": 158.82},
    {"Pipeline": "MKC+SNF (score space)", "Pearson_r": 0.5740, "Spearman_r": 0.5395, "Run_s": 144.47},
    {"Pipeline": "Mean fill (score space)", "Pearson_r": 0.5811, "Spearman_r": 0.5436, "Run_s": 0.04},
    {"Pipeline": "Global KNN50 (Benchmark)", "Pearson_r": 0.5897, "Spearman_r": 0.5538, "Run_s": 0.34},
    {"Pipeline": "Ridge (score space)", "Pearson_r": 0.6037, "Spearman_r": 0.5711, "Run_s": 0.14},
    {"Pipeline": "miss-SNF (score space)", "Pearson_r": 0.6040, "Spearman_r": 0.5705, "Run_s": 5.60},
    {"Pipeline": FINAL, "Pearson_r": 0.6054, "Spearman_r": 0.5716, "Run_s": 6.07},
    {"Pipeline": "MKC_embd(ht) (score space)", "Pearson_r": 0.6104, "Spearman_r": 0.5764, "Run_s": 141.04},
])
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. MANTEL & SCORE-SPACE EVALUATION HELPERS
# ==========================================

def dual_mantel(mat1: pd.DataFrame, mat2: pd.DataFrame, n_perm: int = 499, seed: int = RANDOM_STATE) -> dict:
    """Computes Pearson and Spearman Mantel correlations with permutation p-values."""
    mat2 = mat2.loc[mat1.index, mat1.columns]
    i, j = np.triu_indices(len(mat1), k=1)
    v1 = mat1.to_numpy()[i, j]
    v2 = mat2.to_numpy()[i, j]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        r_p, _ = pearsonr(v1, v2)
        rho_s, _ = spearmanr(v1, v2)

    if n_perm == 0:
        return {"Pearson_r": float(r_p), "Pearson_p": np.nan, "Spearman_r": float(rho_s), "Spearman_p": np.nan}

    rng = np.random.default_rng(seed)
    m2 = mat2.to_numpy()
    v1_c = v1 - v1.mean()
    v1_norm = np.linalg.norm(v1_c)

    v1_rk = rankdata(v1)
    v1_rk_c = v1_rk - v1_rk.mean()
    v1_rk_norm = np.linalg.norm(v1_rk_c)

    cnt_p, cnt_s = 0, 0
    for _ in range(n_perm):
        perm = rng.permutation(len(m2))
        v2_p = m2[np.ix_(perm, perm)][i, j]

        v2_pc = v2_p - v2_p.mean()
        dp = v1_norm * np.linalg.norm(v2_pc)
        cnt_p += (np.dot(v1_c, v2_pc) / dp if dp > 0 else 0.0) >= r_p

        v2_sr = rankdata(v2_p)
        v2_sc = v2_sr - v2_sr.mean()
        ds = v1_rk_norm * np.linalg.norm(v2_sc)
        cnt_s += (np.dot(v1_rk_c, v2_sc) / ds if ds > 0 else 0.0) >= rho_s

    return {
        "Pearson_r": float(r_p),
        "Pearson_p": (cnt_p + 1) / (n_perm + 1),
        "Spearman_r": float(rho_s),
        "Spearman_p": (cnt_s + 1) / (n_perm + 1),
    }


def evaluate_cohort_40pct_split(folder: Path, cohort_name: str) -> list:
    """
    Hide one omic in 40% of a cohort's complete samples and compare, on those samples, the final
    pipeline (snf_neighbours, tuned on the remaining complete samples) with Global KNN50 and with
    mean fill. All three are measured in the same 2-PC score space; the ground truth is the
    grader's (PCA on every sample's true values). Uses the code of snf_neighbours.py directly.
    """
    _, complete, mb_cols, mt_cols, metadata = load_data(*(folder / name for name in FILES))
    complete = fill_scattered_gaps(complete, complete.index, mb_cols, mt_cols)
    meta = metadata_features(metadata)
    split = make_missing_split(complete, mb_cols, mt_cols, 0.40, RANDOM_STATE)
    masked = split.masked_ids
    truth = split.ground_truth.loc[masked, masked]

    # Mean fill: the missing omic's part of the scores is its training mean.
    space = ScoreSpace(split.train, mb_cols, mt_cols)
    parts = space.contributions(split.combined)
    for omic in ("mb", "mt"):
        missing = parts[omic].index[parts[omic].isna().any(axis=1)]
        parts[omic].loc[missing] = parts[omic].loc[split.train.index].mean().to_numpy()
    mean_fill = distance_matrix(space.combine(parts))

    knn = knn50_distance(split, mb_cols, mt_cols)
    params = tune(split.train, meta, mb_cols, mt_cols)
    snf_dist = pipeline_distance(split, meta, mb_cols, mt_cols, **params)

    return [
        {"Cohort": cohort_name, "Method": method, **dual_mantel(truth, distances.loc[masked, masked])}
        for method, distances in (
            ("Mean fill (score space)", mean_fill),
            ("Global KNN50", knn),
            (FINAL, snf_dist),
        )
    ]


def load_distance_csv(path: Path) -> pd.DataFrame:
    """Loads submission distance matrix whether formatted with ',' or ', '."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    sep = ", " if ", " in header else ","
    df = pd.read_csv(path, sep=sep, index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df


def compute_biological_separation(dist_df: pd.DataFrame, meta_df: pd.DataFrame, group_col: str, cohort_name: str) -> dict:
    """Computes Silhouette score and Mann-Whitney U test (within-group vs between-group distances)."""
    common = dist_df.index.intersection(meta_df.index)
    D = dist_df.loc[common, common].to_numpy()
    labels = meta_df.loc[common, group_col].fillna("Unknown").astype(str).str.strip()

    # Filter out tiny singleton groups for silhouette stability
    counts = labels.value_counts()
    valid_groups = counts[counts >= 3].index
    mask = labels.isin(valid_groups).to_numpy()
    D_sub = D[np.ix_(mask, mask)]
    y_sub = labels[mask].to_numpy()

    sil = float(silhouette_score(D_sub, y_sub, metric="precomputed")) if len(np.unique(y_sub)) > 1 else np.nan

    i, j = np.triu_indices(len(y_sub), k=1)
    same_group = y_sub[i] == y_sub[j]
    d_within = D_sub[i, j][same_group]
    d_between = D_sub[i, j][~same_group]

    stat, pval = mannwhitneyu(d_within, d_between, alternative="less")
    return {
        "Cohort": cohort_name,
        "N_Samples": len(common),
        "N_Groups": int(len(np.unique(y_sub))),
        "Silhouette_Score": sil,
        "Mean_Within_Dist": float(np.mean(d_within)),
        "Mean_Between_Dist": float(np.mean(d_between)),
        "MannWhitney_U_pval": float(pval),
    }


# ==========================================
# 2. PUBLICATION FIGURES (300 DPI)
# ==========================================

def plot_figure_1_progression():
    """Figure 1: Pipeline progression across MetaCardis iterations (Mantel r & runtime)."""
    df = HISTORICAL_MAIN_SPLIT.copy()
    main_csv = RES_DIR / "main_split.csv"
    if main_csv.exists():
        # Columns written by snf_neighbours.py: Pipeline, r_pearson, r_spearman, Run_s, ...
        current = pd.read_csv(main_csv).rename(columns={"r_pearson": "Pearson_r", "r_spearman": "Spearman_r"})
        current["Pipeline"] = current["Pipeline"].replace({"SNF neighbours": FINAL})
        current = current[["Pipeline", "Pearson_r", "Spearman_r", "Run_s"]]
        df = pd.concat([df[~df["Pipeline"].isin(current["Pipeline"])], current], ignore_index=True)
    else:
        print(f"Note: {main_csv} not found (run snf_neighbours.py experiments); Figure 1 shows the recorded results only.")

    df["Short_Name"] = df["Pipeline"].str.replace(" (score space)", "", regex=False).str.replace(" pipeline", "", regex=False)
    df = df.sort_values("Pearson_r", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.5, 1]})

    y = np.arange(len(df))
    height = 0.36
    ax1.barh(y + height / 2, df["Pearson_r"], height, label="Pearson Mantel r", color=PALETTE[0])
    if "Spearman_r" in df.columns:
        ax1.barh(y - height / 2, df["Spearman_r"], height, label="Spearman Mantel r", color=PALETTE[1])

    # Highlight final chosen model
    final_idx = [i for i, name in enumerate(df["Short_Name"]) if "snf" in name.lower() and "miss" not in name.lower() and "mkc" not in name.lower()]
    for idx in final_idx:
        ax1.get_yticklabels()[idx].set_fontweight("bold")
        ax1.get_yticklabels()[idx].set_color("#B22222")

    ax1.set_yticks(y)
    ax1.set_yticklabels(df["Short_Name"])
    ax1.set_xlabel("Mantel Correlation (r)")
    ax1.set_title("A. Multi-Omic Distance Accuracy (MetaCardis)")
    ax1.set_xlim(0, 0.72)
    ax1.legend(loc="lower right", frameon=True)

    # Panel B: Inference Runtime (log scale)
    colors_rt = ["#B22222" if i in final_idx else PALETTE[7] for i in range(len(df))]
    ax2.barh(y, np.maximum(df["Run_s"], 0.05), height=0.55, color=colors_rt)
    ax2.set_xscale("log")
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.set_xlabel("Execution Runtime (seconds, log scale)")
    ax2.set_title("B. Computational Cost")
    for idx, val in enumerate(df["Run_s"]):
        ax2.text(max(val, 0.05) * 1.2, idx, f"{val:.2f}s", va="center", fontsize=9.5)
    ax2.set_xlim(0.03, 600)

    fig.tight_layout()
    out_path = FIG_DIR / "fig1_pipeline_progression.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_figure_2_missingness_sweep(cohort_bench_df: pd.DataFrame):
    """
    Figure 2: Robustness across missingness fractions (from the MetaCardis sweep results)
    + Cross-cohort Mantel comparison on MetaCardis, Franzosa, and Yachida.
    """
    sweep_csv = RES_DIR / "missing_sweep.csv"  # written by snf_neighbours.py

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A: Missingness Sweep (10% - 80%)
    if sweep_csv.exists():
        sweep = pd.read_csv(sweep_csv)
        metric_col = "r_masked_pearson"
        keep_keywords = ["snf neighbours", "knn50", "mean fill", "ridge"]
        sweep_sub = sweep[sweep["Pipeline"].str.lower().apply(lambda s: any(k in s for k in keep_keywords))].copy()
        sweep_sub["Method"] = sweep_sub["Pipeline"].str.split(" \\(").str[0]

        sns.lineplot(
            data=sweep_sub,
            x="Missing_fraction",
            y=metric_col,
            hue="Method",
            marker="o",
            linewidth=2.2,
            palette=PALETTE[: sweep_sub["Method"].nunique()],
            ax=axes[0],
        )
        axes[0].set_xticks([0.1, 0.2, 0.4, 0.6, 0.8])
        axes[0].set_xticklabels(["10%", "20%", "40%", "60%", "80%"])
        axes[0].set_xlabel("Cohort Missingness Fraction")
        axes[0].set_ylabel("Pearson Mantel r (Masked Subjects)")
        axes[0].set_title("A. Robustness Across Missingness Levels (MetaCardis)")
        axes[0].legend(title="", frameon=True)
    else:
        axes[0].text(0.5, 0.5, f"{sweep_csv} not found", ha="center", va="center")
        axes[0].set_title("A. Missingness Sweep")

    # Panel B: External Validation Benchmark (MetaCardis vs Franzosa vs Yachida)
    sns.barplot(
        data=cohort_bench_df,
        x="Cohort",
        y="Pearson_r",
        hue="Method",
        palette=[PALETTE[7], PALETTE[0], "#B22222"][: cohort_bench_df["Method"].nunique()],
        ax=axes[1],
    )
    axes[1].set_ylabel("Pearson Mantel r (40% Missingness)")
    axes[1].set_xlabel("")
    axes[1].set_title("B. Cross-Cohort Generalization (40% Masked)")
    axes[1].set_ylim(0, max(0.85, cohort_bench_df["Pearson_r"].max() + 0.1))
    axes[1].legend(title="", frameon=True)

    fig.tight_layout()
    out_path = FIG_DIR / "fig2_missingness_sweep.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


def plot_figure_3_pcoa(cohorts_info: list):
    """Figure 3: 3-panel PCoA / MDS embedding colored by clinical disease group across cohorts."""
    fig, axes = plt.subplots(1, len(cohorts_info), figsize=(5.4 * len(cohorts_info), 5.2), squeeze=False)
    axes = axes[0]

    for ax, (title, dist_df, meta_df, group_col) in zip(axes, cohorts_info):
        common = dist_df.index.intersection(meta_df.index)
        D = dist_df.loc[common, common].to_numpy()
        groups = meta_df.loc[common, group_col].fillna("Unknown").astype(str).str.strip()

        # Classical MDS (PCoA) on precomputed Euclidean distance matrix
        mds = MDS(
            n_components=2,
            metric="precomputed",
            init="classical_mds",
            random_state=RANDOM_STATE,
            normalized_stress="auto",
        )
        coords = mds.fit_transform(D)

        plot_df = pd.DataFrame({"PCoA 1": coords[:, 0], "PCoA 2": coords[:, 1], "Group": groups.values})
        unique_groups = sorted(plot_df["Group"].unique())
        palette_use = sns.color_palette("tab10", n_colors=len(unique_groups))

        sns.scatterplot(
            data=plot_df,
            x="PCoA 1",
            y="PCoA 2",
            hue="Group",
            hue_order=unique_groups,
            palette=palette_use,
            s=36,
            alpha=0.8,
            edgecolor="none",
            ax=ax,
        )
        ax.set_title(f"{title} (n={len(common)})")
        ax.legend(title="Clinical Group", fontsize=8.5, title_fontsize=9, loc="best", frameon=True)

    fig.suptitle("Predicted Multi-Omic Score-Space Geometry Across Cohorts (`snf_neighbours`)", y=1.02, fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    out_path = FIG_DIR / "fig3_external_validation_pcoa.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path}")


# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Cohorts with both input files and a submission matrix (run snf_neighbours.py first).
    available = {}
    for name, (folder, matrix) in COHORTS.items():
        if all((folder / file).exists() for file in FILES) and matrix.exists():
            meta = pd.read_csv(folder / "metadata.csv", index_col=0)
            meta.index = meta.index.astype(str).str.strip()
            available[name] = (folder, load_distance_csv(matrix), meta)
        else:
            print(f"Skipping {name}: input files in {folder} or {matrix} not found")
    if not available:
        raise SystemExit("No cohort to evaluate.")

    # 1. 40% missingness Mantel benchmark in every cohort
    print("=" * 75)
    print("RUNNING 40% MISSINGNESS MANTEL BENCHMARK ACROSS COHORTS")
    print("=" * 75)
    bench_rows = []
    for name, (folder, _, _) in available.items():
        bench_rows.extend(evaluate_cohort_40pct_split(folder, name))
    bench_df = pd.DataFrame(bench_rows)
    print("\n--- Mantel Benchmark Results (40% Masked) ---")
    print(bench_df.round(4).to_string(index=False))

    # 2. Biological group separation on the full submission matrices
    print("\n" + "=" * 75)
    print("BIOLOGICAL GROUP SEPARATION ON FULL SUBMISSION MATRICES")
    print("=" * 75)
    group_col = "PATGROUPFINAL_C"
    sep_df = pd.DataFrame([
        compute_biological_separation(distances, meta, group_col, name)
        for name, (_, distances, meta) in available.items()
    ])
    print(sep_df.round(4).to_string(index=False))

    summary_out = pd.merge(bench_df[bench_df["Method"] == FINAL], sep_df, on="Cohort", how="outer")
    summary_path = RES_DIR / "validation_summary.csv"
    summary_out.to_csv(summary_path, index=False)
    print(f"\n[OK] Saved summary table to {summary_path}")

    # 3. Figures
    print("\n" + "=" * 75)
    print("GENERATING 300 DPI FIGURES")
    print("=" * 75)
    plot_figure_1_progression()
    plot_figure_2_missingness_sweep(bench_df)
    plot_figure_3_pcoa([(name, distances, meta, group_col) for name, (_, distances, meta) in available.items()])
