"""
SNF Neighbours: the final pipeline (Milestone 3)

Estimates an n x n distance matrix between samples from microbiome + metabolome data when one
omic is missing for some samples, graded with the same Mantel test as pipeline.py.

For each omic, the samples missing it are filled in as follows:
1. Inputs: the observed omic (log-transformed; CLR for the compositional microbiome) and the
   metadata, each standard-scaled on the complete training samples.
2. One scaled exponential affinity network per input (Wang et al., 2014), built over the complete
   training samples and the samples missing this omic.
3. SNF fuses the two networks into one similarity network.
4. A sample's missing omic is the weighted mean of the standard-scaled features of its k_pred most
   strongly linked training samples, weighted by the fused similarity.

The observed omic keeps its own standard-scaled features (scaler fitted on the complete training
samples), and the distance is Euclidean over all features of both omics. The pipeline itself uses
no PCA: PCA appears only in the grader's ground truth (first two PCs of the standard-scaled data),
which the Mantel test and the parameter tuning compare against.

Reference point: Global KNN50 (the Milestone 1 benchmark) fills in the missing omic with a
50-neighbour average and is measured in the same feature space.

Every result is reported with both a Pearson and a Spearman Mantel r, so it can be compared with
earlier (Pearson) results. Parameters are tuned on one of them (--tune-metric, default Spearman).

Experiments (all results are also written to --out as CSV, plus one figure):
- Main split: same split and masks as pipeline.py, with Mantel r, p-value, tuning time and
  prediction time.
- Paired comparison over random splits: Mantel r, gain over KNN50 and mean run time.
- Missing-percentage sweep: one omic is hidden in 10% to 80% of ALL samples, so the pool of
  complete reference samples shrinks as the percentage grows. Scored against the grader's
  geometry (PCA fitted on every sample's true values), over all samples and over the samples
  missing an omic only.
"""

import argparse
import itertools
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import ConstantInputWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline import (
    RANDOM_STATE,
    TEST_SIZE,
    apply_mask,
    load_data,
    make_mask,
    mantel_test,
    pca_distance,
    reconstruct_knn,
    snf,
)

OMICS = ("mb", "mt")
OTHER = {"mb": "mt", "mt": "mb"}
PSEUDOCOUNT = 1e-6
PIPELINE = "SNF neighbours"
BENCHMARK = "Global KNN50 (Benchmark)"
METRICS = ("pearson", "spearman")
PARAM_GRID = [{"K": K, "k_pred": k_pred} for K, k_pred in itertools.product([10, 20, 40], [10, 30, 60])]

# ==========================================
# 1. SPLITS & INPUT FEATURES
# ==========================================

@dataclass
class Split:
    train: pd.DataFrame
    combined: pd.DataFrame
    masked_ids: np.ndarray
    ground_truth: pd.DataFrame


def make_split(complete, mb_cols, mt_cols, test_size, split_seed, mask_seed):
    """Hold out complete samples and hide one omic in each: half lose the metabolome, half the microbiome."""
    train_ids, test_ids = train_test_split(complete.index, test_size=test_size, random_state=split_seed)
    train, test = complete.loc[train_ids], complete.loc[test_ids]
    miss_mt, miss_mb = make_mask(test.index, np.random.default_rng(mask_seed))
    combined = pd.concat([train, apply_mask(test, miss_mt, miss_mb, mb_cols, mt_cols)])
    masked_ids = np.concatenate([miss_mt, miss_mb])
    return Split(train, combined, masked_ids, pca_distance(train, test.loc[masked_ids]))


def make_missing_split(complete, mb_cols, mt_cols, missing_fraction, seed):
    """
    Hide one omic in a fraction of ALL samples (half lose the metabolome, half the microbiome).

    The untouched samples are the only reference the pipeline may learn from, so a higher
    fraction means both more samples to fill in and fewer samples to learn from. The ground truth
    is the grader's: PCA fitted on the true values of every sample, hidden ones included, with
    distances over all samples. For one seed the hidden sets are nested across fractions.
    """
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.asarray(complete.index))
    n_masked = int(round(missing_fraction * len(ids)))
    masked, kept = ids[:n_masked], ids[n_masked:]
    miss_mt, miss_mb = make_mask(masked, rng)
    train = complete.loc[kept]
    combined = pd.concat([train, apply_mask(complete.loc[masked], miss_mt, miss_mb, mb_cols, mt_cols)])
    masked_ids = np.concatenate([miss_mt, miss_mb])
    return Split(train, combined, masked_ids, pca_distance(complete, complete.loc[combined.index]))


def omic_features(block, omic):
    """Log-ratio inputs for the affinity networks (CLR for the compositional microbiome)."""
    logged = np.log(block + PSEUDOCOUNT)
    if omic == "mb":
        return logged.sub(logged.mean(axis=1), axis=0)
    return logged


def metadata_features(metadata):
    categorical = pd.get_dummies(metadata[["PATGROUPFINAL_C", "CENTER_C"]].astype(str)).astype(float)
    numeric = metadata[["GENDER", "BMI_C", "AGE"]]
    features = pd.concat(
        [categorical, numeric.fillna(numeric.median()), numeric.isna().astype(float).add_suffix("_missing")],
        axis=1,
    )
    return features.loc[:, ~features.T.duplicated()]

# ==========================================
# 2. FEATURE SPACE (WHERE DISTANCES ARE MEASURED)
# ==========================================

class FeatureSpace:
    """Each omic is its own block of standard-scaled features (scaler fitted on the complete training samples)."""

    def __init__(self, train, mb_cols, mt_cols):
        self.columns = {"mb": mb_cols, "mt": mt_cols}
        self.scaler = StandardScaler().fit(train)

    def blocks(self, data):
        """Each omic's scaled features; rows where that omic is missing come out as NaN."""
        z = (data - self.scaler.mean_) / self.scaler.scale_
        return {omic: z[cols].copy() for omic, cols in self.columns.items()}

    @staticmethod
    def combine(blocks):
        """Full feature vector: the two omics' blocks side by side."""
        return pd.concat([blocks["mb"], blocks["mt"]], axis=1)


def distance_matrix(coords):
    return pd.DataFrame(cdist(coords, coords), index=coords.index, columns=coords.index)


@dataclass
class Context:
    """
    What the predictor may use: omic and metadata inputs plus the known scaled features.

    targets[omic] holds that omic's scaled features for every sample, NaN where the omic is
    missing; the predictor fills those rows.
    """
    views: dict
    targets: dict
    train_ids: pd.Index

    def missing(self, omic):
        target = self.targets[omic]
        return target.index[target.isna().any(axis=1)]

    def standardized(self, view, rows):
        X = self.views[view]
        return StandardScaler().fit(X.loc[self.train_ids]).transform(X.loc[rows])

# ==========================================
# 3. SNF NEIGHBOURS
# ==========================================

def local_affinity(X, K=20, mu=0.5):
    """Scaled exponential affinity as in SNF: the kernel width adapts to each pair's neighbourhood density."""
    D = squareform(pdist(X))
    knn_mean = np.sort(D, axis=1)[:, 1:K + 1].mean(axis=1)
    width = np.maximum(mu * (knn_mean[:, None] + knn_mean[None, :] + D) / 3.0, np.finfo(float).eps)
    return np.exp(-(D ** 2) / (2.0 * width ** 2))


def snf_predictor(ctx, K=20, k_pred=20, T=20, mu=0.5):
    """Missing omic's features = weighted mean over the sample's strongest fused links to training samples."""
    n_train = len(ctx.train_ids)
    k_pred = min(k_pred, n_train)
    predictions = {}
    for omic in OMICS:
        nodes = ctx.train_ids.append(ctx.missing(omic))
        affinities = [local_affinity(ctx.standardized(view, nodes), K, mu) for view in (OTHER[omic], "meta")]
        links = snf(affinities, K=K, T=T, minmax=False)[n_train:, :n_train]
        kth_strongest = -np.partition(-links, k_pred - 1, axis=1)[:, [k_pred - 1]]
        links = np.where(links >= kth_strongest, links, 0.0)
        train_features = ctx.targets[omic].loc[ctx.train_ids].to_numpy()
        predictions[omic] = (links / links.sum(axis=1, keepdims=True)) @ train_features
    return predictions


def pipeline_distance(split, meta, mb_cols, mt_cols, **params):
    """Distance matrix over all samples of the split: missing omics filled in by SNF neighbours."""
    space = FeatureSpace(split.train, mb_cols, mt_cols)
    combined = split.combined
    views = {
        "mb": omic_features(combined[mb_cols], "mb"),
        "mt": omic_features(combined[mt_cols], "mt"),
        "meta": meta.loc[combined.index],
    }
    ctx = Context(views, space.blocks(combined), split.train.index)
    for omic, predicted in snf_predictor(ctx, **params).items():
        ctx.targets[omic].loc[ctx.missing(omic)] = predicted
    return distance_matrix(space.combine(ctx.targets))


def knn50_distance(split, mb_cols, mt_cols):
    """Milestone 1 benchmark: KNN50 imputation, distances in the same feature space."""
    reconstructed = reconstruct_knn(split.combined, split.train.index, n_neighbors=50)
    space = FeatureSpace(split.train, mb_cols, mt_cols)
    return distance_matrix(space.combine(space.blocks(reconstructed)))

# ==========================================
# 4. TUNING & EVALUATION
# ==========================================

def short(name):
    return name.split(" (")[0]


def evaluate(split, distances, n_permutations, ids=None, method="spearman"):
    """Mantel r and p-value (method: "pearson" or "spearman") over `ids` (default: the samples missing an omic)."""
    ids = split.masked_ids if ids is None else ids
    return mantel_test(split.ground_truth.loc[ids, ids], distances.loc[ids, ids],
                       n_permutations=n_permutations, method=method)


def tune(train, meta, mb_cols, mt_cols, metric="spearman"):
    """Pick K and k_pred on an inner masked split of the training samples only (same protocol as pipeline.py)."""
    inner = make_split(train, mb_cols, mt_cols, 0.25, RANDOM_STATE, RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        scores = [
            evaluate(inner, pipeline_distance(inner, meta, mb_cols, mt_cols, **params), 0, method=metric)[0]
            for params in PARAM_GRID
        ]
    # A degenerate setting (all distances equal) has no Mantel r; it must never be selected.
    return PARAM_GRID[int(np.argmax(np.nan_to_num(scores, nan=-np.inf)))]


def run_all(split, meta, mb_cols, mt_cols, params=None, tune_metric="spearman"):
    """
    Distance matrices over all samples of the split, from the KNN50 benchmark and from SNF neighbours.

    Tune_s is the time spent choosing parameters and Run_s the time from the masked data to the
    distance matrix. Neither includes the Mantel test. With params given, tuning is skipped;
    otherwise the parameters maximise the `tune_metric` Mantel r.
    """
    start = time.perf_counter()
    knn = knn50_distance(split, mb_cols, mt_cols)
    runs = [{"Pipeline": BENCHMARK, "distances": knn, "params": {},
             "Tune_s": 0.0, "Run_s": time.perf_counter() - start}]

    tune_s = 0.0
    if params is None:
        start = time.perf_counter()
        params = tune(split.train, meta, mb_cols, mt_cols, tune_metric)
        tune_s = time.perf_counter() - start
    start = time.perf_counter()
    distances = pipeline_distance(split, meta, mb_cols, mt_cols, **params)
    runs.append({"Pipeline": PIPELINE, "distances": distances, "params": params,
                 "Tune_s": tune_s, "Run_s": time.perf_counter() - start})
    return runs


def compare_on_split(split, meta, mb_cols, mt_cols, n_permutations, tune_metric="spearman"):
    """
    Score the pipeline and the benchmark on the samples missing an omic, with both Mantel correlations.

    Returns the table (r_pearson, p_pearson, r_spearman, p_spearman per pipeline) and the chosen parameters.
    """
    rows, chosen = [], {}
    for run in run_all(split, meta, mb_cols, mt_cols, tune_metric=tune_metric):
        row = {"Pipeline": run["Pipeline"]}
        for metric in METRICS:
            row[f"r_{metric}"], row[f"p_{metric}"] = evaluate(split, run["distances"], n_permutations, method=metric)
        row.update({"Params": run["params"] or "-", "Tune_s": run["Tune_s"], "Run_s": run["Run_s"]})
        rows.append(row)
        if run["Pipeline"] == PIPELINE:
            chosen = run["params"]
    return pd.DataFrame(rows), chosen

# ==========================================
# 5. MISSING-PERCENTAGE SWEEP
# ==========================================

def missing_sweep(complete, meta, mb_cols, mt_cols, fractions, n_seeds, params, tune_metric="spearman"):
    """
    Mantel r (Pearson and Spearman) as the share of samples missing an omic grows.

    r_all_* is scored over all samples, complete ones included, which is how a submitted n x n
    matrix is graded. r_masked_* is scored over the samples missing an omic only, which isolates
    how well the missing omic is predicted. Run_s includes tuning when params is None.
    """
    rows = []
    for fraction in fractions:
        for seed in range(n_seeds):
            split = make_missing_split(complete, mb_cols, mt_cols, fraction, seed)
            everyone = split.combined.index
            runs = run_all(split, meta, mb_cols, mt_cols, params, tune_metric)
            for run in runs:
                row = {
                    "Missing_fraction": fraction,
                    "Seed": seed,
                    "Pipeline": run["Pipeline"],
                    "Complete_samples": len(split.train),
                }
                for metric in METRICS:
                    row[f"r_all_{metric}"] = evaluate(split, run["distances"], 0, everyone, metric)[0]
                    row[f"r_masked_{metric}"] = evaluate(split, run["distances"], 0, method=metric)[0]
                row["Run_s"] = run["Tune_s"] + run["Run_s"]
                rows.append(row)
            latest = rows[-len(runs):]
            print(f"missing {fraction:.0%} seed {seed} (r_all pearson / spearman): " + "  ".join(
                f"{short(row['Pipeline'])}={row['r_all_pearson']:.3f}/{row['r_all_spearman']:.3f}" for row in latest
            ))
    return pd.DataFrame(rows)


def plot_sweep(sweep, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed: figure skipped (the numbers are in the CSV files)")
        return
    panels = (("r_all", "All samples (as graded)"), ("r_masked", "Samples missing an omic only"))
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(12, 4.5 * len(METRICS)), sharex=True)
    for metric, row_axes in zip(METRICS, axes):
        for ax, (column, title) in zip(row_axes, panels):
            stats = sweep.groupby(["Pipeline", "Missing_fraction"], sort=False)[f"{column}_{metric}"].agg(["mean", "std"])
            for name in sweep["Pipeline"].unique():
                line = stats.loc[name]
                ax.errorbar(line.index * 100, line["mean"], yerr=line["std"].fillna(0.0),
                            marker="o", capsize=3, linewidth=1.5, label=short(name))
            ax.set_title(f"{title}, {metric.capitalize()}")
            ax.set_xlabel("Samples missing one omic (%)")
            ax.set_ylabel(f"Mantel r ({metric.capitalize()}) vs ground truth")
            ax.grid(alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("SNF neighbours vs KNN50: distances over the standard-scaled features")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"figure saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the SNF neighbours pipeline against the KNN50 benchmark.")
    parser.add_argument("--splits", type=int, default=10, help="extra random splits for the paired comparison (0 skips)")
    parser.add_argument("--permutations", type=int, default=999, help="Mantel permutations on the main split")
    parser.add_argument("--sweep-seeds", type=int, default=3, help="repeats per missing fraction in the sweep (0 skips)")
    parser.add_argument("--fractions", default="0.1,0.2,0.4,0.6,0.8",
                        help="comma-separated shares of all samples that lose one omic in the sweep")
    parser.add_argument("--sweep-retune", action="store_true",
                        help="tune inside every sweep split instead of reusing the main split's parameters (slower)")
    parser.add_argument("--tune-metric", choices=METRICS, default="spearman",
                        help="Mantel correlation that the parameter tuning maximises (both are always reported)")
    parser.add_argument("--out", default="results", help="folder for the CSV tables and the figure")
    return parser.parse_args()

# ==========================================
# 6. EXECUTION
# ==========================================

if __name__ == "__main__":
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    order = [BENCHMARK, PIPELINE]
    data, complete, mb_cols, mt_cols, metadata = load_data("microbiome.csv", "metabolome.csv", "metadata.csv")
    meta = metadata_features(metadata)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    print("\n" + "=" * 60)
    print(f"MAIN SPLIT (same train/test split and masks as pipeline.py; tuned on {args.tune_metric})")
    print("=" * 60)
    main_split = make_split(complete, mb_cols, mt_cols, TEST_SIZE, RANDOM_STATE, RANDOM_STATE + 10_000)
    results, main_params = compare_on_split(main_split, meta, mb_cols, mt_cols, args.permutations, args.tune_metric)
    print(results.round(4).to_string(index=False))
    print("(Tune_s: choosing parameters. Run_s: masked data to distance matrix. Seconds, Mantel test excluded.)")
    print(f"Chosen parameters: {main_params}")
    results.to_csv(out / "main_split.csv", index=False)

    if args.splits:
        print("\n" + "=" * 60)
        print(f"PAIRED COMPARISON OVER {args.splits} RANDOM SPLITS (tuned inside each split on {args.tune_metric})")
        print("=" * 60)
        per_split = []
        for seed in range(args.splits):
            split = make_split(complete, mb_cols, mt_cols, TEST_SIZE, seed, seed + 10_000)
            split_results, _ = compare_on_split(split, meta, mb_cols, mt_cols, 0, args.tune_metric)
            per_split.append(split_results.drop(columns=[f"p_{metric}" for metric in METRICS]).assign(Split=seed))
            print(f"split {seed} (pearson / spearman): " + "  ".join(
                f"{short(row.Pipeline)}={row.r_pearson:.3f}/{row.r_spearman:.3f}" for row in split_results.itertuples()
            ))

        paired = pd.concat(per_split, ignore_index=True)
        seconds = paired.groupby("Pipeline")[["Tune_s", "Run_s"]].mean().loc[order]
        summaries = {}
        for metric in METRICS:
            table = paired.pivot(index="Split", columns="Pipeline", values=f"r_{metric}")[order]
            gain = table.sub(table[BENCHMARK], axis=0)
            summaries[metric] = pd.DataFrame({
                "mean_r": table.mean(),
                "sd_r": table.std(),
                "mean_gain_vs_KNN50": gain.mean(),
                "sd_gain": gain.std(),
                "wins_vs_KNN50": (gain > 0).sum().astype(int),
                "mean_tune_s": seconds["Tune_s"],
                "mean_run_s": seconds["Run_s"],
            })
        summary = pd.concat(summaries, names=["Metric", "Pipeline"])
        print("\n" + summary.round(4).to_string())
        paired.to_csv(out / "paired_splits.csv", index=False)
        summary.to_csv(out / "paired_summary.csv")

    if args.sweep_seeds:
        fractions = [float(value) for value in args.fractions.split(",")]
        if not all(0.0 < fraction < 1.0 for fraction in fractions):
            raise SystemExit("--fractions must lie strictly between 0 and 1")
        print("\n" + "=" * 60)
        print(f"MISSING-PERCENTAGE SWEEP ({args.sweep_seeds} repeats per percentage, "
              + ("tuned inside each split" if args.sweep_retune else "main-split parameters") + ")")
        print("=" * 60)
        sweep = missing_sweep(complete, meta, mb_cols, mt_cols, fractions, args.sweep_seeds,
                              None if args.sweep_retune else main_params, args.tune_metric)
        grouped = sweep.groupby(["Missing_fraction", "Pipeline"])
        columns = [(f"r_{scope}_{metric}", f"Mean {metric.capitalize()} Mantel r over {title}")
                   for metric in METRICS
                   for scope, title in (("all", "ALL samples (as graded)"), ("masked", "samples missing an omic only"))]
        for column, title in columns + [("Run_s", "Mean run time in seconds")]:
            print(f"\n{title}; rows: share of samples missing one omic")
            print(grouped[column].mean().unstack()[order].rename(columns=short).round(4).to_string())
        sweep.to_csv(out / "missing_sweep.csv", index=False)
        grouped[[column for column, _ in columns] + ["Run_s"]].agg(["mean", "std"]).to_csv(
            out / "missing_sweep_summary.csv")
        plot_sweep(sweep, out / "missing_sweep.png")

    print(f"\nTables saved to {out}/")
