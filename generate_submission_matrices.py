"""
Generate final n x n submission distance matrices using the selected
`snf_neighbours` score-space pipeline across MetaCardis, Franzosa, and Yachida.

Outputs:
  - student_res_metacardis.csv
  - student_res_franzosa.csv
  - student_res_yachida.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import core utilities from existing project modules
from pipeline import RANDOM_STATE, load_data, snf
import snf_neighbours as snf_mod

OMICS = ("mb", "mt")
OTHER = {"mb": "mt", "mt": "mb"}
PSEUDOCOUNT = 1e-6


def detect_csv_delimiter(sample_path: Path) -> str:
    """
    Inspect the first line of an existing project CSV to check whether
    standard ',' or comma-space ', ' is used.
    """
    if sample_path.exists():
        with open(sample_path, "r", encoding="utf-8") as f:
            header = f.readline()
        if ", " in header:
            return ", "
    return ","


def omic_features(block: pd.DataFrame, omic: str) -> pd.DataFrame:
    """Log-ratio transformation (CLR for microbiome, log for metabolome)."""
    logged = np.log(block + PSEUDOCOUNT)
    if omic == "mb":
        return logged.sub(logged.mean(axis=1), axis=0)
    return logged


def metadata_features(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Encode demographic and clinical metadata into numeric features,
    gracefully handling missing columns in external validation cohorts.
    """
    parts = []
    cat_cols = [c for c in ("PATGROUPFINAL_C", "CENTER_C") if c in metadata.columns]
    if cat_cols:
        cats = pd.get_dummies(metadata[cat_cols].astype(str)).astype(float)
        parts.append(cats)

    num_cols = [c for c in ("GENDER", "BMI_C", "AGE") if c in metadata.columns]
    if num_cols:
        num = metadata[num_cols].apply(pd.to_numeric, errors="coerce")
        filled = num.fillna(num.median()).fillna(0.0)
        miss_flags = num.isna().astype(float).add_suffix("_missing")
        parts.extend([filled, miss_flags])

    if not parts:
        # Fallback constant feature if no metadata columns exist
        return pd.DataFrame({"const": np.ones(len(metadata))}, index=metadata.index)

    features = pd.concat(parts, axis=1)
    return features.loc[:, ~features.T.duplicated()]


def local_affinity(X: np.ndarray, K: int = 20, mu: float = 0.5) -> np.ndarray:
    """Scaled exponential similarity kernel used by SNF."""
    n = X.shape[0]
    K_eff = min(K, max(1, n - 1))
    D = squareform(pdist(X, metric="euclidean"))
    knn_mean = np.sort(D, axis=1)[:, 1 : K_eff + 1].mean(axis=1)
    width = np.maximum(
        mu * (knn_mean[:, None] + knn_mean[None, :] + D) / 3.0,
        np.finfo(float).eps,
    )
    return np.exp(-(D**2) / (2.0 * width**2))


class ScoreSpace:
    """
    2-PC PCA fitted on complete reference samples, decomposed into
    additive per-omic score contributions: score = z_mb @ L_mb.T + z_mt @ L_mt.T.
    """

    def __init__(self, train: pd.DataFrame, mb_cols: list, mt_cols: list):
        self.columns = {"mb": mb_cols, "mt": mt_cols}
        self.features = list(train.columns)
        self.scaler = StandardScaler().fit(train)
        self.pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(
            self.scaler.transform(train)
        )
        self.loadings = pd.DataFrame(self.pca.components_, columns=train.columns)

    def contributions(self, data: pd.DataFrame) -> dict:
        z = (data - self.scaler.mean_) / self.scaler.scale_ - self.pca.mean_
        return {
            omic: pd.DataFrame(
                z[cols].to_numpy() @ self.loadings[cols].to_numpy().T,
                index=data.index,
            )
            for omic, cols in self.columns.items()
        }

    def combine(self, contributions: dict) -> pd.DataFrame:
        return contributions["mb"] + contributions["mt"]


def run_snf_neighbours_full_cohort(
    mb_path: Path,
    mt_path: Path,
    meta_path: Path,
    K: int = 20,
    k_pred: int = 60,
    T: int = 20,
    mu: float = 0.5,
) -> pd.DataFrame:
    """
    Fits ScoreSpace on complete samples in the cohort, predicts missing
    2-D score contributions via `snf_neighbours` for incomplete samples,
    and returns the full n x n pairwise Euclidean distance matrix.
    """
    # Use snf_neighbours.py's native runner if exposed, otherwise execute directly
    if hasattr(snf_mod, "predict_full_cohort"):
        return snf_mod.predict_full_cohort(mb_path, mt_path, meta_path)

    data, complete, mb_cols, mt_cols, metadata = load_data(
        str(mb_path), str(mt_path), str(meta_path)
    )
    all_ids = data.index
    train_ids = complete.index
    n_train = len(train_ids)

    if n_train < 3:
        raise ValueError(f"Insufficient complete samples ({n_train}) in {mb_path.parent}")

    # Safely cap neighbor hyperparameters for smaller external cohorts
    K_eff = min(K, max(2, n_train - 1))
    k_pred_eff = min(k_pred, n_train)

    # 1. Fit ScoreSpace on complete reference samples
    target = ScoreSpace(complete[mb_cols + mt_cols], mb_cols, mt_cols)
    contributions = target.contributions(data[mb_cols + mt_cols])

    # 2. Build predictor views (CLR microbiome, log metabolome, encoded metadata)
    views = {
        "mb": omic_features(data[mb_cols], "mb"),
        "mt": omic_features(data[mt_cols], "mt"),
        "meta": metadata_features(metadata.loc[all_ids]),
    }

    # 3. Predict missing 2-D score contributions for each modality
    for omic in OMICS:
        miss_ids = contributions[omic].index[contributions[omic].isna().any(axis=1)]
        if len(miss_ids) == 0:
            continue

        obs_omic = OTHER[omic]
        # Verify the observed omic actually exists for these missing samples
        valid_miss = miss_ids[views[obs_omic].loc[miss_ids].notna().all(axis=1)]
        nodes = train_ids.append(valid_miss)

        affinities = []
        for view in (obs_omic, "meta"):
            scaler_v = StandardScaler().fit(views[view].loc[train_ids])
            X_nodes = np.nan_to_num(
                scaler_v.transform(views[view].loc[nodes]), nan=0.0
            )
            affinities.append(local_affinity(X_nodes, K=K_eff, mu=mu))

        fused = snf(affinities, K=K_eff, T=T, minmax=False)
        links = fused[n_train:, :n_train]

        kth_strongest = -np.partition(-links, k_pred_eff - 1, axis=1)[
            :, [k_pred_eff - 1]
        ]
        links = np.where(links >= kth_strongest, links, 0.0)
        row_sums = np.maximum(links.sum(axis=1, keepdims=True), np.finfo(float).eps)
        weights = links / row_sums

        train_scores = contributions[omic].loc[train_ids].to_numpy()
        contributions[omic].loc[valid_miss] = weights @ train_scores

        # Fallback to training mean if any sample lacked BOTH omics
        still_missing = contributions[omic].index[
            contributions[omic].isna().any(axis=1)
        ]
        if len(still_missing) > 0:
            contributions[omic].loc[still_missing] = train_scores.mean(axis=0)

    # 4. Combine contributions into 2D score coordinates & compute Euclidean distances
    coords = target.combine(contributions).loc[all_ids]
    dist_arr = cdist(coords.to_numpy(), coords.to_numpy(), metric="euclidean")
    dist_arr = (dist_arr + dist_arr.T) / 2.0
    np.fill_diagonal(dist_arr, 0.0)

    return pd.DataFrame(dist_arr, index=all_ids, columns=all_ids)


def validate_and_export(
    dist_df: pd.DataFrame, out_path: Path, sep: str = ","
) -> None:
    """Enforce symmetry, zero diagonal, no NaNs/Infs, and write formatted CSV."""
    assert list(dist_df.index) == list(dist_df.columns), "Row/col SampleIDs mismatch!"
    arr = dist_df.to_numpy()
    assert not np.isnan(arr).any(), f"NaN values detected in {out_path.name}!"
    assert not np.isinf(arr).any(), f"Inf values detected in {out_path.name}!"
    assert np.allclose(arr, arr.T, atol=1e-10), f"Matrix not symmetric in {out_path.name}!"
    assert np.allclose(np.diag(arr), 0.0, atol=1e-12), f"Non-zero diagonal in {out_path.name}!"

    if sep == ",":
        dist_df.to_csv(out_path, index=True)
    else:
        # Support explicit comma-space delimiter if matched in root CSVs
        lines = [sep.join([""] + [str(c) for c in dist_df.columns])]
        for idx, row in dist_df.iterrows():
            lines.append(sep.join([str(idx)] + [f"{v:.8f}" for v in row.values]))
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n[OK] Saved {out_path.name} | Shape: {dist_df.shape} | Delimiter: {repr(sep)}")
    print("Top-left 5x5 slice:")
    print(dist_df.iloc[:5, :5].round(4).to_string())


if __name__ == "__main__":
    root = Path(".")
    delimiter = detect_csv_delimiter(root / "metadata.csv")

    datasets = {
        "metacardis": (
            root / "microbiome.csv",
            root / "metabolome.csv",
            root / "metadata.csv",
        ),
        "franzosa": (
            root / "aligned_datasets" / "franzosa" / "microbiome.csv",
            root / "aligned_datasets" / "franzosa" / "metabolome.csv",
            root / "aligned_datasets" / "franzosa" / "metadata.csv",
        ),
        "yachida": (
            root / "aligned_datasets" / "yachida" / "microbiome.csv",
            root / "aligned_datasets" / "yachida" / "metabolome.csv",
            root / "aligned_datasets" / "yachida" / "metadata.csv",
        ),
    }

    for name, (mb_p, mt_p, meta_p) in datasets.items():
        print("=" * 70)
        print(f"Processing cohort: {name.upper()}")
        dist_matrix = run_snf_neighbours_full_cohort(mb_p, mt_p, meta_p)
        validate_and_export(
            dist_matrix, root / f"student_res_{name}.csv", sep=delimiter
        )