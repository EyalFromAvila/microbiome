"""
Score-Space Pipelines
Three predictors of the missing omic, graded with the same Mantel test as pipeline.py.

The ground truth is Euclidean distance on the first two principal components of the
standard-scaled microbiome + metabolome features. PCA is linear, so each sample's score
splits exactly into a microbiome part and a metabolome part:

    score = z_mb @ L_mb.T + z_mt @ L_mt.T

The observed omic's part is computed exactly; only the missing omic's part (two numbers
per sample) is predicted, and distances are Euclidean in the resulting 2-D score space.

Predictors of the missing part:
1. Ridge: linear regression from the observed omic and metadata
2. MKC_embd(ht) (Bhadra, Kaski & Rousu, 2017): kernel reconstruction weights learned in the
   views where a sample is observed and transferred to the view where it is missing
3. SNF neighbours (Wang et al., 2014): the observed omic and metadata fused into one
   similarity network whose strongest links average the training samples' scores
"""

import argparse
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
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
    rbf_kernel,
    reconstruct_knn,
    snf,
)

OMICS = ("mb", "mt")
OTHER = {"mb": "mt", "mt": "mb"}
PSEUDOCOUNT = 1e-6
BENCHMARK = "Global KNN50 (Benchmark)"

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


def omic_features(block, omic):
    """Log-ratio inputs for the predictors (CLR for the compositional microbiome); the target geometry stays raw."""
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
# 2. SCORE SPACE (SHARED BY ALL THREE PREDICTORS)
# ==========================================

class ScoreSpace:
    """Ground-truth PCA fitted on complete training samples, split into per-omic contributions."""

    def __init__(self, train, mb_cols, mt_cols):
        self.columns = {"mb": mb_cols, "mt": mt_cols}
        self.scaler = StandardScaler().fit(train)
        self.pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(self.scaler.transform(train))
        self.loadings = pd.DataFrame(self.pca.components_, columns=train.columns)

    def contributions(self, data):
        """Each omic's part of the PC scores; rows where that omic is missing come out as NaN."""
        z = (data - self.scaler.mean_) / self.scaler.scale_ - self.pca.mean_
        return {
            omic: pd.DataFrame(z[cols].to_numpy() @ self.loadings[cols].to_numpy().T, index=data.index)
            for omic, cols in self.columns.items()
        }


@dataclass
class Context:
    """What a predictor may use: omic and metadata inputs plus the known score contributions."""
    views: dict
    contributions: dict
    train_ids: pd.Index
    diagnostics: dict = field(default_factory=dict)

    def missing(self, omic):
        contribution = self.contributions[omic]
        return contribution.index[contribution.isna().any(axis=1)]

    def standardized(self, view, rows):
        X = self.views[view]
        return StandardScaler().fit(X.loc[self.train_ids]).transform(X.loc[rows])


def score_space_distance(split, meta, mb_cols, mt_cols, predictor, **params):
    space = ScoreSpace(split.train, mb_cols, mt_cols)
    combined = split.combined
    views = {
        "mb": omic_features(combined[mb_cols], "mb"),
        "mt": omic_features(combined[mt_cols], "mt"),
        "meta": meta.loc[combined.index],
    }
    ctx = Context(views, space.contributions(combined), split.train.index)
    for omic, predicted in predictor(ctx, **params).items():
        ctx.contributions[omic].loc[ctx.missing(omic)] = predicted
    coords = ctx.contributions["mb"] + ctx.contributions["mt"]
    distances = pd.DataFrame(cdist(coords, coords), index=coords.index, columns=coords.index)
    return distances, ctx.diagnostics

# ==========================================
# 3. PREDICTOR 1: RIDGE REGRESSION
# ==========================================

def ridge_predictor(ctx, alphas=np.logspace(-1, 5, 25)):
    """Regress the missing omic's PC contribution on the observed omic + metadata."""
    predictions = {}
    for omic in OMICS:
        inputs = pd.concat([ctx.views[OTHER[omic]], ctx.views["meta"]], axis=1)
        scaler = StandardScaler().fit(inputs.loc[ctx.train_ids])
        model = RidgeCV(alphas=alphas).fit(
            scaler.transform(inputs.loc[ctx.train_ids]), ctx.contributions[omic].loc[ctx.train_ids]
        )
        predictions[omic] = model.predict(scaler.transform(inputs.loc[ctx.missing(omic)]))
        ctx.diagnostics[f"ridge alpha ({omic} missing)"] = float(model.alpha_)
    return predictions

# ==========================================
# 4. PREDICTOR 2: MULTI-VIEW KERNEL COMPLETION, MKC_embd(ht)
# ==========================================

def project_to_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > css - 1)[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1), 0.0)


class MKC_embd_ht:
    """
    Multi-view kernel completion with heterogeneous embeddings (Bhadra, Kaski & Rousu, 2017).

    In view m each sample's feature map is rebuilt from basis samples, phi_t ≈ sum_i a_it phi_i,
    so the completed kernel is A^T K A. The objective has three parts:
      - within-view: A^T K A must reproduce the observed block of K
      - between-view (c1): A^(m) ≈ sum_l s_ml A^(l), a learned convex combination of the other views
      - l2,1 on the rows of A^(m) (c2): keeps a sparse basis set
    A sample missing in view m enters only the between-view term, so it inherits the
    reconstruction weights it has in the views where it is observed.

    The basis is restricted to samples observed in every view: a basis sample missing from
    another view would produce weights that cannot be transferred to that view.

    Optimisation: proximal gradient (FISTA with backtracking) on the observed columns, with S
    re-fitted after each outer round. Columns of samples missing from a view appear only in the
    weakly curved between-view term, so a shared gradient step would barely move them; they are
    instead set to their closed-form optimum after every step (each sample may be missing from
    at most one view). The l2,1 penalty is taken over the observed columns.
    """

    def __init__(self, c1=1.0, c2=0.1, n_outer=4, n_inner=50):
        self.c1 = c1
        self.c2 = c2
        self.n_outer = n_outer
        self.n_inner = n_inner

    def fit(self, kernels, observed):
        """kernels: N x N per view (only observed x observed entries are read); observed: boolean masks."""
        M, N = len(kernels), len(observed[0])
        observed = np.asarray(observed)
        if (~observed).sum(axis=0).max() > 1:
            raise ValueError("each sample may be missing from at most one view")
        self.basis_ = np.where(observed.all(axis=0))[0]
        obs = [np.where(o)[0] for o in observed]
        free = [np.where(~o)[0] for o in observed]
        G = [K[np.ix_(self.basis_, self.basis_)] for K in kernels]
        targets = [K[np.ix_(o, o)] for K, o in zip(kernels, obs)]

        def mix(W, X):
            return (W @ X.reshape(M, -1)).reshape(X.shape)

        def fill_free(A, S):
            """Closed-form minimiser of the between-view loss over each view's missing columns."""
            A = A.copy()
            for m in range(M):
                if len(free[m]) == 0:
                    continue
                cols = A[:, :, free[m]]
                others = [v for v in range(M) if v != m]
                total = sum(S[m, l] * cols[l] for l in others)
                for v in others:
                    rest = cols[v] - sum(S[v, l] * cols[l] for l in others if l != v)
                    total = total + S[v, m] * rest
                A[m][:, free[m]] = total / (1.0 + sum(S[v, m] ** 2 for v in others))
            return A

        def smooth(A, S, with_grad=True):
            value, grad = 0.0, np.zeros_like(A) if with_grad else None
            for m in range(M):
                C = A[m][:, obs[m]]
                GC = G[m] @ C
                R = C.T @ GC - targets[m]
                value += np.sum(R ** 2) / len(obs[m])
                if with_grad:
                    grad[m][:, obs[m]] = 4.0 * (GC @ R) / len(obs[m])
            E = A - mix(S, A)
            value += self.c1 * np.sum(E ** 2)
            if with_grad:
                grad += 2.0 * self.c1 * (E - mix(S.T, E))
                for m in range(M):
                    grad[m][:, free[m]] = 0.0
            return value, grad

        def l21(A):
            return sum(np.linalg.norm(A[m][:, obs[m]], axis=1).sum() for m in range(M))

        def prox(A, step):
            A = A.copy()
            for m in range(M):
                block = A[m][:, obs[m]]
                norms = np.linalg.norm(block, axis=1, keepdims=True)
                A[m][:, obs[m]] = block * np.maximum(0.0, 1.0 - step * self.c2 / np.maximum(norms, 1e-12))
            return A

        A = np.zeros((M, len(self.basis_), N))
        A[:, np.arange(len(self.basis_)), self.basis_] = 1.0
        S = (1.0 - np.eye(M)) / (M - 1)
        L, self.history_ = 1.0, []
        for _ in range(self.n_outer):
            A = fill_free(A, S)
            Y, A_prev, t = A, A, 1.0
            for _ in range(self.n_inner):
                fY, gY = smooth(Y, S)
                while True:
                    A_new = fill_free(prox(Y - gY / L, 1.0 / L), S)
                    D = A_new - Y
                    for m in range(M):
                        D[m][:, free[m]] = 0.0
                    if smooth(A_new, S, with_grad=False)[0] <= fY + np.sum(gY * D) + 0.5 * L * np.sum(D ** 2):
                        break
                    L *= 2.0
                t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
                Y = fill_free(A_new + ((t - 1.0) / t_next) * (A_new - A_prev), S)
                A_prev, t = A_new, t_next
                L *= 0.9
            A = A_prev
            S = self._update_view_weights(A)
            self.history_.append(smooth(fill_free(A, S), S, with_grad=False)[0] + self.c2 * l21(A))

        self.A_, self.S_ = fill_free(A, S), S
        return self

    @staticmethod
    def _update_view_weights(A):
        """Row m of S: least-squares convex weights of the other views' reconstruction matrices."""
        M = len(A)
        flat = A.reshape(M, -1)
        gram = flat @ flat.T
        S = np.zeros((M, M))
        for m in range(M):
            others = [l for l in range(M) if l != m]
            Q, q = gram[np.ix_(others, others)], gram[m, others]
            s = np.full(len(others), 1.0 / len(others))
            step = 1.0 / (2.0 * np.linalg.eigvalsh(Q).max() + 1e-12)
            for _ in range(200):
                s = project_to_simplex(s - step * 2.0 * (Q @ s - q))
            S[m, others] = s
        return S


def mkc_predictor(ctx, c1=1.0, c2=0.1):
    """Missing omic's contribution = its MKC reconstruction weights applied to the basis samples' contributions."""
    views = ("mb", "mt", "meta")
    index = ctx.contributions["mb"].index
    kernels, observed = [], []
    for view in views:
        obs = ctx.views[view].notna().all(axis=1).to_numpy()
        K = np.full((len(index), len(index)), np.nan)
        K[np.ix_(obs, obs)] = rbf_kernel(ctx.standardized(view, index[obs]))
        kernels.append(K)
        observed.append(obs)

    mkc = MKC_embd_ht(c1=c1, c2=c2).fit(kernels, observed)
    active = [int((np.linalg.norm(mkc.A_[m][:, obs], axis=1) > 1e-8).sum()) for m, obs in enumerate(observed)]
    ctx.diagnostics["MKC view weights s_ml (rows: mb, mt, meta)"] = np.round(mkc.S_, 2).tolist()
    ctx.diagnostics["MKC active basis samples per view"] = dict(zip(views, active))
    ctx.diagnostics["MKC objective after each outer round"] = np.round(mkc.history_, 3).tolist()

    predictions = {}
    for m, omic in enumerate(OMICS):
        basis_scores = ctx.contributions[omic].iloc[mkc.basis_].to_numpy()
        predictions[omic] = mkc.A_[m][:, ~observed[m]].T @ basis_scores
    return predictions

# ==========================================
# 5. PREDICTOR 3: SNF NEIGHBOURS
# ==========================================

def local_affinity(X, K=20, mu=0.5):
    """Scaled exponential affinity as in SNF: the kernel width adapts to each pair's neighbourhood density."""
    D = squareform(pdist(X))
    knn_mean = np.sort(D, axis=1)[:, 1:K + 1].mean(axis=1)
    width = np.maximum(mu * (knn_mean[:, None] + knn_mean[None, :] + D) / 3.0, np.finfo(float).eps)
    return np.exp(-(D ** 2) / (2.0 * width ** 2))


def snf_predictor(ctx, K=20, k_pred=20, T=20, mu=0.5):
    """Missing omic's contribution = weighted mean over the query's strongest fused links to training samples."""
    n_train = len(ctx.train_ids)
    predictions = {}
    for omic in OMICS:
        nodes = ctx.train_ids.append(ctx.missing(omic))
        affinities = [local_affinity(ctx.standardized(view, nodes), K, mu) for view in (OTHER[omic], "meta")]
        links = snf(affinities, K=K, T=T, minmax=False)[n_train:, :n_train]
        kth_strongest = -np.partition(-links, k_pred - 1, axis=1)[:, [k_pred - 1]]
        links = np.where(links >= kth_strongest, links, 0.0)
        train_scores = ctx.contributions[omic].loc[ctx.train_ids].to_numpy()
        predictions[omic] = (links / links.sum(axis=1, keepdims=True)) @ train_scores
    return predictions

# ==========================================
# 6. TUNING & EVALUATION
# ==========================================

def grid(**axes):
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


METHODS = {
    "Ridge (score space)": (ridge_predictor, [{}]),
    "MKC_embd(ht) (score space)": (mkc_predictor, grid(c1=[0.1, 10.0], c2=[0.01, 1.0, 10.0])),
    "SNF neighbours (score space)": (snf_predictor, grid(K=[10, 20, 40], k_pred=[10, 30, 60])),
}


def evaluate(split, distances, n_permutations):
    masked = split.masked_ids
    return mantel_test(split.ground_truth, distances.loc[masked, masked], n_permutations=n_permutations)


def tune(predictor, param_grid, train, meta, mb_cols, mt_cols):
    """Pick parameters on an inner masked split of the training samples only (same protocol as pipeline.py)."""
    if len(param_grid) == 1:
        return param_grid[0]
    inner = make_split(train, mb_cols, mt_cols, 0.25, RANDOM_STATE, RANDOM_STATE)
    scores = [
        evaluate(inner, score_space_distance(inner, meta, mb_cols, mt_cols, predictor, **params)[0], 0)[0]
        for params in param_grid
    ]
    return param_grid[int(np.argmax(scores))]


def compare_on_split(split, meta, mb_cols, mt_cols, n_permutations):
    start = time.perf_counter()
    reconstructed = reconstruct_knn(split.combined, split.train.index, n_neighbors=50)
    r, p = evaluate(split, pca_distance(split.train, reconstructed.loc[split.masked_ids]), n_permutations)
    rows = [{"Pipeline": BENCHMARK, "Mantel_r": r, "P_Value": p, "Params": "-",
             "Tune_s": 0.0, "Run_s": time.perf_counter() - start}]
    diagnostics = {}

    for name, (predictor, param_grid) in METHODS.items():
        start = time.perf_counter()
        params = tune(predictor, param_grid, split.train, meta, mb_cols, mt_cols)
        tuned = time.perf_counter()
        distances, diagnostics[name] = score_space_distance(split, meta, mb_cols, mt_cols, predictor, **params)
        r, p = evaluate(split, distances, n_permutations)
        rows.append({"Pipeline": name, "Mantel_r": r, "P_Value": p, "Params": params or "-",
                     "Tune_s": tuned - start, "Run_s": time.perf_counter() - tuned})
    return pd.DataFrame(rows), diagnostics


def parse_args():
    parser = argparse.ArgumentParser(description="Compare score-space predictors against the KNN50 benchmark.")
    parser.add_argument("--splits", type=int, default=10, help="extra random splits for the paired comparison (0 skips)")
    parser.add_argument("--permutations", type=int, default=999, help="Mantel permutations on the main split")
    return parser.parse_args()

# ==========================================
# 7. EXECUTION
# ==========================================

if __name__ == "__main__":
    args = parse_args()
    data, complete, mb_cols, mt_cols, metadata = load_data("microbiome.csv", "metabolome.csv", "metadata.csv")
    meta = metadata_features(metadata)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    print("\n" + "=" * 60)
    print("MAIN SPLIT (same train/test split and masks as pipeline.py)")
    print("=" * 60)
    main_split = make_split(complete, mb_cols, mt_cols, TEST_SIZE, RANDOM_STATE, RANDOM_STATE + 10_000)
    results, diagnostics = compare_on_split(main_split, meta, mb_cols, mt_cols, args.permutations)
    print(results.round(4).to_string(index=False))
    for name, info in diagnostics.items():
        for key, value in info.items():
            print(f"  {name} | {key}: {value}")

    if args.splits:
        print("\n" + "=" * 60)
        print(f"PAIRED COMPARISON OVER {args.splits} RANDOM SPLITS (tuned inside each split)")
        print("=" * 60)
        per_split = []
        for seed in range(args.splits):
            split = make_split(complete, mb_cols, mt_cols, TEST_SIZE, seed, seed + 10_000)
            split_results, _ = compare_on_split(split, meta, mb_cols, mt_cols, n_permutations=0)
            per_split.append(split_results.set_index("Pipeline")["Mantel_r"].rename(seed))
            print(f"split {seed}: " + "  ".join(f"{k.split(' (')[0]}={v:.3f}" for k, v in per_split[-1].items()))

        table = pd.concat(per_split, axis=1).T
        gain = table.sub(table[BENCHMARK], axis=0)
        summary = pd.DataFrame({
            "mean_r": table.mean(),
            "sd_r": table.std(),
            "mean_gain_vs_KNN50": gain.mean(),
            "sd_gain": gain.std(),
            "wins_vs_KNN50": (gain > 0).sum().astype(int),
        })
        print("\n" + summary.round(4).to_string())
