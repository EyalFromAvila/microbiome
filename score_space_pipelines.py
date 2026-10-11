"""
Score-Space Pipelines
Five predictors of the missing omic, plus the Milestone 1 pipeline, graded with the same Mantel
test as pipeline.py.

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
4. miss-SNF (Gliozzo et al., 2025): all three views fused over all samples at once; a sample
   missing an omic enters that omic's network through a self-loop, and message passing from
   the views it does have recovers its links there
5. MKC+SNF: the Milestone 1 chain with its output moved into the score space. MKC completes
   each omic's kernel, SNF fuses the two completed kernels, and the fused network's strongest
   links average the training samples' scores

The Milestone 1 pipeline itself is evaluated next to them as "MKC+SNF kernel": the same completed
kernels and the same fusion, but the distance is read off the fused similarity (1 - similarity)
as proposed in Milestone 1, not measured in the score space. It shares the working MKC of
predictor 2; the version in pipeline.py is the old one, whose optimiser never moved.

Two reference points sit next to them:
- Global KNN50 (the Milestone 1 benchmark). It keeps the observed features untouched and PCA is
  linear, so it too gets the observed omic's part exactly and only fills in the missing part,
  with a 50-neighbour average. The gain of the predictors above over KNN50 is therefore the gain
  of a better predictor of the missing part, not of the score-space decomposition itself.
- Mean fill: the missing part is set to the training mean. This is the floor, the score of a
  pipeline that learns nothing about the missing omic.

Experiments (all results are also written to --out as CSV, plus one figure):
- Main split: same split and masks as pipeline.py, with Mantel r, p-value, tuning time and
  prediction time for every pipeline.
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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import ConstantInputWarning
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pipeline import (
    RANDOM_STATE,
    TEST_SIZE,
    apply_mask,
    full_kernel,
    fused_to_distance,
    knn_kernel,
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


def make_missing_split(complete, mb_cols, mt_cols, missing_fraction, seed):
    """
    Hide one omic in a fraction of ALL samples (half lose the metabolome, half the microbiome).

    The untouched samples are the only reference the pipelines may learn from, so a higher
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
# 2. SCORE SPACE (SHARED BY ALL PREDICTORS)
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

    def key(self):
        """Identifies the split: its samples, its training samples and who is missing which omic."""
        index = self.contributions["mb"].index
        return hash((tuple(index), tuple(self.train_ids), tuple(self.missing("mb")), tuple(self.missing("mt"))))


# Steps shared by several pipelines (an MKC fit, a fused network) are computed once per split.
# Each entry remembers how long it took, and a pipeline that reuses it is charged that time once,
# so reported run times are those of every pipeline run on its own.
_CACHE = {}              # key -> (value, its own seconds, keys of the cached steps it used)
_COMPUTING = []          # entries being computed: [seconds spent in nested entries, keys used]
_CHARGED = set()         # entries already counted for the pipeline being timed
_REUSED_SECONDS = [0.0]


def _charge(key):
    if key not in _CHARGED:
        _CHARGED.add(key)
        _, seconds, used = _CACHE[key]
        _REUSED_SECONDS[0] += seconds
        for dependency in used:
            _charge(dependency)


def cached(key, compute):
    if _COMPUTING:
        _COMPUTING[-1][1].append(key)
    if key in _CACHE:
        _charge(key)
        return _CACHE[key][0]
    _COMPUTING.append([0.0, []])
    start = time.perf_counter()
    value = compute()
    elapsed = time.perf_counter() - start
    nested, used = _COMPUTING.pop()
    _CACHE[key] = (value, elapsed - nested, used)
    _CHARGED.add(key)
    if _COMPUTING:
        _COMPUTING[-1][0] += elapsed
    return value


def timed(action):
    """Result of `action` and its seconds, including the cost of every cached step it reused."""
    _CHARGED.clear()
    _REUSED_SECONDS[0] = 0.0
    start = time.perf_counter()
    result = action()
    return result, time.perf_counter() - start + _REUSED_SECONDS[0]


def score_space_distance(split, meta, mb_cols, mt_cols, predictor, **params):
    """
    Distance matrix over all samples of the split.

    A predictor returns the missing score contributions per omic, and distances are then Euclidean
    in the score space. A pipeline that defines its own distance returns the matrix directly.
    """
    space = ScoreSpace(split.train, mb_cols, mt_cols)
    combined = split.combined
    views = {
        "mb": omic_features(combined[mb_cols], "mb"),
        "mt": omic_features(combined[mt_cols], "mt"),
        "meta": meta.loc[combined.index],
    }
    ctx = Context(views, space.contributions(combined), split.train.index)
    predictions = predictor(ctx, **params)
    if isinstance(predictions, pd.DataFrame):
        return predictions, ctx.diagnostics
    for omic, predicted in predictions.items():
        ctx.contributions[omic].loc[ctx.missing(omic)] = predicted
    coords = ctx.contributions["mb"] + ctx.contributions["mt"]
    distances = pd.DataFrame(cdist(coords, coords), index=coords.index, columns=coords.index)
    return distances, ctx.diagnostics


def mean_predictor(ctx):
    """Floor: the missing omic's contribution is its training mean (what mean imputation of that omic gives)."""
    return {
        omic: np.tile(ctx.contributions[omic].loc[ctx.train_ids].mean().to_numpy(), (len(ctx.missing(omic)), 1))
        for omic in OMICS
    }

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


MKC_VIEWS = ("mb", "mt", "meta")


def fit_mkc(ctx, c1, c2):
    """MKC over the two omics and metadata; one fit per split and (c1, c2), shared by every pipeline built on it."""
    def compute():
        index = ctx.contributions["mb"].index
        kernels, observed = [], []
        for view in MKC_VIEWS:
            obs = ctx.views[view].notna().all(axis=1).to_numpy()
            K = np.full((len(index), len(index)), np.nan)
            K[np.ix_(obs, obs)] = rbf_kernel(ctx.standardized(view, index[obs]))
            kernels.append(K)
            observed.append(obs)
        return MKC_embd_ht(c1=c1, c2=c2).fit(kernels, observed), kernels, observed

    return cached(("mkc", ctx.key(), c1, c2), compute)


def mkc_predictor(ctx, c1=1.0, c2=0.1):
    """Missing omic's contribution = its MKC reconstruction weights applied to the basis samples' contributions."""
    views = MKC_VIEWS
    mkc, _, observed = fit_mkc(ctx, c1, c2)
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
    k_pred = min(k_pred, n_train)
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
# 6. PREDICTOR 4: MISS-SNF
# ==========================================

MISS_SNF_SELF_LOOP = {"one": 1.0, "equidistant": 0.5, "zero": 0.0}


def _normalized(P):
    P = full_kernel(P)
    return (P + P.T) / 2.0


def miss_snf(affinities, observed, K=20, T=20, strategy="one"):
    """
    SNF for partial data (Gliozzo et al., 2025).

    affinities[v] is the affinity matrix of view v among the samples observed in it and
    observed[v] the boolean mask of those samples; the result is one fused n x n network.
    A sample i with view s entirely missing has no row in W(s), P(s), S(s), so the update
    P(s) <- S(s) @ mean(P(v != s)) @ S(s).T could not reach it. miss-SNF writes that row by hand:
      - "one":         self-loop 1, nothing else. Row i of the update becomes
                       P(s)(i, j) = sum_l mean_P(i, l) * S(s)(j, l): i is linked to j in the
                       missing view when its neighbours in the other views are j's neighbours in s.
      - "equidistant": self-loop 0.5 and 0.5 / (n - 1) to every other sample.
      - "zero":        no self-loop; the sample is ignored in view s and the fused network
                       averages each pair over the views where both samples are observed.
    """
    n, m = len(observed[0]), len(affinities)
    self_loop = MISS_SNF_SELF_LOOP[strategy]
    partial = [np.where(~np.asarray(obs))[0] for obs in observed]
    P, S = [], []
    for A, obs, miss in zip(affinities, observed, partial):
        W = np.zeros((n, n))
        W[np.ix_(obs, obs)] = (A + A.T) / 2.0
        if strategy == "equidistant":
            W[miss, :] = W[:, miss] = 0.5 / (n - 1)
        W[miss, miss] = self_loop
        global_net = _normalized(W)
        global_net[miss, :], global_net[:, miss] = W[miss, :], W[:, miss]
        local_net = W.copy()
        local_net[miss, miss] = 1.0                      # keeps the row normalisation in knn_kernel finite
        local_net = knn_kernel(local_net, K)
        local_net[miss, :] = W[miss, :]                  # partial rows are set by the strategy, not by KNN
        P.append(global_net)
        S.append(local_net)

    for _ in range(T):
        previous = P
        P = []
        for v in range(m):
            others = sum(previous[u] for u in range(m) if u != v) / (m - 1)
            updated = _normalized(S[v] @ others @ S[v].T)
            if strategy == "zero":
                updated[partial[v], :] = updated[:, partial[v]] = 0.0
            P.append(updated)

    if strategy == "zero":
        available = sum(np.outer(obs, obs).astype(float) for obs in observed)
        fused = sum(P) / np.maximum(available, 1.0)
    else:
        fused = sum(P) / m
    return (fused + fused.T) / 2.0


def miss_snf_predictor(ctx, K=20, k_pred=20, T=20, mu=0.5, strategy="one"):
    """
    Missing omic's contribution = weighted mean over the query's strongest miss-SNF links to training samples.

    Unlike snf_predictor, one network is fused over all samples and all three views, including the
    omic being predicted: its structure among the samples that have it shapes the fused links.
    """
    views = ("mb", "mt", "meta")
    index = ctx.contributions["mb"].index
    observed = [ctx.views[view].notna().all(axis=1).to_numpy() for view in views]
    affinities = [
        local_affinity(ctx.standardized(view, index[obs]), K, mu) for view, obs in zip(views, observed)
    ]
    fused = pd.DataFrame(miss_snf(affinities, observed, K=K, T=T, strategy=strategy), index=index, columns=index)
    return strongest_link_average(ctx, fused, k_pred)


def strongest_link_average(ctx, fused, k_pred):
    """Missing omic's contribution = weighted mean over the sample's k_pred strongest links to training samples."""
    k_pred = min(k_pred, len(ctx.train_ids))
    predictions = {}
    for omic in OMICS:
        links = fused.loc[ctx.missing(omic), ctx.train_ids].to_numpy()
        kth_strongest = -np.partition(-links, k_pred - 1, axis=1)[:, [k_pred - 1]]
        links = np.where(links >= kth_strongest, links, 0.0)
        weights = links / np.maximum(links.sum(axis=1, keepdims=True), np.finfo(float).eps)
        predictions[omic] = weights @ ctx.contributions[omic].loc[ctx.train_ids].to_numpy()
    return predictions

# ==========================================
# 7. MILESTONE 1 PIPELINE: MKC COMPLETION, THEN SNF
# ==========================================

def mkc_snf_network(ctx, c1, c2, K, T=20):
    """
    The Milestone 1 chain: incomplete kernels -> MKC completion -> SNF fusion.

    MKC completes each omic's kernel as A^T K A, which gives every sample a row in both kernels,
    including the kernel of the omic it lacks. SNF then fuses the two completed kernels into one
    similarity network over all samples.
    """
    def compute():
        mkc, kernels, _ = fit_mkc(ctx, c1, c2)
        completed = []
        for m in range(len(OMICS)):
            basis_kernel = kernels[m][np.ix_(mkc.basis_, mkc.basis_)]
            completed.append(mkc.A_[m].T @ basis_kernel @ mkc.A_[m])
        return snf(completed, K=K, T=T, minmax=True)

    return cached(("mkc-snf", ctx.key(), c1, c2, K, T), compute)


def mkc_snf_kernel_distance(ctx, c1=1.0, c2=0.1, K=20):
    """Milestone 1 pipeline as proposed: the distance is 1 - fused similarity, not a score-space distance."""
    index = ctx.contributions["mb"].index
    return pd.DataFrame(fused_to_distance(mkc_snf_network(ctx, c1, c2, K)), index=index, columns=index)


def mkc_snf_predictor(ctx, c1=1.0, c2=0.1, K=20, k_pred=20):
    """Same fused network, used to predict the missing omic's contribution from the strongest links."""
    index = ctx.contributions["mb"].index
    fused = pd.DataFrame(mkc_snf_network(ctx, c1, c2, K), index=index, columns=index)
    return strongest_link_average(ctx, fused, k_pred)

# ==========================================
# 8. TUNING & EVALUATION
# ==========================================

def grid(**axes):
    return [dict(zip(axes, values)) for values in itertools.product(*axes.values())]


METHODS = {
    "Mean fill (score space floor)": (mean_predictor, [{}]),
    "Ridge (score space)": (ridge_predictor, [{}]),
    "MKC_embd(ht) (score space)": (mkc_predictor, grid(c1=[0.1, 10.0], c2=[0.01, 1.0, 10.0])),
    "SNF neighbours (score space)": (snf_predictor, grid(K=[10, 20, 40], k_pred=[10, 30, 60])),
    "miss-SNF (score space)": (
        miss_snf_predictor,
        grid(strategy=["one", "equidistant"], K=[10, 20, 40], k_pred=[10, 30, 60]),
    ),
    "MKC+SNF (score space)": (
        mkc_snf_predictor,
        grid(c1=[0.1, 10.0], c2=[0.01, 1.0, 10.0], K=[10, 20, 40], k_pred=[10, 30, 60]),
    ),
    "MKC+SNF kernel (Milestone 1 pipeline)": (
        mkc_snf_kernel_distance,
        grid(c1=[0.1, 10.0], c2=[0.01, 1.0, 10.0], K=[10, 20, 40]),
    ),
}


def short(name):
    return name.split(" (")[0]


def evaluate(split, distances, n_permutations, ids=None):
    """Mantel r and p-value over `ids` (default: the samples missing an omic)."""
    ids = split.masked_ids if ids is None else ids
    return mantel_test(split.ground_truth.loc[ids, ids], distances.loc[ids, ids], n_permutations=n_permutations)


def tune(predictor, param_grid, train, meta, mb_cols, mt_cols):
    """Pick parameters on an inner masked split of the training samples only (same protocol as pipeline.py)."""
    if len(param_grid) == 1:
        return param_grid[0]
    inner = make_split(train, mb_cols, mt_cols, 0.25, RANDOM_STATE, RANDOM_STATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        scores = [
            evaluate(inner, score_space_distance(inner, meta, mb_cols, mt_cols, predictor, **params)[0], 0)[0]
            for params in param_grid
        ]
    # A degenerate setting (all distances equal) has no Mantel r; it must never be selected.
    return param_grid[int(np.argmax(np.nan_to_num(scores, nan=-np.inf)))]


def run_all(split, meta, mb_cols, mt_cols, fixed_params=None):
    """
    Distance matrices over all samples of the split, from the benchmark and from every method.

    Tune_s is the time spent choosing parameters and Run_s the time from the masked data to the
    distance matrix. Neither includes the Mantel test, so the timings compare the pipelines only.
    With fixed_params (name -> parameters) tuning is skipped.
    """
    _CACHE.clear()
    start = time.perf_counter()
    reconstructed = reconstruct_knn(split.combined, split.train.index, n_neighbors=50)
    distances = pca_distance(split.train, reconstructed)
    runs = [{"Pipeline": BENCHMARK, "distances": distances, "params": {}, "diagnostics": {},
             "Tune_s": 0.0, "Run_s": time.perf_counter() - start}]

    for name, (predictor, param_grid) in METHODS.items():
        if fixed_params is None:
            params, tune_s = timed(lambda: tune(predictor, param_grid, split.train, meta, mb_cols, mt_cols))
        else:
            params, tune_s = fixed_params[name], 0.0
        (distances, diagnostics), run_s = timed(
            lambda: score_space_distance(split, meta, mb_cols, mt_cols, predictor, **params)
        )
        runs.append({"Pipeline": name, "distances": distances, "params": params, "diagnostics": diagnostics,
                     "Tune_s": tune_s, "Run_s": run_s})
    return runs


def compare_on_split(split, meta, mb_cols, mt_cols, n_permutations):
    """Score every pipeline on the samples missing an omic; returns the table, diagnostics and chosen parameters."""
    rows, diagnostics, chosen = [], {}, {}
    for run in run_all(split, meta, mb_cols, mt_cols):
        r, p = evaluate(split, run["distances"], n_permutations)
        rows.append({"Pipeline": run["Pipeline"], "Mantel_r": r, "P_Value": p, "Params": run["params"] or "-",
                     "Tune_s": run["Tune_s"], "Run_s": run["Run_s"]})
        if run["Pipeline"] != BENCHMARK:
            diagnostics[run["Pipeline"]] = run["diagnostics"]
            chosen[run["Pipeline"]] = run["params"]
    return pd.DataFrame(rows), diagnostics, chosen

# ==========================================
# 9. MISSING-PERCENTAGE SWEEP
# ==========================================

def missing_sweep(complete, meta, mb_cols, mt_cols, fractions, n_seeds, fixed_params):
    """
    Mantel r of every pipeline as the share of samples missing an omic grows.

    r_all is scored over all samples, complete ones included, which is how a submitted n x n
    matrix is graded. r_masked is scored over the samples missing an omic only, which isolates
    how well the missing part is predicted. Run_s includes tuning when fixed_params is None.
    """
    rows = []
    for fraction in fractions:
        for seed in range(n_seeds):
            split = make_missing_split(complete, mb_cols, mt_cols, fraction, seed)
            everyone = split.combined.index
            for run in run_all(split, meta, mb_cols, mt_cols, fixed_params):
                rows.append({
                    "Missing_fraction": fraction,
                    "Seed": seed,
                    "Pipeline": run["Pipeline"],
                    "Complete_samples": len(split.train),
                    "r_all": evaluate(split, run["distances"], 0, everyone)[0],
                    "r_masked": evaluate(split, run["distances"], 0)[0],
                    "Run_s": run["Tune_s"] + run["Run_s"],
                })
            latest = rows[-(len(METHODS) + 1):]
            print(f"missing {fraction:.0%} seed {seed} (r_all): "
                  + "  ".join(f"{short(row['Pipeline'])}={row['r_all']:.3f}" for row in latest))
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, (column, title) in zip(axes, panels):
        stats = sweep.groupby(["Pipeline", "Missing_fraction"], sort=False)[column].agg(["mean", "std"])
        for name in sweep["Pipeline"].unique():
            line = stats.loc[name]
            ax.errorbar(line.index * 100, line["mean"], yerr=line["std"].fillna(0.0),
                        marker="o", capsize=3, linewidth=1.5, label=short(name))
        ax.set_title(title)
        ax.set_xlabel("Samples missing one omic (%)")
        ax.set_ylabel("Mantel r vs ground truth")
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"figure saved to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare score-space predictors against the KNN50 benchmark.")
    parser.add_argument("--splits", type=int, default=10, help="extra random splits for the paired comparison (0 skips)")
    parser.add_argument("--permutations", type=int, default=999, help="Mantel permutations on the main split")
    parser.add_argument("--sweep-seeds", type=int, default=3, help="repeats per missing fraction in the sweep (0 skips)")
    parser.add_argument("--fractions", default="0.1,0.2,0.4,0.6,0.8",
                        help="comma-separated shares of all samples that lose one omic in the sweep")
    parser.add_argument("--sweep-retune", action="store_true",
                        help="tune inside every sweep split instead of reusing the main split's parameters (much slower)")
    parser.add_argument("--out", default="results", help="folder for the CSV tables and the figure")
    parser.add_argument("--methods", default="",
                        help="comma-separated parts of pipeline names to run, e.g. \"MKC+SNF,Ridge\" "
                             "(default: all; the KNN50 benchmark always runs)")
    return parser.parse_args()

# ==========================================
# 10. EXECUTION
# ==========================================

if __name__ == "__main__":
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [part.strip().lower() for part in args.methods.split(",") if part.strip()]
    if wanted:
        METHODS = {name: method for name, method in METHODS.items() if any(part in name.lower() for part in wanted)}
        if not METHODS:
            raise SystemExit("--methods matched no pipeline")
        print("Pipelines: " + ", ".join([BENCHMARK, *METHODS]))
    order = [BENCHMARK, *METHODS]
    data, complete, mb_cols, mt_cols, metadata = load_data("microbiome.csv", "metabolome.csv", "metadata.csv")
    meta = metadata_features(metadata)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    print("\n" + "=" * 60)
    print("MAIN SPLIT (same train/test split and masks as pipeline.py)")
    print("=" * 60)
    main_split = make_split(complete, mb_cols, mt_cols, TEST_SIZE, RANDOM_STATE, RANDOM_STATE + 10_000)
    results, diagnostics, main_params = compare_on_split(main_split, meta, mb_cols, mt_cols, args.permutations)
    print(results.round(4).to_string(index=False))
    print("(Tune_s: choosing parameters. Run_s: masked data to distance matrix. Seconds, Mantel test excluded.)")
    for name, info in diagnostics.items():
        for key, value in info.items():
            print(f"  {name} | {key}: {value}")
    results.to_csv(out / "main_split.csv", index=False)

    if args.splits:
        print("\n" + "=" * 60)
        print(f"PAIRED COMPARISON OVER {args.splits} RANDOM SPLITS (tuned inside each split)")
        print("=" * 60)
        per_split = []
        for seed in range(args.splits):
            split = make_split(complete, mb_cols, mt_cols, TEST_SIZE, seed, seed + 10_000)
            split_results, _, _ = compare_on_split(split, meta, mb_cols, mt_cols, n_permutations=0)
            per_split.append(split_results.drop(columns="P_Value").assign(Split=seed))
            print(f"split {seed}: " + "  ".join(
                f"{short(name)}={r:.3f}" for name, r in zip(split_results["Pipeline"], split_results["Mantel_r"])
            ))

        paired = pd.concat(per_split, ignore_index=True)
        table = paired.pivot(index="Split", columns="Pipeline", values="Mantel_r")[order]
        gain = table.sub(table[BENCHMARK], axis=0)
        seconds = paired.groupby("Pipeline")[["Tune_s", "Run_s"]].mean().loc[order]
        summary = pd.DataFrame({
            "mean_r": table.mean(),
            "sd_r": table.std(),
            "mean_gain_vs_KNN50": gain.mean(),
            "sd_gain": gain.std(),
            "wins_vs_KNN50": (gain > 0).sum().astype(int),
            "mean_tune_s": seconds["Tune_s"],
            "mean_run_s": seconds["Run_s"],
        })
        print("\n" + summary.round(4).to_string())
        paired.to_csv(out / "paired_splits.csv", index=False)
        summary.to_csv(out / "paired_summary.csv")

    if args.sweep_seeds:
        fractions = [float(value) for value in args.fractions.split(",")]
        if not all(0.0 < fraction < 1.0 for fraction in fractions):
            raise SystemExit("--fractions must lie strictly between 0 and 1")
        print("\n" + "=" * 60)
        print(f"MISSING-PERCENTAGE SWEEP ({args.sweep_seeds} repeats per percentage, "
              + ("tuned inside each split)" if args.sweep_retune else "main-split parameters)"))
        print("=" * 60)
        sweep = missing_sweep(complete, meta, mb_cols, mt_cols, fractions, args.sweep_seeds,
                              None if args.sweep_retune else main_params)
        grouped = sweep.groupby(["Missing_fraction", "Pipeline"])
        for column, title in (("r_all", "Mean Mantel r over ALL samples (as graded)"),
                              ("r_masked", "Mean Mantel r over samples missing an omic only"),
                              ("Run_s", "Mean run time in seconds")):
            print(f"\n{title}; rows: share of samples missing one omic")
            print(grouped[column].mean().unstack()[order].rename(columns=short).round(4).to_string())
        sweep.to_csv(out / "missing_sweep.csv", index=False)
        grouped[["r_all", "r_masked", "Run_s"]].agg(["mean", "std"]).to_csv(out / "missing_sweep_summary.csv")
        plot_sweep(sweep, out / "missing_sweep.png")

    print(f"\nTables saved to {out}/")