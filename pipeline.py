"""
Multi-Omic Evaluation Pipeline
Compares 4 Pipeline Strategies using Mantel Test Correlation (Mantel r)

1. Global KNN50 (Untouched Milestone 1 Benchmark)
2. Stratified KNN5 + SNF
3. Constrained MKC_embd(ht) -> Average Distance
4. Constrained MKC_embd(ht) -> SNF -> Fused Distance
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
PATIENT_GROUP_COL = "PATGROUPFINAL_C"

# ==========================================
# 1. DATA LOADING & EVALUATION TOOLS
# ==========================================

def load_data(microbiome_path="microbiome.csv", metabolome_path="metabolome.csv", metadata_path="metadata.csv"):
    microbiome = pd.read_csv(microbiome_path, index_col=0)
    metabolome = pd.read_csv(metabolome_path, index_col=0)
    metadata = pd.read_csv(metadata_path, index_col=0)

    # Ensure all data frames align
    common = microbiome.index.intersection(metabolome.index).intersection(metadata.index)
    microbiome = microbiome.loc[common]
    metabolome = metabolome.loc[common]
    metadata = metadata.loc[common]

    data = microbiome.join(metabolome, how="inner")
    mb_cols = list(microbiome.columns)
    mt_cols = list(metabolome.columns)

    mb_missing = data[mb_cols].isna().all(axis=1)
    mt_missing = data[mt_cols].isna().all(axis=1)
    
    complete = data.loc[~mb_missing & ~mt_missing].copy()
    
    print(f"Total Subjects: {len(data)}")
    print(f"Complete Subjects (Both Omics & Metadata): {len(complete)}")
    
    return data, complete, mb_cols, mt_cols, metadata

def make_mask(index, rng):
    ids = rng.permutation(np.asarray(index))
    half = len(ids) // 2
    return ids[:half], ids[half:]

def apply_mask(data, miss_mt, miss_mb, mb_cols, mt_cols):
    masked = data.copy()
    masked.loc[miss_mt, mt_cols] = np.nan
    masked.loc[miss_mb, mb_cols] = np.nan
    return masked

def pca_distance(train, evaluation):
    scaler = StandardScaler().fit(train)
    pca = PCA(n_components=2, random_state=RANDOM_STATE).fit(scaler.transform(train))
    coords = pca.transform(scaler.transform(evaluation))
    return pd.DataFrame(
        cdist(coords, coords, metric="euclidean"),
        index=evaluation.index,
        columns=evaluation.index,
    )

def mantel_test(matrix1, matrix2, n_permutations=999, random_state=RANDOM_STATE):
    matrix2 = matrix2.loc[matrix1.index, matrix1.columns]
    i, j = np.triu_indices(len(matrix1), k=1)
    v1 = matrix1.to_numpy()[i, j]
    v2 = matrix2.to_numpy()[i, j]
    
    if len(v1) == 0:
        return 0.0, 1.0
        
    r, _ = pearsonr(v1, v2)
    if n_permutations == 0:
        return float(r), np.nan

    rng = np.random.default_rng(random_state)
    m2 = matrix2.to_numpy()
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(m2))
        permuted = m2[np.ix_(perm, perm)]
        r_perm, _ = pearsonr(v1, permuted[i, j])
        count += r_perm >= r

    return float(r), (count + 1) / (n_permutations + 1)

# ==========================================
# 2. IMPUTATION ALGORITHMS (GLOBAL VS STRATIFIED)
# ==========================================

def reconstruct_knn(combined, reference_ids, n_neighbors=50):
    """Pipeline 1 Benchmark: Fits scaler and KNN strictly on complete training subjects globally."""
    reference = combined.loc[reference_ids]
    scaler = StandardScaler().fit(reference)
    reference_scaled = scaler.transform(reference)
    
    imputer = KNNImputer(n_neighbors=n_neighbors).fit(reference_scaled)
    imputed_scaled = imputer.transform(scaler.transform(combined))

    return pd.DataFrame(
        scaler.inverse_transform(imputed_scaled),
        index=combined.index,
        columns=combined.columns,
    )

def reconstruct_stratified_knn(combined, metadata, reference_ids, n_neighbors=5):
    """Pipelines 2-4: Fits scaler and KNN strictly within demographic disease cohorts to prevent cross-batch blurring."""
    reconstructed = combined.copy()
    groups = metadata.loc[combined.index, PATIENT_GROUP_COL].fillna("Unknown")
    
    for g in groups.unique():
        group_idx = groups[groups == g].index
        ref_group_idx = group_idx.intersection(reference_ids)
        
        if len(ref_group_idx) == 0:
            # Fallback for extremely rare edge cases
            reconstructed.loc[group_idx] = combined.loc[group_idx].fillna(combined.loc[reference_ids].median())
            continue
            
        group_data = combined.loc[group_idx]
        ref_data = combined.loc[ref_group_idx]
        
        scaler = StandardScaler().fit(ref_data)
        ref_scaled = scaler.transform(ref_data)
        
        k = min(n_neighbors, len(ref_group_idx))
        if k == 0: k = 1
            
        imputer = KNNImputer(n_neighbors=k).fit(ref_scaled)
        imputed_scaled = imputer.transform(scaler.transform(group_data))
        
        reconstructed.loc[group_idx] = scaler.inverse_transform(imputed_scaled)
        
    return reconstructed

# ==========================================
# 3. RBF KERNEL & SNF TOOLS
# ==========================================

def rbf_kernel(Z, sigma=None):
    D = squareform(pdist(Z, metric="euclidean"))
    if sigma is None:
        sigma = np.median(D[np.triu_indices_from(D, k=1)])
    return np.exp(-(D ** 2) / (2.0 * sigma ** 2))

def minmax_normalize(K):
    K = (K + K.T) / 2.0
    kmin, kmax = K.min(), K.max()
    W = (K - kmin) / (kmax - kmin) if kmax > kmin else np.zeros_like(K)
    np.fill_diagonal(W, 1.0)
    return W

def full_kernel(W):
    W = W.copy()
    np.fill_diagonal(W, 0.0)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = np.finfo(float).eps
    P = W / (2.0 * row_sums)
    np.fill_diagonal(P, 0.5)
    return P

def knn_kernel(W, K=20):
    n = W.shape[0]
    W_noself = W.copy()
    np.fill_diagonal(W_noself, -np.inf)
    idx = np.argpartition(-W_noself, K - 1, axis=1)[:, :K]
    S = np.zeros_like(W)
    rows = np.arange(n)[:, None]
    S[rows, idx] = W[rows, idx]
    S[np.arange(n), np.arange(n)] = W[np.arange(n), np.arange(n)]
    S /= S.sum(axis=1, keepdims=True)
    return S

def _symmetric_full_kernel(P):
    P = full_kernel(P)
    return (P + P.T) / 2.0

def snf(similarities, K=20, T=20, minmax=True):
    m = len(similarities)
    W = [minmax_normalize(k) if minmax else (k + k.T) / 2.0 for k in similarities]
    P = [_symmetric_full_kernel(w) for w in W]
    S = [knn_kernel(w, K) for w in W]

    for t in range(T):
        P_prev = [p.copy() for p in P]
        for v in range(m):
            other = sum(P_prev[k] for k in range(m) if k != v) / (m - 1)
            P[v] = _symmetric_full_kernel(S[v] @ other @ S[v].T)

    P_fused = sum(P) / m
    return (P_fused + P_fused.T) / 2.0

def fused_to_distance(P_fused):
    D = 1.0 - 2.0 * P_fused
    D = np.clip((D + D.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return D

# ==========================================
# 4. CONSTRAINED MULTI-VIEW KERNEL COMPLETION (MKC)
# ==========================================

class Constrained_MKC_embd_ht:
    """
    Implements MKC_embd(ht) constrained by demographic metadata.
    Missing subjects can only draw reconstruction weights from observed peers in their specific patient group.
    """
    def __init__(self, lambda1=1.0, lambda2=0.5, max_iter=50, gamma=0.01):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        self.gamma = gamma

    def _proximal_operator(self, A_step, step_size):
        A_next = np.zeros_like(A_step)
        for i in range(A_step.shape[0]):
            row_norm = np.linalg.norm(A_step[i, :])
            if row_norm > 0:
                scale = max(0, 1 - (step_size * self.lambda2) / row_norm)
                A_next[i, :] = scale * A_step[i, :]
        return A_next

    def _initialize_weights(self, N, observed_mask, allowed_mask):
        A = np.zeros((N, N))
        obs_idx = np.where(observed_mask)[0]
        mis_idx = np.where(~observed_mask)[0]
        
        for idx in obs_idx:
            A[idx, idx] = 1.0
            
        if len(mis_idx) > 0:
            # Random initialization restricted strictly to the allowed demographic mask
            rand_A = np.random.uniform(-1, 1, (N, len(mis_idx)))
            A[:, mis_idx] = rand_A * allowed_mask[:, mis_idx]
        return A

    def fit_transform(self, K1, K2, mask1, mask2, group_labels):
        N = K1.shape[0]
        group_labels = np.asarray(group_labels).ravel()

        # Matrix mask restricting basis combinations to members of the same group
        allowed_mask = (group_labels[:, None] == group_labels[None, :]).astype(float)
        
        A1 = self._initialize_weights(N, mask1, allowed_mask)
        A2 = self._initialize_weights(N, mask2, allowed_mask)
        
        kernel_norm = max(np.linalg.norm(K1, ord=2), np.linalg.norm(K2, ord=2))
        lipschitz = 2 * kernel_norm**2 + 4 * self.lambda1
        step_size = min(self.gamma, 1 / lipschitz) if lipschitz > 0 else self.gamma
        
        for iteration in range(self.max_iter):
            grad_A1 = 2 * (K1 @ A1 @ K1 - K1 @ K1) + 2 * self.lambda1 * (A1 - A2)
            grad_A2 = 2 * (K2 @ A2 @ K2 - K2 @ K2) + 2 * self.lambda1 * (A2 - A1)
            
            A1_step = A1 - step_size * grad_A1
            A2_step = A2 - step_size * grad_A2
            
            A1 = self._proximal_operator(A1_step, step_size)
            A2 = self._proximal_operator(A2_step, step_size)
            
            # Constraint projection: Erase any weight attempting to cross demographic borders
            A1 = A1 * allowed_mask
            A2 = A2 * allowed_mask

        K1_hat = A1.T @ K1 @ A1
        K2_hat = A2.T @ K2 @ A2
        return K1_hat, K2_hat

def grid_search_mkc_constrained(train_val, metadata_subset, mb_cols, mt_cols, grid_l1, grid_l2):
    """Tunes MKC hyperparameters utilizing the stratified pre-imputation to guide local variance."""
    print(f"\n--- Running Constrained MKC Hyperparameter Grid Search ---")
    t_train_ids, t_val_ids = train_test_split(train_val.index, test_size=0.25, random_state=RANDOM_STATE)
    
    t_train = train_val.loc[t_train_ids]
    t_val = train_val.loc[t_val_ids]
    
    rng = np.random.default_rng(RANDOM_STATE)
    miss_mt, miss_mb = make_mask(t_val.index, rng)
    
    masked_t_val = apply_mask(t_val, miss_mt, miss_mb, mb_cols, mt_cols)
    combined_tune = pd.concat([t_train, masked_t_val])
    masked_ids = np.concatenate([miss_mt, miss_mb])
    
    gt_dist = pca_distance(t_train, t_val.loc[masked_ids])
    
    # Pre-impute kernels using Stratified KNN to provide local variance for tuning
    Zm = reconstruct_stratified_knn(combined_tune[mb_cols], metadata_subset, t_train.index, n_neighbors=5).to_numpy()
    Zb = reconstruct_stratified_knn(combined_tune[mt_cols], metadata_subset, t_train.index, n_neighbors=5).to_numpy()
    
    K_mi = rbf_kernel(Zm)
    K_me = rbf_kernel(Zb)
    
    obs_micro = ~combined_tune.index.isin(miss_mb)
    obs_meta = ~combined_tune.index.isin(miss_mt)
    group_labels = metadata_subset.loc[combined_tune.index, PATIENT_GROUP_COL].fillna("Unknown").values
    
    best_r = -np.inf
    best_params = (1.0, 0.5)
    
    for l1 in grid_l1:
        for l2 in grid_l2:
            mkc = Constrained_MKC_embd_ht(lambda1=l1, lambda2=l2, max_iter=30)
            K_mi_hat, K_me_hat = mkc.fit_transform(K_mi, K_me, obs_micro, obs_meta, group_labels)
            
            W_mi = minmax_normalize(K_mi_hat)
            W_me = minmax_normalize(K_me_hat)
            S_avg = (W_mi + W_me) / 2.0
            D_mkc = 1.0 - S_avg
            np.fill_diagonal(D_mkc, 0.0)
            
            D_df = pd.DataFrame(D_mkc, index=combined_tune.index, columns=combined_tune.index)
            pred_dist = D_df.loc[masked_ids, masked_ids]
            
            r, _ = mantel_test(gt_dist, pred_dist, n_permutations=0)
            print(f"Tested lambda1={l1:5}, lambda2={l2:5} -> Mantel r = {r:.4f}")
            if r > best_r:
                best_r = r
                best_params = (l1, l2)
                
    print(f"Best Parameters Selected: lambda1={best_params[0]}, lambda2={best_params[1]}")
    return best_params

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================

if __name__ == "__main__":
    # 1. Load Data
    data, complete, mb_cols, mt_cols, metadata = load_data("microbiome.csv", "metabolome.csv", "metadata.csv")

    # 2. Strict Train/Test Separation 
    train_val_ids, test_ids = train_test_split(complete.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val = complete.loc[train_val_ids]
    test = complete.loc[test_ids]

    rng = np.random.default_rng(RANDOM_STATE + 10_000)
    miss_mt, miss_mb = make_mask(test.index, rng)
    masked_test = apply_mask(test, miss_mt, miss_mb, mb_cols, mt_cols)
    
    combined = pd.concat([train_val, masked_test])
    masked_ids = np.concatenate([miss_mt, miss_mb])

    # 3. Ground Truth Geometry
    print("\nCalculating Ground Truth Geometry...")
    gt_dist = pca_distance(train_val, test.loc[masked_ids])

    results = []

    # ---------------------------------------------------------
    # PIPELINE 1: Global KNN50 Imputation (Untouched Benchmark)
    # ---------------------------------------------------------
    print("\n--- Running Pipeline 1: Global KNN50 (Benchmark) ---")
    reconstructed_knn50 = reconstruct_knn(combined, train_val.index, n_neighbors=50)
    pred_knn50 = pca_distance(train_val, reconstructed_knn50.loc[masked_ids])
    r_knn50, p_knn50 = mantel_test(gt_dist, pred_knn50)
    results.append({"Pipeline": "1. Global KNN50 (Benchmark)", "Mantel_r": r_knn50, "P_Value": p_knn50})

    # ---------------------------------------------------------
    # PREPARATION FOR STRATIFIED PIPELINES (2, 3, 4)
    # ---------------------------------------------------------
    print("\n--- Preparing Stratified Imputations ---")
    Zm_strat = reconstruct_stratified_knn(combined[mb_cols], metadata, train_val.index, n_neighbors=5).to_numpy()
    Zb_strat = reconstruct_stratified_knn(combined[mt_cols], metadata, train_val.index, n_neighbors=5).to_numpy()
    
    K_mi = rbf_kernel(Zm_strat)
    K_me = rbf_kernel(Zb_strat)

    # ---------------------------------------------------------
    # PIPELINE 2: Stratified KNN5 + SNF
    # ---------------------------------------------------------
    print("--- Running Pipeline 2: Stratified KNN5 + SNF ---")
    P_fused_knn_strat = snf([K_mi, K_me], K=20, T=20, minmax=True)
    D_snf_knn_strat = fused_to_distance(P_fused_knn_strat)
    D_snf_knn_strat_df = pd.DataFrame(D_snf_knn_strat, index=combined.index, columns=combined.index)
    
    pred_snf_knn_strat = D_snf_knn_strat_df.loc[masked_ids, masked_ids]
    r_snf_knn_strat, p_snf_knn_strat = mantel_test(gt_dist, pred_snf_knn_strat)
    results.append({"Pipeline": "2. Stratified KNN5 + SNF", "Mantel_r": r_snf_knn_strat, "P_Value": p_snf_knn_strat})

    # ---------------------------------------------------------
    # MKC TUNING & EXECUTION (Pipelines 3 & 4)
    # ---------------------------------------------------------
    group_labels_combined = metadata.loc[combined.index, PATIENT_GROUP_COL].fillna("Unknown").values

    # Expanded grid limits to allow gradient descent to escape non-convex saddle points
    best_l1, best_l2 = grid_search_mkc_constrained(
        train_val, metadata.loc[train_val.index], mb_cols, mt_cols, 
        grid_l1=[0.1, 1.0, 10.0, 100.0], grid_l2=[0.01, 0.1, 1.0, 10.0]
    )

    print("\n--- Running Pipeline 3: Constrained MKC (Average) ---")
    obs_micro = ~combined.index.isin(miss_mb)
    obs_meta = ~combined.index.isin(miss_mt)

    mkc = Constrained_MKC_embd_ht(lambda1=best_l1, lambda2=best_l2, max_iter=50)
    K_mi_hat, K_me_hat = mkc.fit_transform(K_mi, K_me, obs_micro, obs_meta, group_labels_combined)

    W_mi = minmax_normalize(K_mi_hat)
    W_me = minmax_normalize(K_me_hat)

    S_avg = (W_mi + W_me) / 2.0
    D_mkc_avg = 1.0 - S_avg
    np.fill_diagonal(D_mkc_avg, 0.0)
    
    D_mkc_avg_df = pd.DataFrame(D_mkc_avg, index=combined.index, columns=combined.index)
    pred_mkc_avg = D_mkc_avg_df.loc[masked_ids, masked_ids]
    r_mkc_avg, p_mkc_avg = mantel_test(gt_dist, pred_mkc_avg)
    results.append({"Pipeline": "3. Constrained MKC (Avg)", "Mantel_r": r_mkc_avg, "P_Value": p_mkc_avg})

    print("--- Running Pipeline 4: Constrained MKC + SNF ---")
    P_fused_mkc = snf([W_mi, W_me], K=20, T=20, minmax=False)
    D_mkc_snf = fused_to_distance(P_fused_mkc)
    
    D_mkc_snf_df = pd.DataFrame(D_mkc_snf, index=combined.index, columns=combined.index)
    pred_mkc_snf = D_mkc_snf_df.loc[masked_ids, masked_ids]
    r_mkc_snf, p_mkc_snf = mantel_test(gt_dist, pred_mkc_snf)
    results.append({"Pipeline": "4. Constrained MKC + SNF", "Mantel_r": r_mkc_snf, "P_Value": p_mkc_snf})

    # ---------------------------------------------------------
    # FINAL RESULTS
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("PIPELINE COMPARISON GRADES (Mantel r vs Ground Truth)")
    print("="*60)
    results_df = pd.DataFrame(results).sort_values("Mantel_r", ascending=False)
    print(results_df.to_string(index=False))