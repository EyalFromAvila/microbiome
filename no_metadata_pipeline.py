"""
Multi-Omic Evaluation Pipeline
Compares 4 Pipeline Strategies using Mantel Test Correlation (Mantel r)
1. KNN50
2. KNN50 + SNF (Updated from Median)
3. MKC_embd(ht) -> Average Distance
4. MKC_embd(ht) -> SNF -> Fused Distance

================================================================================
PIPELINE SUMMARY: RAW DATA TO MKC TERMINATION
================================================================================

Step 1: Preprocessing and the Missing Data Challenge
* The pipeline ingests features from the MetaCardis cohort datasets and standardizes 
  them using a StandardScaler[cite: 5].
* To evaluate the model, the algorithm isolates the 1,042 subjects with complete 
  data and artificially masks one modality in 40% of the samples[cite: 5]. 
* This specifically mimics the real-world challenge of the full cohort, where 348 
  out of 1,738 subjects are entirely missing microbiome data and another 348 are 
  entirely missing metabolome data[cite: 5].

Step 2: Initial RBF Kernel Calculation
* The pipeline converts the standardized features into initial similarity grids 
  using an RBF kernel formula[cite: 5]. 
* The Gaussian width parameter (sigma) is dynamically set to the median of the 
  pairwise distances between subjects in that specific modality[cite: 5].

Step 3: Initialization of Reconstruction Weights (Difficulty Highlight)
* The algorithm defines an N x N reconstruction weight matrix (A^m) for each 
  modality, dictating how each subject is mathematically built from the observed 
  subjects[cite: 5].
* The Identity Block: The sub-matrix corresponding to the observed subjects is 
  initialized as an Identity matrix[cite: 5].
* The Random Block (Key Concept): The 348 missing subjects do not start as zero 
  vectors. Instead, their weights are initialized using a uniform random 
  distribution between -1 and 1[cite: 5]. This is mathematically necessary to 
  break symmetry and provide a sloped landscape for the non-convex optimization 
  process to escape saddle points[cite: 5].

Step 4: The Dual-Optimization Loop (Difficulty Highlight)
* Within-View Grounding: Within-view reconstruction is unique to each subject and 
  serves to constrain the imputed data to the specific non-linear structural space 
  of the target modality[cite: 5].
* Between-View Weight of 1.0 (Key Concept): Because this pipeline only uses two 
  omics modalities, the algorithm is forced to assign a convex combination weight 
  of exactly 1.0 to the secondary modality[cite: 5]. This means 100% of the 
  structural guidance for a missing subject comes directly from their relationships 
  in the other available omic[cite: 5]. 
* Row-Level Sparsity via l_2,1 Norm (Key Concept): The proximal gradient descent 
  operator applies the l_2,1 norm regularization as a strict threshold, shrinking 
  entire rows of the weight matrix to exactly zero[cite: 5]. This actively 
  destroys parts of the initial Identity block to force the selection of a small, 
  localized "basis set" of representative subjects[cite: 5].

Step 5: Final Kernel Reconstruction (Difficulty Highlight)
* The MKC algorithm terminates by calculating the fully imputed similarity grids 
  using the formula: K_hat = (A^m)^T * K^m * A^m[cite: 5].
* Final Matrix Multiplication (Key Concept): Multiplying a single column of A^m 
  by the kernel does not yield the final similarities. Reconstructing the 
  similarity score between any two specific subjects requires the transposed 
  weights of the first subject, multiplied by the original kernel, multiplied by 
  the weights of the second subject[cite: 5].
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

# ==========================================
# 1. DATA LOADING & EVALUATION TOOLS
# ==========================================

def load_data(microbiome_path, metabolome_path):
    microbiome = pd.read_csv(microbiome_path, index_col=0)
    metabolome = pd.read_csv(metabolome_path, index_col=0)

    data = microbiome.join(metabolome, how="inner")
    mb_cols = list(microbiome.columns)
    mt_cols = list(metabolome.columns)

    mb_missing = data[mb_cols].isna().all(axis=1)
    mt_missing = data[mt_cols].isna().all(axis=1)
    
    complete = data.loc[~mb_missing & ~mt_missing].copy()
    
    print(f"Total Subjects: {len(data)}")
    print(f"Complete Subjects (Both Omics): {len(complete)}")
    
    return data, complete, mb_cols, mt_cols

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
# 2. KNN IMPUTATION (LEAKAGE-FREE)
# ==========================================

def reconstruct_knn(combined, reference_ids, n_neighbors=50):
    """Fit scaler and KNN imputer strictly on complete training subjects."""
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
# 4. MULTI-VIEW KERNEL COMPLETION (MKC_embd_ht)
# ==========================================

class MKC_embd_ht:
    """
    Implements the heterogeneous embeddings variant of MKC (Algorithm 1)[cite: 8].
    Specifically designed for views with different kernel functions/eigen-spectra[cite: 8].
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

    def _initialize_weights(self, N, observed_mask):
        A = np.zeros((N, N))
        obs_idx = np.where(observed_mask)[0]
        mis_idx = np.where(~observed_mask)[0]
        for idx in obs_idx:
            A[idx, idx] = 1.0
        if len(mis_idx) > 0:
            A[:, mis_idx] = np.random.uniform(-1, 1, (N, len(mis_idx)))
        return A

    def fit_transform(self, K1, K2, mask1, mask2):
        N = K1.shape[0]
        A1 = self._initialize_weights(N, mask1)
        A2 = self._initialize_weights(N, mask2)
        
        kernel_norm = max(np.linalg.norm(K1, ord=2), np.linalg.norm(K2, ord=2))
        lipschitz = 2 * kernel_norm**2 + 4 * self.lambda1
        step_size = min(self.gamma, 1 / lipschitz) if lipschitz > 0 else self.gamma
        
        for iteration in range(self.max_iter):
            # MKC_embd_ht gradient penalty enforces similarity between embeddings A1 and A2[cite: 5, 8]
            grad_A1 = 2 * (K1 @ A1 @ K1 - K1 @ K1) + 2 * self.lambda1 * (A1 - A2)
            grad_A2 = 2 * (K2 @ A2 @ K2 - K2 @ K2) + 2 * self.lambda1 * (A2 - A1)
            
            A1_step = A1 - step_size * grad_A1
            A2_step = A2 - step_size * grad_A2
            
            A1 = self._proximal_operator(A1_step, step_size)
            A2 = self._proximal_operator(A2_step, step_size)

        K1_hat = A1.T @ K1 @ A1
        K2_hat = A2.T @ K2 @ A2
        return K1_hat, K2_hat


def grid_search_mkc(train_val, mb_cols, mt_cols, grid_l1, grid_l2):
    """
    Tunes MKC hyperparameters via internal validation split on the train_val subset
    to strictly prevent data leakage from the final test set.
    """
    print(f"\n--- Running MKC Hyperparameter Grid Search ---")
    t_train, t_val = train_test_split(train_val, test_size=0.25, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    miss_mt, miss_mb = make_mask(t_val.index, rng)
    
    masked_t_val = apply_mask(t_val, miss_mt, miss_mb, mb_cols, mt_cols)
    combined_tune = pd.concat([t_train, masked_t_val])
    masked_ids = np.concatenate([miss_mt, miss_mb])
    
    # Validation ground truth
    gt_dist = pca_distance(t_train, t_val.loc[masked_ids])
    
    # Pre-impute kernels using KNN to provide localized variance 
    Zm = reconstruct_knn(combined_tune[mb_cols], t_train.index, n_neighbors=50).to_numpy()
    Zb = reconstruct_knn(combined_tune[mt_cols], t_train.index, n_neighbors=50).to_numpy()
    
    K_mi = rbf_kernel(Zm)
    K_me = rbf_kernel(Zb)
    
    obs_micro = ~combined_tune.index.isin(miss_mb)
    obs_meta = ~combined_tune.index.isin(miss_mt)
    
    best_r = -np.inf
    best_params = (1.0, 0.5)
    
    for l1 in grid_l1:
        for l2 in grid_l2:
            mkc = MKC_embd_ht(lambda1=l1, lambda2=l2, max_iter=30)
            K_mi_hat, K_me_hat = mkc.fit_transform(K_mi, K_me, obs_micro, obs_meta)
            
            W_mi = minmax_normalize(K_mi_hat)
            W_me = minmax_normalize(K_me_hat)
            S_avg = (W_mi + W_me) / 2.0
            D_mkc = 1.0 - S_avg
            np.fill_diagonal(D_mkc, 0.0)
            
            D_df = pd.DataFrame(D_mkc, index=combined_tune.index, columns=combined_tune.index)
            pred_dist = D_df.loc[masked_ids, masked_ids]
            
            r, _ = mantel_test(gt_dist, pred_dist, n_permutations=0)
            print(f"Tested lambda1={l1}, lambda2={l2} -> Mantel r = {r:.4f}")
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
    data, complete, mb_cols, mt_cols = load_data("microbiome.csv", "metabolome.csv")

    # 2. Strict Train/Test Separation 
    train_val_ids, test_ids = train_test_split(
        complete.index, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
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
    # PIPELINE 1: KNN50 Imputation
    # ---------------------------------------------------------
    print("\n--- Running Pipeline 1: KNN50 ---")
    reconstructed_knn = reconstruct_knn(combined, train_val.index, n_neighbors=50)
    pred_knn = pca_distance(train_val, reconstructed_knn.loc[masked_ids])
    r_knn, p_knn = mantel_test(gt_dist, pred_knn)
    results.append({"Pipeline": "KNN50", "Mantel_r": r_knn, "P_Value": p_knn})

    # ---------------------------------------------------------
    # PREPARATION FOR PIPELINES 2, 3, 4: KNN Kernels
    # ---------------------------------------------------------
    # Impute via KNN (replaces Median to preserve local SNF structures)[cite: 5]
    Zm_knn = reconstruct_knn(combined[mb_cols], train_val.index, n_neighbors=50).to_numpy()
    Zb_knn = reconstruct_knn(combined[mt_cols], train_val.index, n_neighbors=50).to_numpy()
    
    K_mi = rbf_kernel(Zm_knn)
    K_me = rbf_kernel(Zb_knn)

    # ---------------------------------------------------------
    # PIPELINE 2: KNN + SNF
    # ---------------------------------------------------------
    print("--- Running Pipeline 2: KNN + SNF ---")
    P_fused_knn = snf([K_mi, K_me], K=20, T=20, minmax=True)
    D_snf_knn = fused_to_distance(P_fused_knn)
    D_snf_knn_df = pd.DataFrame(D_snf_knn, index=combined.index, columns=combined.index)
    
    pred_snf_knn = D_snf_knn_df.loc[masked_ids, masked_ids]
    r_snf_knn, p_snf_knn = mantel_test(gt_dist, pred_snf_knn)
    results.append({"Pipeline": "KNN + SNF", "Mantel_r": r_snf_knn, "P_Value": p_snf_knn})

    # ---------------------------------------------------------
    # MKC TUNING & EXECUTION (Pipelines 3 & 4)
    # ---------------------------------------------------------
    # Grid Search defined per guidelines[cite: 5, 8]
    best_l1, best_l2 = grid_search_mkc(train_val, mb_cols, mt_cols, grid_l1=[0.1, 1.0, 10.0], grid_l2=[0.1, 0.5, 1.0])

    print("\n--- Running Pipeline 3: MKC (Average) ---")
    obs_micro = ~combined.index.isin(miss_mb)
    obs_meta = ~combined.index.isin(miss_mt)

    mkc = MKC_embd_ht(lambda1=best_l1, lambda2=best_l2, max_iter=50)
    K_mi_hat, K_me_hat = mkc.fit_transform(K_mi, K_me, obs_micro, obs_meta)

    W_mi = minmax_normalize(K_mi_hat)
    W_me = minmax_normalize(K_me_hat)

    S_avg = (W_mi + W_me) / 2.0
    D_mkc_avg = 1.0 - S_avg
    np.fill_diagonal(D_mkc_avg, 0.0)
    
    D_mkc_avg_df = pd.DataFrame(D_mkc_avg, index=combined.index, columns=combined.index)
    pred_mkc_avg = D_mkc_avg_df.loc[masked_ids, masked_ids]
    r_mkc_avg, p_mkc_avg = mantel_test(gt_dist, pred_mkc_avg)
    results.append({"Pipeline": f"MKC_embd_ht (Avg)", "Mantel_r": r_mkc_avg, "P_Value": p_mkc_avg})

    print("--- Running Pipeline 4: MKC + SNF ---")
    P_fused_mkc = snf([W_mi, W_me], K=20, T=20, minmax=False)
    D_mkc_snf = fused_to_distance(P_fused_mkc)
    
    D_mkc_snf_df = pd.DataFrame(D_mkc_snf, index=combined.index, columns=combined.index)
    pred_mkc_snf = D_mkc_snf_df.loc[masked_ids, masked_ids]
    r_mkc_snf, p_mkc_snf = mantel_test(gt_dist, pred_mkc_snf)
    results.append({"Pipeline": f"MKC_embd_ht + SNF", "Mantel_r": r_mkc_snf, "P_Value": p_mkc_snf})

    # ---------------------------------------------------------
    # FINAL RESULTS
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("PIPELINE COMPARISON GRADES (Mantel r vs Ground Truth)")
    print("="*60)
    results_df = pd.DataFrame(results).sort_values("Mantel_r", ascending=False)
    print(results_df.to_string(index=False))