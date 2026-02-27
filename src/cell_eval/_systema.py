"""
Residual-DE ("systema") pipeline for single-cell perturbation data.

For each perturbation P and gene g, computes residual effect:
  r_P(g) = Δ_P(g) − μ_{−P}(g)
where Δ_P(g) = mean(P,g) − mean(Control,g) and μ_{−P}(g) is the mean of Δ_Q(g) over Q≠P.
Tests H0: r_P(g)=0 (two-sided), outputs p-values and BH-FDR per perturbation.

Output schema matches standard DE (target, feature, fold_change, log2_fold_change,
abs_log2_fold_change, p_value, fdr) for compatibility with DEComparison.
"""

from __future__ import annotations

import logging
from typing import Optional

import anndata as ad
import numpy as np
import polars as pl
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

# Minimum variance floor for numerical stability
DEFAULT_EPS = 1e-10


def _get_dense_matrix(adata: ad.AnnData) -> np.ndarray:
    """Return adata.X as dense (n_obs x n_vars)."""
    X = adata.X
    if issparse(X):
        return X.toarray()
    return np.asarray(X)


def _group_stats(
    X: np.ndarray,
    group_indices: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-group mean, variance (s^2), and count for each gene.
    group_indices: int array of shape (n_obs,) with values in [0, n_groups-1].
    Returns (means, vars, counts); counts shape (n_groups,), means/vars shape (n_groups, n_genes).
    """
    n_genes = X.shape[1]
    means = np.zeros((n_groups, n_genes), dtype=np.float64)
    vars_ = np.zeros((n_groups, n_genes), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.float64)

    for i in range(n_groups):
        mask = group_indices == i
        n = np.sum(mask)
        counts[i] = n
        if n == 0:
            continue
        group_X = X[mask]
        means[i] = np.mean(group_X, axis=0)
        if n > 1:
            vars_[i] = np.var(group_X, axis=0, ddof=1)
        # else vars_[i] stays 0

    return means, vars_, counts


def compute_systema_de(
    adata: ad.AnnData,
    pert_col: str,
    control_pert: str = "non-targeting",
    stratify_col: Optional[str] = None,
    eps: float = DEFAULT_EPS,
    fdr_threshold: float = 0.05,
) -> pl.DataFrame:
    """
    Compute residual-DE (systema) for each perturbation.

    Residual r_P(g) = Δ_P(g) − μ_{−P}(g). Variance uses analytic formula with
    covariance correction for shared control. Output has standard DE schema.

    Args:
        adata: AnnData with .X (expression) and obs[pert_col].
        pert_col: Column in adata.obs with perturbation labels.
        control_pert: Control perturbation name.
        stratify_col: If set, run pipeline separately per stratum (e.g. cell_line).
        eps: Variance floor for numerical stability.
        fdr_threshold: Unused here; FDR is computed and stored; filtering is caller's responsibility.

    Returns:
        Polars DataFrame with columns: target, feature, fold_change, log2_fold_change,
        abs_log2_fold_change, p_value, fdr.
    """
    if stratify_col is not None and stratify_col in adata.obs.columns:
        strata = adata.obs[stratify_col].astype(str).unique().tolist()
        parts = []
        for s in strata:
            adata_s = adata[adata.obs[stratify_col].astype(str) == s]
            parts.append(
                compute_systema_de(
                    adata_s,
                    pert_col=pert_col,
                    control_pert=control_pert,
                    stratify_col=None,
                    eps=eps,
                    fdr_threshold=fdr_threshold,
                )
            )
        return pl.concat(parts, how="vertical_relaxed")

    X = _get_dense_matrix(adata)
    n_obs, n_genes = X.shape
    genes = adata.var_names.tolist()
    labels = np.array(adata.obs[pert_col].astype(str).tolist())

    unique_perts = sorted(set(labels))
    if control_pert not in unique_perts:
        raise ValueError(f"Control '{control_pert}' not found in {pert_col}")
    pert_ids = [p for p in unique_perts if p != control_pert]
    K = len(pert_ids)
    if K == 0:
        return pl.DataFrame(
            schema={
                "target": pl.Utf8,
                "feature": pl.Utf8,
                "fold_change": pl.Float32,
                "log2_fold_change": pl.Float32,
                "abs_log2_fold_change": pl.Float32,
                "p_value": pl.Float32,
                "fdr": pl.Float32,
            }
        )

    # Group indices: 0..K-1 for pert_ids, K for control
    id_to_idx = {p: i for i, p in enumerate(pert_ids)}
    id_to_idx[control_pert] = K
    n_groups = K + 1
    group_indices = np.array([id_to_idx[l] for l in labels])

    # (n_groups, n_genes): means, variances, and counts (per group)
    means, vars_, counts = _group_stats(X, group_indices, n_groups)
    # control is last
    mean_C = means[K]
    var_C = vars_[K]
    n_C = counts[K]
    if n_C <= 0:
        raise ValueError("No control cells found")

    # Delta_P(g) = mean_P(g) - mean_C(g), shape (K, n_genes)
    deltas = means[:K] - mean_C
    # v_P(g) = s_P^2/n_P + s_C^2/n_C for Var(Delta_P)
    v_control = np.maximum(var_C / n_C, eps)
    v_delta = np.zeros_like(deltas)
    for i in range(K):
        n_P = max(counts[i], 1)
        v_P = np.maximum(vars_[i] / n_P, 0.0)
        v_delta[i] = v_P + v_control

    # Precompute sums over perturbations (excluding control)
    sum_delta = np.sum(deltas, axis=0)
    sum_v = np.sum(v_delta, axis=0)

    K_minus_1 = max(K - 1, 1)
    # Per-perturbation: r_P, Var(r_P), then t/z and BH
    rows = []
    for i, P in enumerate(pert_ids):
        mu_negP = (sum_delta - deltas[i]) / K_minus_1
        r_P = deltas[i] - mu_negP
        var_mu_negP = (sum_v - v_delta[i]) / (K_minus_1**2) + v_control / K_minus_1
        cov_term = 2 * v_control / (n_C * K_minus_1)
        var_r = np.maximum(v_delta[i] + var_mu_negP - cov_term, eps)
        z = r_P / np.sqrt(var_r)
        # Two-sided normal p-values
        p_values = 2 * (1 - _norm_cdf(np.abs(z)))
        p_values = np.clip(p_values, 0.0, 1.0)
        # BH FDR per perturbation
        q_values = _bh_fdr(p_values)
        # fold_change in ratio form: 2^r for log2-scale r
        log2_fc = r_P.astype(np.float32)
        fold_change = np.power(2.0, np.clip(log2_fc, -20, 20)).astype(np.float32)
        abs_log2_fc = np.abs(log2_fc).astype(np.float32)
        for g in range(n_genes):
            rows.append({
                "target": P,
                "feature": genes[g],
                "fold_change": float(fold_change[g]),
                "log2_fold_change": float(log2_fc[g]),
                "abs_log2_fold_change": float(abs_log2_fc[g]),
                "p_value": float(p_values[g]),
                "fdr": float(q_values[g]),
            })

    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.col("target").cast(pl.Categorical),
        pl.col("feature").cast(pl.Categorical),
        pl.col("fold_change").cast(pl.Float32),
        pl.col("log2_fold_change").cast(pl.Float32),
        pl.col("abs_log2_fold_change").cast(pl.Float32),
        pl.col("p_value").cast(pl.Float32),
        pl.col("fdr").cast(pl.Float32),
    )


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF."""
    from scipy.stats import norm
    return norm.cdf(x)


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(p)
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    p_sorted = p[order]
    q_sorted = np.minimum(1.0, p_sorted * n / np.arange(1, n + 1, dtype=np.float64))
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = q_sorted
    return q
