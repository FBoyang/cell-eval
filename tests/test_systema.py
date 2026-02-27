"""Minimal tests for residual-DE (systema) pipeline."""

import numpy as np
import pytest

from cell_eval import compute_systema_de
from cell_eval.data import CONTROL_VAR, PERT_COL, build_random_anndata
from cell_eval._types import initialize_de_comparison


def test_systema_de_output_schema():
    """compute_systema_de returns a DataFrame with standard DE schema."""
    adata = build_random_anndata(n_cells=200, n_genes=50, n_perts=3, random_state=42)
    df = compute_systema_de(
        adata,
        pert_col=PERT_COL,
        control_pert=CONTROL_VAR,
    )
    required = {"target", "feature", "fold_change", "log2_fold_change", "abs_log2_fold_change", "p_value", "fdr"}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"
    assert df.height > 0
    # n_perts=3 gives pert_0, pert_1, pert_2 (control excluded from DE output)
    assert df["target"].n_unique() == 3
    assert df["feature"].n_unique() == 50


def test_systema_de_p_value_fdr_bounds():
    """p_value and fdr are in [0, 1]."""
    adata = build_random_anndata(n_cells=150, n_genes=30, n_perts=2, random_state=123)
    df = compute_systema_de(adata, pert_col=PERT_COL, control_pert=CONTROL_VAR)
    assert df["p_value"].min() >= 0 and df["p_value"].max() <= 1
    assert df["fdr"].min() >= 0 and df["fdr"].max() <= 1


def test_systema_de_wraps_in_de_comparison():
    """Systema output can be wrapped in DEComparison for downstream metrics."""
    adata = build_random_anndata(n_cells=100, n_genes=20, n_perts=2, random_state=1)
    df_real = compute_systema_de(adata, pert_col=PERT_COL, control_pert=CONTROL_VAR)
    # Pred can be same or another adata; same frame is enough to check wrapper
    df_pred = compute_systema_de(adata, pert_col=PERT_COL, control_pert=CONTROL_VAR)
    comp = initialize_de_comparison(real=df_real, pred=df_pred)
    # build_random_anndata(n_perts=2) gives pert_0, pert_1 + control -> 2 perturbations in DE
    assert comp.n_perts == 2
    assert list(comp.iter_perturbations())
