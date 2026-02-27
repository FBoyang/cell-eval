import logging
import multiprocessing as mp
import os
from typing import Any, Literal

import anndata as ad
import pandas as pd
import polars as pl
import scanpy as sc
from pdex import parallel_differential_expression

from cell_eval.utils import guess_is_lognorm

from ._pipeline import MetricPipeline
from ._systema import compute_systema_de
from ._types import PerturbationAnndataPair, initialize_de_comparison

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Evaluates benchmarking metrics of a predicted and real anndata object.

    Arguments
    =========

    adata_pred: ad.AnnData | str
        Predicted anndata object or path to anndata object.
    adata_real: ad.AnnData | str
        Real anndata object or path to anndata object.
    de_pred: pl.DataFrame | str | None = None
        Predicted differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    de_real: pl.DataFrame | str | None = None
        Real differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    control_pert: str = "non-targeting"
        Control perturbation name.
    pert_col: str = "target"
        Perturbation column name.
    de_method: str = "wilcoxon"
        Differential expression method.
    num_threads: int = -1
        Number of threads for parallel differential expression.
    batch_size: int = 100
        Batch size for parallel differential expression.
    outdir: str = "./cell-eval-outdir"
        Output directory.
    allow_discrete: bool = False
        Allow discrete data.
    prefix: str | None = None
        Prefix for output files.
    pdex_kwargs: dict[str, Any] | None = None
        Keyword arguments for parallel_differential_expression.
        These will overwrite arguments passed to MetricsEvaluator.__init__ if they conflict.
    de_mode: str = "standard"
        DE computation mode: "standard" (pdex), "systema" (residual-DE), or "both".
    """

    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred: pl.DataFrame | str | None = None,
        de_real: pl.DataFrame | str | None = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        de_method: str = "wilcoxon",
        num_threads: int = -1,
        batch_size: int = 100,
        outdir: str = "./cell-eval-outdir",
        allow_discrete: bool = False,
        prefix: str | None = None,
        pdex_kwargs: dict[str, Any] | None = None,
        skip_de: bool = False,
        de_mode: str = "standard",
    ):
        # Enable a global string cache for categorical columns
        pl.enable_string_cache()

        if os.path.exists(outdir):
            logger.warning(
                f"Output directory {outdir} already exists, potential overwrite occurring"
            )
        os.makedirs(outdir, exist_ok=True)

        de_mode = de_mode.lower().strip()
        if de_mode not in ("standard", "systema", "both"):
            raise ValueError(f"de_mode must be 'standard', 'systema', or 'both', got {de_mode!r}")

        self.de_mode = de_mode
        self.anndata_pair = _build_anndata_pair(
            real=adata_real,
            pred=adata_pred,
            control_pert=control_pert,
            pert_col=pert_col,
            allow_discrete=allow_discrete,
        )
        self.anndata_pair_systema = None
        self.de_comparison = None
        self.de_comparison_systema = None

        if skip_de:
            pass
        elif de_mode in ("standard", "both"):
            self.de_comparison, self.anndata_pair = _build_de_comparison(
                anndata_pair=self.anndata_pair,
                de_pred=de_pred,
                de_real=de_real,
                de_method=de_method,
                num_threads=num_threads if num_threads != -1 else mp.cpu_count(),
                batch_size=batch_size,
                allow_discrete=allow_discrete,
                outdir=outdir,
                prefix=prefix,
                pdex_kwargs=pdex_kwargs or {},
            )
        if not skip_de and de_mode in ("systema", "both"):
            sys_prefix = f"{prefix}_systema" if prefix else "systema"
            self.de_comparison_systema, self.anndata_pair_systema = _build_systema_comparison(
                anndata_pair=self.anndata_pair,
                outdir=outdir,
                prefix=sys_prefix,
            )
            if de_mode == "systema":
                self.anndata_pair = self.anndata_pair_systema

        self.outdir = outdir
        self.prefix = prefix

    def compute(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame] | tuple[tuple[pl.DataFrame, pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]:
        if self.de_comparison_systema is None:
            return self._compute_single(
                profile=profile,
                metric_configs=metric_configs,
                skip_metrics=skip_metrics,
                basename=basename,
                write_csv=write_csv,
                break_on_error=break_on_error,
                de_comparison=self.de_comparison,
                anndata_pair=self.anndata_pair,
                suffix="_standard",
            )
        if self.de_comparison is None:
            # Systema-only mode
            return self._compute_single(
                profile=profile,
                metric_configs=metric_configs,
                skip_metrics=skip_metrics,
                basename=basename,
                write_csv=write_csv,
                break_on_error=break_on_error,
                de_comparison=self.de_comparison_systema,
                anndata_pair=self.anndata_pair_systema,
                suffix="_systema",
            )
        # Both modes: run standard and systema separately, each with its own AnnData metrics on the matching pair
        results_std, agg_std = self._compute_single(
            profile=profile,
            metric_configs=metric_configs,
            skip_metrics=skip_metrics,
            basename=basename,
            write_csv=write_csv,
            break_on_error=break_on_error,
            de_comparison=self.de_comparison,
            anndata_pair=self.anndata_pair,
            suffix="_standard",
        )
        results_sys, agg_sys = self._compute_single(
            profile=profile,
            metric_configs=metric_configs,
            skip_metrics=skip_metrics,
            basename=basename,
            write_csv=write_csv,
            break_on_error=break_on_error,
            de_comparison=self.de_comparison_systema,
            anndata_pair=self.anndata_pair_systema,
            suffix="_systema",
        )
        return (results_std, agg_std), (results_sys, agg_sys)

    def _compute_single(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"],
        metric_configs: dict[str, dict[str, Any]] | None,
        skip_metrics: list[str] | None,
        basename: str,
        write_csv: bool,
        break_on_error: bool,
        de_comparison: Any,
        anndata_pair: PerturbationAnndataPair | None,
        suffix: str,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
            break_on_error=break_on_error,
        )
        if skip_metrics is not None:
            pipeline.skip_metrics(skip_metrics)
        pipeline.compute_de_metrics(de_comparison)
        pipeline.compute_anndata_metrics(anndata_pair)
        results = pipeline.get_results()
        agg_results = pipeline.get_agg_results()

        if write_csv and basename:
            _prefix = ((self.prefix or "") + suffix).strip()
            if _prefix:
                _prefix = _prefix.replace("/", "-")
            _basename = basename.replace("/", "-")
            outpath = os.path.join(
                self.outdir,
                f"{_prefix}_{_basename}" if _prefix else _basename,
            )
            agg_outpath = os.path.join(
                self.outdir,
                f"{_prefix}_agg_{_basename}" if _prefix else f"agg_{_basename}",
            )
            logger.info(f"Writing perturbation level metrics to {outpath}")
            results.write_csv(outpath)
            logger.info(f"Writing aggregate metrics to {agg_outpath}")
            agg_results.write_csv(agg_outpath)

        return results, agg_results


def _build_anndata_pair(
    real: ad.AnnData | str,
    pred: ad.AnnData | str,
    control_pert: str,
    pert_col: str,
    allow_discrete: bool = False,
):
    if isinstance(real, str):
        logger.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        logger.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)

    # Validate that the input is normalized and log-transformed
    _convert_to_normlog(real, which="real", allow_discrete=allow_discrete)
    _convert_to_normlog(pred, which="pred", allow_discrete=allow_discrete)

    # Build the anndata pair
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


def _convert_to_normlog(
    adata: ad.AnnData,
    which: str | None = None,
    allow_discrete: bool = False,
):
    """Performs a norm-log conversion if the input is integer data (inplace).

    Will skip if the input is not integer data.
    """
    validate_scale = (which == "real") and (not allow_discrete)
    min_tolerance = 1e-6 if which == "real" else 0.0
    if guess_is_lognorm(
        adata=adata,
        validate=validate_scale,
        min_tolerance=min_tolerance,
    ):
        logger.info(
            "Input is found to be log-normalized already - skipping transformation."
        )
        return  # Input is already log-normalized

    # User specified that they want to allow discrete data
    if allow_discrete:
        if which:
            logger.info(
                f"Discovered integer data for {which}. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        else:
            logger.info(
                "Discovered integer data. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        return  # proceed without conversion

    # Convert the data to norm-log
    if which:
        logger.info(f"Discovered integer data for {which}. Converting to norm-log.")
    sc.pp.normalize_total(adata=adata, inplace=True)  # normalize to median
    sc.pp.log1p(adata)  # log-transform (log1p)


def _build_de_comparison(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: pl.DataFrame | str | None = None,
    de_real: pl.DataFrame | str | None = None,
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    allow_discrete: bool = False,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] | None = None,
):
    if anndata_pair is None:
        raise ValueError("anndata_pair must be provided")

    real_frame = _load_or_build_de(
        mode="real",
        de_path=de_real,
        anndata_pair=anndata_pair,
        de_method=de_method,
        num_threads=num_threads,
        batch_size=batch_size,
        allow_discrete=allow_discrete,
        outdir=outdir,
        prefix=prefix,
        pdex_kwargs=pdex_kwargs or {},
    )
    pred_frame = _load_or_build_de(
        mode="pred",
        de_path=de_pred,
        anndata_pair=anndata_pair,
        de_method=de_method,
        num_threads=num_threads,
        batch_size=batch_size,
        allow_discrete=allow_discrete,
        outdir=outdir,
        prefix=prefix,
        pdex_kwargs=pdex_kwargs or {},
    )

    # Drop perturbations where real has zero significant genes (FDR < 0.05), and log them.
    fdr_threshold = 0.05
    sig_counts = (
        real_frame.group_by("target")
        .agg((pl.col("fdr") < fdr_threshold).sum().alias("n_sig"))
    )
    no_de = sig_counts.filter(pl.col("n_sig") == 0).select("target").to_series().to_list()
    if len(no_de) > 0:
        if outdir is not None:
            _prefix = prefix.replace("/", "-") if prefix else None
            log_name = (
                f"{_prefix}_real_no_de_perturbations.csv"
                if _prefix
                else "real_no_de_perturbations.csv"
            )
            sig_counts.filter(pl.col("target").is_in(no_de)).write_csv(
                os.path.join(outdir, log_name)
            )

        # Filter DE frames
        real_frame = real_frame.filter(~pl.col("target").is_in(no_de))
        pred_frame = pred_frame.filter(~pl.col("target").is_in(no_de))

        # Filter AnnData and rebuild PerturbationAnndataPair to keep masks/caches consistent
        pert_col = anndata_pair.pert_col
        ctrl = anndata_pair.control_pert
        mask_real = (~anndata_pair.real.obs[pert_col].isin(no_de)) | (anndata_pair.real.obs[pert_col] == ctrl)
        mask_pred = (~anndata_pair.pred.obs[pert_col].isin(no_de)) | (anndata_pair.pred.obs[pert_col] == ctrl)
        anndata_pair = PerturbationAnndataPair(
            real=anndata_pair.real[mask_real].copy(),
            pred=anndata_pair.pred[mask_pred].copy(),
            pert_col=pert_col,
            control_pert=ctrl,
            embed_key=anndata_pair.embed_key,
        )

    return initialize_de_comparison(real=real_frame, pred=pred_frame), anndata_pair


def _build_systema_comparison(
    anndata_pair: PerturbationAnndataPair,
    outdir: str | None = None,
    prefix: str | None = None,
):
    """Build DEComparison from residual-DE (systema) on real and pred; filter zero-DE perturbations."""
    real_frame = compute_systema_de(
        anndata_pair.real,
        pert_col=anndata_pair.pert_col,
        control_pert=anndata_pair.control_pert,
    )
    pred_frame = compute_systema_de(
        anndata_pair.pred,
        pert_col=anndata_pair.pert_col,
        control_pert=anndata_pair.control_pert,
    )
    fdr_threshold = 0.05
    sig_counts = (
        real_frame.group_by("target")
        .agg((pl.col("fdr") < fdr_threshold).sum().alias("n_sig"))
    )
    no_de = sig_counts.filter(pl.col("n_sig") == 0).select("target").to_series().to_list()
    if len(no_de) > 0:
        if outdir is not None and prefix is not None:
            _prefix = prefix.replace("/", "-")
            log_name = f"{_prefix}_real_no_de_perturbations.csv"
            sig_counts.filter(pl.col("target").is_in(no_de)).write_csv(
                os.path.join(outdir, log_name)
            )
        real_frame = real_frame.filter(~pl.col("target").is_in(no_de))
        pred_frame = pred_frame.filter(~pl.col("target").is_in(no_de))
        pert_col = anndata_pair.pert_col
        ctrl = anndata_pair.control_pert
        mask_real = (~anndata_pair.real.obs[pert_col].isin(no_de)) | (anndata_pair.real.obs[pert_col] == ctrl)
        mask_pred = (~anndata_pair.pred.obs[pert_col].isin(no_de)) | (anndata_pair.pred.obs[pert_col] == ctrl)
        anndata_pair = PerturbationAnndataPair(
            real=anndata_pair.real[mask_real].copy(),
            pred=anndata_pair.pred[mask_pred].copy(),
            pert_col=pert_col,
            control_pert=ctrl,
            embed_key=anndata_pair.embed_key,
        )
    return initialize_de_comparison(real=real_frame, pred=pred_frame), anndata_pair


def _build_pdex_kwargs(
    reference: str,
    groupby_key: str,
    num_workers: int,
    batch_size: int,
    metric: str,
    allow_discrete: bool,
    pdex_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pdex_kwargs = pdex_kwargs or {}
    if "reference" not in pdex_kwargs:
        pdex_kwargs["reference"] = reference
    if "groupby_key" not in pdex_kwargs:
        pdex_kwargs["groupby_key"] = groupby_key
    if "num_workers" not in pdex_kwargs:
        pdex_kwargs["num_workers"] = num_workers
    if "batch_size" not in pdex_kwargs:
        pdex_kwargs["batch_size"] = batch_size
    if "metric" not in pdex_kwargs:
        pdex_kwargs["metric"] = metric
    if "is_log1p" not in pdex_kwargs:
        if allow_discrete:
            pdex_kwargs["is_log1p"] = False
        else:
            pdex_kwargs["is_log1p"] = True

    # always return polars DataFrames
    pdex_kwargs["as_polars"] = True
    return pdex_kwargs


def _load_or_build_de(
    mode: Literal["pred", "real"],
    de_path: pl.DataFrame | str | None = None,
    anndata_pair: PerturbationAnndataPair | None = None,
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    outdir: str | None = None,
    prefix: str | None = None,
    allow_discrete: bool = False,
    pdex_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    if de_path is None:
        if anndata_pair is None:
            raise ValueError("anndata_pair must be provided if de_path is not provided")
        logger.info(f"Computing DE for {mode} data")
        pdex_kwargs = _build_pdex_kwargs(
            reference=anndata_pair.control_pert,
            groupby_key=anndata_pair.pert_col,
            num_workers=num_threads,
            metric=de_method,
            batch_size=batch_size,
            allow_discrete=allow_discrete,
            pdex_kwargs=pdex_kwargs or {},
        )
        logger.info(f"Using the following pdex kwargs: {pdex_kwargs}")
        frame = parallel_differential_expression(
            adata=anndata_pair.real if mode == "real" else anndata_pair.pred,
            **pdex_kwargs,
        )
        # Sanitize p_value for FDR: log invalid rows to outdir (summary-stats dir), then either drop (real) or treat as non-DE (pred)
        valid_p = (pl.col("p_value") >= 0) & (pl.col("p_value") <= 1)
        n_invalid = frame.filter(~valid_p).height
        if n_invalid > 0:
            invalid_rows = frame.filter(~valid_p)
            if outdir is not None:
                _prefix = prefix.replace("/", "-") if prefix else None
                log_name = f"{_prefix}_{mode}_de_invalid_pvalues.csv" if _prefix else f"{mode}_de_invalid_pvalues.csv"
                log_path = os.path.join(outdir, log_name)
                invalid_rows.write_csv(log_path)
                logger.info(f"Logged {n_invalid} rows with invalid p_value to {log_path}")
            if mode == "real":
                frame = frame.filter(valid_p)
            else:
                # pred: keep rows but set invalid p_value to 1.0 so perturbation is treated as non-DE
                frame = frame.with_columns(
                    pl.when(valid_p).then(pl.col("p_value")).otherwise(1.0).alias("p_value")
                )
        if outdir is not None:
            if prefix is not None:
                prefix = prefix.replace(
                    "/", "-"
                )  # some prefixes (e.g. HepG2/C3A) may have slashes in them
            pathname = f"{mode}_de.csv" if not prefix else f"{prefix}_{mode}_de.csv"
            logger.info(f"Writing {mode} DE results to: {pathname}")
            frame.write_csv(os.path.join(outdir, pathname))

        return frame  # type: ignore
    elif isinstance(de_path, str):
        logger.info(f"Reading {mode} DE results from {de_path}")
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.read_csv(
            de_path,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )
    elif isinstance(de_path, pl.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return de_path
    elif isinstance(de_path, pd.DataFrame):
        if pdex_kwargs:
            logger.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.from_pandas(de_path)
    else:
        raise TypeError(f"Unexpected type for de_path: {type(de_path)}")
