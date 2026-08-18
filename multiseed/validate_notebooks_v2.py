#!/usr/bin/env python3
"""Independent, read-only validation of the Claude multiseed BAMIC outputs.

The script reads the executed notebooks in the local handoff folder and the raw
CSV/NumPy artifacts in the Google Drive for desktop mirror.  It does not retrain
the model or alter the Drive outputs.  Machine-readable findings are written to
the workspace output directory for auditability.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)


LOCAL = Path("/Users/liyuan/Desktop/disser/AMIC/BAMIC_code/dr_cao_multiseed")
OFFICIAL = Path(
    "/Users/liyuan/Desktop/disser/AMIC/BAMIC_code/BAMIC_codex_updated/"
    "wine_benchmark/01d_bamic_wine_gatefix_bestval.ipynb"
)
DRIVE_OUTPUTS = Path(
    "/Users/liyuan/Library/CloudStorage/"
    "GoogleDrive-593697882qq@gmail.com/My Drive/AMIC project/"
    "wine_benchmark/outputs"
)
AGG = DRIVE_OUTPUTS / "multiseed_results_ms1_v2"
OUT = Path(
    "/Users/liyuan/Desktop/disser/AMIC/BAMIC_code/dr_cao_multiseed/"
    "validation_v2"
)
SEEDS = list(range(20260526, 20260536))
RUN_RE = re.compile(r"seed(\d+)_ms1$")
PRIOR_RE = re.compile(r"seed20260526_prior(0p5|1p0|2p0)$")
DATA = DRIVE_OUTPUTS.parent / "data"


def ffloat(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def json_clean(value):
    """Convert NumPy scalars and non-finite floats to strict JSON values."""
    value = ffloat(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    return value


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def notebook_audit(path: Path) -> dict:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code]
    errors = [
        output
        for cell in code
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    executed = [count for count in counts if count is not None]
    return {
        "file": path.name,
        "sha256": sha256(path),
        "code_cells": len(code),
        "executed_code_cells": len(executed),
        "unexecuted_code_cells": sum(count is None for count in counts),
        "error_outputs": len(errors),
        "execution_counts_strictly_increasing": all(
            b > a for a, b in zip(executed, executed[1:])
        ),
    }


def expected_calibration_error(probs, labels, n_bins=10) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & ((probs <= hi) if hi == 1.0 else (probs < hi))
        if mask.any():
            result += abs(probs[mask].mean() - labels[mask].mean()) * mask.mean()
    return float(result)


def metrics(probs, labels, threshold=0.5) -> dict:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pred = (probs >= threshold).astype(int)
    return {
        "acc": float(accuracy_score(labels, pred)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "auc": float(roc_auc_score(labels, probs)),
        "brier": float(brier_score_loss(labels, probs)),
        "nll": float(log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "ece": expected_calibration_error(probs, labels),
    }


def routing(docs: pd.DataFrame, tau: float) -> dict:
    lo = docs["ci95_lo"].to_numpy(float)
    hi = docs["ci95_hi"].to_numpy(float)
    p = docs["p_mean"].to_numpy(float)
    y = docs["y_true"].to_numpy(int)
    auto = (lo > tau) | (hi < tau)
    pred = (p >= tau).astype(int)
    err = pred != y
    return {
        "auto_coverage": float(auto.mean()),
        "acc_auto_classified": float((pred[auto] == y[auto]).mean()),
        "error_capture_deferred": float(err[~auto].sum() / max(err.sum(), 1)),
    }


def decompose(means: np.ndarray, variances: np.ndarray, between_ddof: int) -> pd.DataFrame:
    within = variances.mean(axis=1)
    between = means.var(axis=1, ddof=between_ddof)
    total = within + between
    share = np.divide(between, total, out=np.zeros_like(total), where=total > 0)
    ratio = np.divide(
        between,
        within,
        out=np.full_like(between, np.nan),
        where=within > 0,
    )
    return pd.DataFrame(
        {
            "within": within,
            "between": between,
            "total": total,
            "between_share": share,
            "between_within_ratio": ratio,
            "between_exceeds_within": between > within,
        }
    )


def summarize_decomp(frame: pd.DataFrame) -> dict:
    mw = float(frame["within"].mean())
    mb = float(frame["between"].mean())
    return {
        "mean_within": mw,
        "mean_between": mb,
        "ratio_of_means": mb / mw,
        "between_share_of_means": mb / float(frame["total"].mean()),
        "median_between_share": float(frame["between_share"].median()),
        "pct_between_exceeds_within": float(
            100 * frame["between_exceeds_within"].mean()
        ),
    }


def assert_close(a, b, *, atol=1e-10, rtol=1e-8, label="value"):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if not np.allclose(aa, bb, atol=atol, rtol=rtol, equal_nan=True):
        delta = float(np.nanmax(np.abs(aa - bb)))
        raise AssertionError(f"{label} differs (maximum absolute difference {delta})")


def locate_seed_folders() -> list[tuple[int, Path]]:
    found = []
    for folder in DRIVE_OUTPUTS.glob("wine_bamic_*seed*_ms1"):
        match = RUN_RE.search(folder.name)
        if match:
            found.append((int(match.group(1)), folder))
    found.sort()
    if [seed for seed, _ in found] != SEEDS:
        raise AssertionError(f"unexpected multiseed folders: {[s for s, _ in found]}")
    return found


def read_run(seed: int, folder: Path) -> dict:
    required = {
        "final_metrics.csv",
        "test_document_uncertainty.csv",
        "test_document_uncertainty_addon.csv",
        "test_prob_samples.npy",
        "panel_word_uncertainty.csv",
        "param_posterior_sd.csv",
        "run_completion.csv",
        "checkpoint_summary.csv",
        "restored_validation_gate_health.csv",
        "run_config.csv",
        "matched_seed_plan.csv",
        "threshold_tuning.csv",
    }
    missing = sorted(name for name in required if not (folder / name).exists())
    if missing:
        raise AssertionError(f"seed {seed} missing files: {missing}")

    config = pd.read_csv(folder / "run_config.csv").iloc[0]
    completion = pd.read_csv(folder / "run_completion.csv").iloc[0]
    checkpoint = pd.read_csv(folder / "checkpoint_summary.csv").iloc[0]
    health = pd.read_csv(folder / "restored_validation_gate_health.csv").iloc[0]
    plan = pd.read_csv(folder / "matched_seed_plan.csv")
    docs = pd.read_csv(folder / "test_document_uncertainty.csv")
    addon = pd.read_csv(folder / "test_document_uncertainty_addon.csv")
    draws = np.load(folder / "test_prob_samples.npy")
    panel = pd.read_csv(folder / "panel_word_uncertainty.csv")
    posterior = pd.read_csv(folder / "param_posterior_sd.csv")
    final = pd.read_csv(folder / "final_metrics.csv")

    if int(config["run_seed"]) != seed or config["run_label"] != "ms1":
        raise AssertionError(f"seed {seed}: run config identity mismatch")
    expected_config = {
        "freeze_embeddings": True,
        "epochs": 15,
        "num_heads": 1,
        "lambda_boilerplate": 0.1,
        "lambda_stopword": 0.01,
        "learning_rate": 0.0003,
        "max_len": 100,
        "batch_size": 128,
        "use_positional_encoding": True,
        "aggregation_mode": "selected_token_mean",
        "lambda_p1": 0.01,
        "lambda_p2_effective": 0.0,
        "lambda_p3": 0.001,
        "use_selection_rate_band": True,
        "selection_rate_band_low": 0.075,
        "selection_rate_band_high": 0.125,
        "lambda_selection_rate_band": 1.0,
        "use_gate_valid_checkpoint": True,
        "checkpoint_metric": "nll",
        "checkpoint_mode": "min",
        "evaluation_scope": "exploratory_reuse_of_previously_inspected_project_test",
        "project_test_previously_inspected": True,
    }
    for field, expected in expected_config.items():
        actual = config[field]
        if isinstance(expected, float):
            assert_close(actual, expected, label=f"seed {seed} config {field}")
        elif actual != expected:
            raise AssertionError(
                f"seed {seed}: config {field}={actual!r}, expected {expected!r}"
            )
    if not bool(completion["run_complete"]) or bool(completion["test_used_for_selection"]):
        raise AssertionError(f"seed {seed}: completion/selection audit failed")
    if not bool(checkpoint["validation_only_selection"]) or bool(
        checkpoint["test_used_for_selection"]
    ):
        raise AssertionError(f"seed {seed}: checkpoint is not validation-only")
    if checkpoint["status"] != "restored_gate_valid_checkpoint":
        raise AssertionError(f"seed {seed}: wrong checkpoint status")
    if not bool(checkpoint["gate_valid_selection"]) or not bool(
        health["gate_checkpoint_eligible"]
    ):
        raise AssertionError(f"seed {seed}: restored checkpoint is gate-invalid")
    plan_flags = [column for column in plan.columns if column != "seed"]
    if (
        list(plan["seed"].astype(int)) != SEEDS
        or not plan_flags
        or not plan[plan_flags].all(axis=None)
    ):
        raise AssertionError(f"seed {seed}: matched seed plan failed")

    if docs.shape[0] != 10000 or addon.shape[0] != 10000:
        raise AssertionError(f"seed {seed}: expected 10,000 test documents")
    if draws.shape != (100, 10000):
        raise AssertionError(f"seed {seed}: unexpected draw shape {draws.shape}")
    if not np.isfinite(draws).all() or draws.min() < 0 or draws.max() > 1:
        raise AssertionError(f"seed {seed}: invalid posterior probabilities")
    if panel.shape[0] != 6579 or panel[["doc_index", "position"]].duplicated().any():
        raise AssertionError(f"seed {seed}: malformed 200-document token panel")
    if posterior["n_params"].sum() != 602 or set(posterior["module"]) != {
        "b_layer",
        "beta_layer",
    }:
        raise AssertionError(f"seed {seed}: unexpected Bayesian parameter inventory")

    # The notebook computed these summaries in float32 before CSV export.
    assert_close(addon["p_mean"], draws.mean(axis=0), atol=1e-6, label="draw p_mean")
    assert_close(addon["var"], draws.var(axis=0), atol=2e-9, label="draw variance")
    assert_close(
        addon["ci95_lo"],
        np.quantile(draws, 0.025, axis=0),
        atol=1e-6,
        label="draw lower quantile",
    )
    assert_close(
        addon["ci95_hi"],
        np.quantile(draws, 0.975, axis=0),
        atol=1e-6,
        label="draw upper quantile",
    )
    assert_close(addon["y_true"], docs["y_true"], label="addon labels")

    test = final[final["split"] == "test"].iloc[0]
    recomputed = metrics(docs["p_mean"], docs["y_true"])
    for key, value in recomputed.items():
        assert_close(value, float(test[key]), atol=2e-12, label=f"seed {seed} {key}")

    threshold_rows = pd.read_csv(folder / "threshold_tuning.csv")
    tau = float(config["best_acc_threshold"])
    tuned = threshold_rows[threshold_rows["rule"] == "tuned by valid acc"].iloc[0]
    assert_close(tau, float(tuned["threshold"]), label=f"seed {seed} tuned tau")

    return {
        "seed": seed,
        "folder": folder,
        "config": config,
        "docs": docs,
        "draws": draws,
        "panel": panel,
        "posterior": posterior,
        "final": test,
    }


def validate_prior() -> tuple[pd.DataFrame, dict]:
    folders = []
    for folder in DRIVE_OUTPUTS.glob("wine_bamic_*seed20260526_prior*"):
        match = PRIOR_RE.search(folder.name)
        if match:
            prior = float(match.group(1).replace("p", "."))
            folders.append((prior, folder))
    folders.sort()
    if [value for value, _ in folders] != [0.5, 1.0, 2.0]:
        raise AssertionError(f"unexpected prior folders: {[x for x, _ in folders]}")

    rows = []
    for prior, folder in folders:
        required = [
            "run_completion.csv",
            "checkpoint_summary.csv",
            "restored_validation_gate_health.csv",
            "run_config.csv",
            "final_metrics.csv",
            "test_document_uncertainty.csv",
            "panel_word_uncertainty.csv",
            "param_posterior_sd.csv",
        ]
        missing = [name for name in required if not (folder / name).exists()]
        if missing:
            raise AssertionError(f"prior {prior} missing files: {missing}")
        completion = pd.read_csv(folder / "run_completion.csv").iloc[0]
        checkpoint = pd.read_csv(folder / "checkpoint_summary.csv").iloc[0]
        health = pd.read_csv(folder / "restored_validation_gate_health.csv").iloc[0]
        config = pd.read_csv(folder / "run_config.csv").iloc[0]
        final = pd.read_csv(folder / "final_metrics.csv")
        test = final[final["split"] == "test"].iloc[0]
        docs = pd.read_csv(folder / "test_document_uncertainty.csv")
        panel = pd.read_csv(folder / "panel_word_uncertainty.csv")
        posterior = pd.read_csv(folder / "param_posterior_sd.csv")
        if not bool(completion["run_complete"]) or bool(completion["test_used_for_selection"]):
            raise AssertionError(f"prior {prior}: completion audit failed")
        if not bool(checkpoint["validation_only_selection"]) or not bool(
            health["gate_checkpoint_eligible"]
        ):
            raise AssertionError(f"prior {prior}: checkpoint audit failed")
        assert_close(posterior["prior_sigma"], prior, label=f"prior {prior} sigma")
        mean_param_sd = float(
            np.average(posterior["mean_posterior_sd"], weights=posterior["n_params"])
        )
        tau = float(config["best_acc_threshold"])
        rows.append(
            {
                "prior_sd": prior,
                "mean_param_posterior_sd": mean_param_sd,
                "median_doc_ci_width": float(docs["ci_width"].median()),
                "median_token_ci_width": float(
                    (panel["z_ci95_hi"] - panel["z_ci95_lo"]).median()
                ),
                **{key: float(test[key]) for key in ["acc", "auc", "nll", "ece"]},
                "routing_coverage_tuned_tau": routing(docs, tau)["auto_coverage"],
                "best_epoch": int(test["best_epoch"]),
            }
        )

    recomputed = pd.DataFrame(rows)
    saved = pd.read_csv(DRIVE_OUTPUTS / "prior_sensitivity_results/prior_sensitivity_summary.csv")
    assert_close(
        recomputed[saved.columns],
        saved,
        atol=2e-12,
        label="prior sensitivity summary",
    )
    relative_ranges = {}
    for column in [
        "mean_param_posterior_sd",
        "median_doc_ci_width",
        "median_token_ci_width",
        "acc",
        "auc",
        "nll",
        "ece",
    ]:
        values = recomputed[column].to_numpy(float)
        relative_ranges[column] = float((values.max() - values.min()) / values.mean())
    return recomputed, relative_ranges


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    notebook_files = [
        LOCAL / "01d_bamic_wine_gatefix_bestval.ipynb",
        LOCAL / "02_bamic_wine_multiseed_driver.ipynb",
        LOCAL / "03_bamic_wine_multiseed_analysis.ipynb",
        LOCAL / "04_bamic_wine_prior_sensitivity.ipynb",
    ]
    notebooks = [notebook_audit(path) for path in notebook_files]
    if any(item["unexecuted_code_cells"] or item["error_outputs"] for item in notebooks):
        raise AssertionError("one or more notebooks were not cleanly executed")
    official_copy_identical = sha256(notebook_files[0]) == sha256(OFFICIAL)
    if not official_copy_identical:
        raise AssertionError("local 01d notebook is not identical to the official pipeline copy")

    runs = [read_run(seed, folder) for seed, folder in locate_seed_folders()]
    split_inventory = {}
    for split in ["train", "valid", "test"]:
        frame = pd.read_csv(DATA / f"wine_{split}.csv", usecols=["y"])
        split_inventory[split] = {
            "documents": int(len(frame)),
            "positive_rate": float(frame["y"].mean()),
        }
    y_ref = runs[0]["docs"]["y_true"].to_numpy(int)
    panel_ref = runs[0]["panel"][["doc_index", "position", "word", "y_true"]]
    for run in runs[1:]:
        if not np.array_equal(y_ref, run["docs"]["y_true"].to_numpy(int)):
            raise AssertionError(f"seed {run['seed']}: test labels/order differ")
        if not panel_ref.equals(
            run["panel"][["doc_index", "position", "word", "y_true"]]
        ):
            raise AssertionError(f"seed {run['seed']}: token panel alignment differs")

    seed_rows = []
    routing_rows = []
    for run in runs:
        test = run["final"]
        seed_rows.append(
            {
                "seed": run["seed"],
                **{key: float(test[key]) for key in ["acc", "f1", "auc", "brier", "nll", "ece"]},
                "best_epoch": int(test["best_epoch"]),
                "tuned_tau_acc": float(run["config"]["best_acc_threshold"]),
                "mean_gate_rate": float(run["docs"]["selection_rate_mean"].mean()),
                "mean_ci_width": float(run["docs"]["ci_width"].mean()),
                "median_ci_width": float(run["docs"]["ci_width"].median()),
            }
        )
        for rule, tau in [
            ("fixed_0.5", 0.5),
            ("tuned_acc", float(run["config"]["best_acc_threshold"])),
        ]:
            routing_rows.append({"seed": run["seed"], "rule": rule, "tau": tau, **routing(run["docs"], tau)})

    seed_table = pd.DataFrame(seed_rows)
    saved_seed = pd.read_csv(AGG / "seed_level_metrics.csv")
    assert_close(seed_table[saved_seed.columns], saved_seed, atol=2e-12, label="seed metrics")
    metric_summary = pd.DataFrame(
        [
            {
                "metric": column,
                "mean": seed_table[column].mean(),
                "sd": seed_table[column].std(ddof=1),
                "min": seed_table[column].min(),
                "max": seed_table[column].max(),
            }
            for column in seed_table.columns
            if column != "seed"
        ]
    )
    saved_metrics = pd.read_csv(AGG / "metrics_summary.csv")
    assert_close(
        metric_summary[["mean", "sd", "min", "max"]],
        saved_metrics[["mean", "sd", "min", "max"]],
        atol=2e-12,
        label="metrics summary",
    )

    routing_table = pd.DataFrame(routing_rows)
    saved_routing = pd.read_csv(AGG / "routing_by_seed.csv")
    assert_close(
        routing_table[saved_routing.columns[2:]],
        saved_routing[saved_routing.columns[2:]],
        atol=2e-12,
        label="routing table",
    )
    routing_summary = (
        routing_table.groupby("rule")[["auto_coverage", "acc_auto_classified", "error_capture_deferred"]]
        .agg(["mean", "std"])
    )

    pmeans = []
    pvars = []
    lmeans = []
    lvars = []
    for run in runs:
        draws = run["draws"].astype(np.float64)
        pmeans.append(draws.mean(axis=0))
        pvars.append(draws.var(axis=0, ddof=0))
        clipped = np.clip(draws, 1e-7, 1 - 1e-7)
        logits = np.log(clipped / (1 - clipped))
        lmeans.append(logits.mean(axis=0))
        lvars.append(logits.var(axis=0, ddof=0))
    P = np.column_stack(pmeans)
    V = np.column_stack(pvars)
    L = np.column_stack(lmeans)
    LV = np.column_stack(lvars)
    doc_prob_sample = decompose(P, V, between_ddof=1)
    doc_logit_sample = decompose(L, LV, between_ddof=1)
    doc_column_means = np.column_stack(
        [run["docs"]["p_mean"].to_numpy(float) for run in runs]
    )
    doc_column_variances = np.column_stack(
        [run["docs"]["var"].to_numpy(float) for run in runs]
    )
    doc_prob_columns = decompose(
        doc_column_means, doc_column_variances, between_ddof=1
    )
    saved_doc_prob = pd.read_csv(AGG / "document_decomposition_prob_from_draws.csv")
    saved_doc_logit = pd.read_csv(AGG / "document_decomposition_logit_from_draws.csv")
    saved_doc_columns = pd.read_csv(
        AGG / "document_decomposition_prob_from_columns.csv"
    )
    assert_close(doc_prob_sample, saved_doc_prob, atol=2e-12, label="document probability decomposition")
    assert_close(doc_logit_sample, saved_doc_logit, atol=2e-12, label="document logit decomposition")
    assert_close(
        doc_prob_columns,
        saved_doc_columns,
        atol=2e-12,
        label="document summary-column decomposition",
    )

    panels = [run["panel"] for run in runs]
    Z = np.column_stack([panel["z_mean"].to_numpy(float) for panel in panels])
    ZV = np.column_stack([panel["z_sd"].to_numpy(float) ** 2 for panel in panels])
    D = np.column_stack([panel["delta_mean"].to_numpy(float) for panel in panels])
    token_decomp = decompose(Z, ZV, between_ddof=1)
    saved_token = pd.read_csv(AGG / "token_decomposition_panel.csv")
    assert_close(
        token_decomp,
        saved_token[token_decomp.columns],
        atol=2e-12,
        label="token decomposition",
    )
    selected_mask = D.mean(axis=1) >= 0.5

    pooled = np.concatenate([run["draws"] for run in runs], axis=0).astype(np.float64)
    pooled_mean = pooled.mean(axis=0)
    pooled_lo = np.quantile(pooled, 0.025, axis=0)
    pooled_hi = np.quantile(pooled, 0.975, axis=0)
    pooled_docs = pd.DataFrame(
        {
            "y_true": y_ref,
            "p_mean": pooled_mean,
            "ci95_lo": pooled_lo,
            "ci95_hi": pooled_hi,
        }
    )
    ensemble = {
        "n_pooled_draws": pooled.shape[0],
        **metrics(pooled_mean, y_ref),
        "mean_ci_width_ensemble": float((pooled_hi - pooled_lo).mean()),
        "median_ci_width_ensemble": float(np.median(pooled_hi - pooled_lo)),
        "mean_ci_width_single_fit": float(seed_table["mean_ci_width"].mean()),
        **routing(pooled_docs, 0.5),
    }
    # Saved ensemble omits ECE and uses a different field name for coverage.
    saved_ensemble = pd.read_csv(AGG / "seed_ensemble_summary.csv").iloc[0]
    ensemble_map = {
        "n_pooled_draws": ensemble["n_pooled_draws"],
        "acc": ensemble["acc"],
        "f1": ensemble["f1"],
        "auc": ensemble["auc"],
        "brier": ensemble["brier"],
        "nll": ensemble["nll"],
        "mean_ci_width_ensemble": ensemble["mean_ci_width_ensemble"],
        "median_ci_width_ensemble": ensemble["median_ci_width_ensemble"],
        "mean_ci_width_single_fit": ensemble["mean_ci_width_single_fit"],
        "auto_coverage_tau0.5": ensemble["auto_coverage"],
        "acc_auto_classified": ensemble["acc_auto_classified"],
        "error_capture_deferred": ensemble["error_capture_deferred"],
    }
    assert_close(
        np.array(list(ensemble_map.values()), dtype=float),
        saved_ensemble[list(ensemble_map)].to_numpy(float),
        # The notebook pooled the saved float32 arrays without promoting them.
        atol=1e-7,
        label="seed ensemble",
    )

    # The pooled draws are an equal-weight empirical mixture over the ten fits.
    # Its exact population-variance identity uses ddof=0 for both components.
    pooled_var = pooled.var(axis=0, ddof=0)
    doc_prob_population = decompose(P, V, between_ddof=0)
    mixture_identity_max_abs_error = float(
        np.max(np.abs(pooled_var - doc_prob_population["total"].to_numpy()))
    )
    if mixture_identity_max_abs_error > 1e-12:
        raise AssertionError("population mixture variance identity failed")

    prior_table, prior_relative_ranges = validate_prior()

    weighted_param = pd.concat(
        [run["posterior"].assign(seed=run["seed"]) for run in runs], ignore_index=True
    )
    weighted_param_by_seed = (
        weighted_param.groupby("seed")
        .apply(
            lambda frame: np.average(
                frame["mean_posterior_sd"], weights=frame["n_params"]
            ),
            include_groups=False,
        )
        .rename("weighted_mean_posterior_sd")
    )
    saved_param = pd.read_csv(AGG / "param_posterior_sd_by_seed.csv")
    if weighted_param["module"].tolist() != saved_param["module"].tolist():
        raise AssertionError("posterior SD module aggregate differs")
    param_numeric = [column for column in saved_param.columns if column != "module"]
    assert_close(
        weighted_param[param_numeric],
        saved_param[param_numeric],
        atol=2e-12,
        label="posterior SD aggregate",
    )

    report = {
        "status": "PASS_WITH_INTERPRETATION_CORRECTIONS",
        "notebooks": notebooks,
        "official_01d_copy_identical": official_copy_identical,
        "drive_inventory": {
            "seed_count": len(runs),
            "seeds": SEEDS,
            "test_documents_per_seed": len(y_ref),
            "positive_rate": float(y_ref.mean()),
            "mc_draws_per_fit": int(runs[0]["draws"].shape[0]),
            "token_panel_documents": 200,
            "token_panel_positions": int(len(panel_ref)),
            "bayesian_parameters_per_fit": int(runs[0]["posterior"]["n_params"].sum()),
            "data_splits": split_inventory,
        },
        "metrics_summary": metric_summary.set_index("metric").to_dict(orient="index"),
        "routing_summary": {
            rule: {
                f"{metric}_{stat}": float(routing_summary.loc[rule, (metric, stat)])
                for metric in ["auto_coverage", "acc_auto_classified", "error_capture_deferred"]
                for stat in ["mean", "std"]
            }
            for rule in routing_summary.index
        },
        "document_probability_decomposition_saved_convention": summarize_decomp(doc_prob_sample),
        "document_probability_summary_column_decomposition": summarize_decomp(
            doc_prob_columns
        ),
        "document_probability_decomposition_population_convention": summarize_decomp(doc_prob_population),
        "document_logit_decomposition_saved_convention": summarize_decomp(doc_logit_sample),
        "token_decomposition_saved_convention_all_positions": summarize_decomp(token_decomp),
        "token_decomposition_saved_convention_selected_positions": summarize_decomp(
            token_decomp[selected_mask]
        ),
        "selected_token_positions": int(selected_mask.sum()),
        "selected_token_fraction": float(selected_mask.mean()),
        "seed_ensemble_empirical_mixture": ensemble,
        "mixture_variance_identity": {
            "max_absolute_error": mixture_identity_max_abs_error,
            "correct_between_ddof": 0,
            "analysis_notebook_between_ddof": 1,
            "ddof1_between_variance_inflation_factor": len(runs) / (len(runs) - 1),
        },
        "posterior_sd": {
            "weighted_mean_across_seeds": float(weighted_param_by_seed.mean()),
            "weighted_sd_across_seeds": float(weighted_param_by_seed.std(ddof=1)),
            "weighted_min": float(weighted_param_by_seed.min()),
            "weighted_max": float(weighted_param_by_seed.max()),
            "initial_sd": float(runs[0]["posterior"]["init_sd_from_init_rho"].iloc[0]),
        },
        "prior_sensitivity": prior_table.to_dict(orient="records"),
        "prior_relative_ranges": prior_relative_ranges,
        "scope_caveats": [
            "The project test set is explicitly labeled as previously inspected; this is not a fresh blind holdout.",
            "The ten fits vary optimizer/random seeds at fixed data and hyperparameters; they do not sample a formal posterior over model backbones.",
            "The pooled seed ensemble is an empirical refit-aware mixture, not a demonstrated calibrated total-uncertainty posterior.",
            "The prior sweep uses one seed, so it supports local sensitivity only.",
            "No bootstrap or independent training-sample resampling is present.",
        ],
    }
    report = json_clean(report)
    (OUT / "validation_results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    seed_table.to_csv(OUT / "recomputed_seed_metrics.csv", index=False)
    routing_table.to_csv(OUT / "recomputed_routing.csv", index=False)
    prior_table.to_csv(OUT / "recomputed_prior_sensitivity.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
