"""
Sensitivity Diagnostics for Hybrid Epidemic Intelligence Manuscript.

This script evaluates forecast sensitivity to:
1) Savitzky-Golay smoothing window size
2) VAR maximum lag configuration

Outputs are written to paper/companion_figures/sensitivity:
- sensitivity_window_results.csv
- sensitivity_lag_results.csv
- sensitivity_window_heatmap_mape_cases.png
- sensitivity_lag_tradeoff.png
- residual_acf_pacf_cases.png

Usage:
    python sensitivity_diagnostics.py
"""

# pyright: reportMissingImports=false, reportMissingModuleSource=false

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except Exception:  # pragma: no cover - optional dependency behavior
    plot_acf = None
    plot_pacf = None

try:
    from dynasir import DataContainer, Model, process_data_from_owid
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "dynasir package is required. Install with: pip install dynasir>=1.0.0"
    ) from exc


WINDOW_GRID = [3, 5, 7, 10, 14]
LAG_GRID = [1, 2, 3, 4, 6, 8, 12]
SPLIT_RATIO = 0.8


def _safe_container(data: pd.DataFrame, window: int) -> DataContainer:
    """Create DataContainer with fallback if window arg is unsupported."""
    try:
        return DataContainer(data, window=window)
    except TypeError:
        return DataContainer(data)


def _extract_forecasts(model: Model, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract forecast arrays for C and D with robust fallbacks."""
    try:
        sim_c = model.results["simulations"].get("C", None)
        sim_d = model.results["simulations"].get("D", None)
        if sim_c is not None and sim_d is not None:
            c_hat = sim_c.mean(axis=1)
            d_hat = sim_d.mean(axis=1)
            c_hat = c_hat.values if hasattr(c_hat, "values") else np.asarray(c_hat)
            d_hat = d_hat.values if hasattr(d_hat, "values") else np.asarray(d_hat)
            return c_hat[:n_test], d_hat[:n_test]
    except Exception:
        pass

    c_alt = getattr(model.forecasting_box, "C_point", None)
    d_alt = getattr(model.forecasting_box, "D_point", None)
    if c_alt is None or d_alt is None:
        raise RuntimeError("Could not extract forecasts from model results")

    c_hat = np.asarray(c_alt)[:n_test]
    d_hat = np.asarray(d_alt)[:n_test]
    return c_hat, d_hat


def _evaluate_configuration(
    data_df: pd.DataFrame,
    window: int,
    max_lag: int,
) -> dict[str, Any]:
    """Fit/evaluate one (window, lag) configuration."""
    n_fit = int(len(data_df) * SPLIT_RATIO)
    data_test = data_df.iloc[n_fit:]

    container = _safe_container(data_df, window=window)
    model = Model(container)
    model.create_model()
    model.fit_model(max_lag=max_lag)

    n_test = len(data_test)
    model.forecast(steps=n_test)
    model.run_simulations(n_jobs=1)
    model.generate_result()

    c_hat, d_hat = _extract_forecasts(model, n_test)

    c_obs = np.asarray(data_test["C"], dtype=float)
    d_obs = np.asarray(data_test["D"], dtype=float)

    mae_c = mean_absolute_error(c_obs, c_hat)
    rmse_c = np.sqrt(mean_squared_error(c_obs, c_hat))
    mape_c = np.mean(np.abs((c_obs - c_hat) / (c_obs + 1.0))) * 100

    mae_d = mean_absolute_error(d_obs, d_hat)
    rmse_d = np.sqrt(mean_squared_error(d_obs, d_hat))
    mape_d = np.mean(np.abs((d_obs - d_hat) / (d_obs + 1.0))) * 100

    return {
        "window": window,
        "max_lag": max_lag,
        "mae_cases": float(mae_c),
        "rmse_cases": float(rmse_c),
        "mape_cases": float(mape_c),
        "mae_deaths": float(mae_d),
        "rmse_deaths": float(rmse_d),
        "mape_deaths": float(mape_d),
        "model": model,
    }


def _plot_window_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    pivot = df.pivot_table(index="window", columns="max_lag", values="mape_cases")

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "MAPE Cases (%)"})
    plt.title("Sensitivity Heatmap: Cases MAPE by Smoothing Window and VAR Lag")
    plt.xlabel("VAR max_lag")
    plt.ylabel("Smoothing window")
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_window_heatmap_mape_cases.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_lag_tradeoff(df: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        df.groupby("max_lag", as_index=False)[["mape_cases", "mape_deaths"]]
        .mean()
        .sort_values("max_lag")
    )

    plt.figure(figsize=(10, 6))
    plt.plot(summary["max_lag"], summary["mape_cases"], marker="o", linewidth=2, label="Cases MAPE")
    plt.plot(summary["max_lag"], summary["mape_deaths"], marker="s", linewidth=2, label="Deaths MAPE")
    plt.title("Lag Sensitivity Tradeoff (Average Across Window Grid)")
    plt.xlabel("VAR max_lag")
    plt.ylabel("MAPE (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sensitivity_lag_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()


def _plot_residual_diagnostics(
    observed: pd.Series,
    predicted: np.ndarray,
    out_dir: Path,
) -> None:
    if plot_acf is None or plot_pacf is None:
        print("Skipping ACF/PACF plot: statsmodels plotting not available.")
        return

    residuals = observed.values - predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(residuals, ax=axes[0], lags=30)
    axes[0].set_title("ACF of Forecast Residuals (Cases)")
    plot_pacf(residuals, ax=axes[1], lags=30, method="ywm")
    axes[1].set_title("PACF of Forecast Residuals (Cases)")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_acf_pacf_cases.png", dpi=300, bbox_inches="tight")
    plt.close()


def run_sensitivity() -> None:
    """Execute window/lag sensitivity and generate report artifacts."""
    print("Loading OWID data via dynasir...")
    raw = process_data_from_owid(include_vaccination=False)
    base = DataContainer(raw)
    data_df = base.data[["C", "D", "I", "R", "N"]].dropna()

    out_dir = Path(__file__).resolve().parents[2] / "paper" / "companion_figures" / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running sensitivity grid: windows={WINDOW_GRID}, lags={LAG_GRID}")
    records: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None

    for window in WINDOW_GRID:
        for lag in LAG_GRID:
            print(f"  -> window={window}, max_lag={lag}")
            try:
                result = _evaluate_configuration(data_df, window=window, max_lag=lag)
                records.append({k: v for k, v in result.items() if k != "model"})
                if best_record is None or result["mape_cases"] < best_record["mape_cases"]:
                    best_record = result
            except Exception as exc:
                print(f"     failed: {exc}")
                records.append(
                    {
                        "window": window,
                        "max_lag": lag,
                        "mae_cases": np.nan,
                        "rmse_cases": np.nan,
                        "mape_cases": np.nan,
                        "mae_deaths": np.nan,
                        "rmse_deaths": np.nan,
                        "mape_deaths": np.nan,
                    }
                )

    results_df = pd.DataFrame(records)

    window_summary = (
        results_df.groupby("window", as_index=False)[["mae_cases", "rmse_cases", "mape_cases", "mae_deaths", "rmse_deaths", "mape_deaths"]]
        .mean(numeric_only=True)
        .sort_values("window")
    )
    lag_summary = (
        results_df.groupby("max_lag", as_index=False)[["mae_cases", "rmse_cases", "mape_cases", "mae_deaths", "rmse_deaths", "mape_deaths"]]
        .mean(numeric_only=True)
        .sort_values("max_lag")
    )

    window_summary.to_csv(out_dir / "sensitivity_window_results.csv", index=False)
    lag_summary.to_csv(out_dir / "sensitivity_lag_results.csv", index=False)
    results_df.to_csv(out_dir / "sensitivity_full_grid_results.csv", index=False)

    _plot_window_heatmap(results_df, out_dir)
    _plot_lag_tradeoff(results_df, out_dir)

    if best_record is not None:
        n_fit = int(len(data_df) * SPLIT_RATIO)
        observed_cases = data_df.iloc[n_fit:]["C"]
        c_hat, _ = _extract_forecasts(best_record["model"], len(observed_cases))
        _plot_residual_diagnostics(observed_cases, c_hat, out_dir)

        print("Best case-MAPE configuration:")
        print(
            f"  window={best_record['window']}, max_lag={best_record['max_lag']}, "
            f"mape_cases={best_record['mape_cases']:.3f}%"
        )

    print(f"Sensitivity artifacts saved to: {out_dir}")


if __name__ == "__main__":
    run_sensitivity()
