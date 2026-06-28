"""
Regenerate the main-manuscript figures for paper/figures/.

Produces the three data-driven main figures using the SAME canonical
configuration as benchmark_static_vs_dynamic.py (80/20 chronological split,
12-step held-out horizon, modern dynasir Model API):

- sird_compartments_visualization.png  (Fig 1: S, I, R, D over full history)
- time_dependent_rates_visualization.png (Fig 2: alpha/beta/gamma over history)
- forecasting_analysis_visualization.png (Fig 4: 12-step forecast vs actual)

This supersedes the legacy notebook (notebooks/report/report.py), which was
pinned to an obsolete 2020-training / 30-step configuration and produced
figures inconsistent with the manuscript benchmark.

DATA VINTAGE NOTE: process_data_from_owid fetches the live OWID series, which
is frozen and currently returns exactly 2,319 daily observations
(2020-01-11 -> 2026-05-17), so the 80% split lands on 2025-02-07 / forecast
start 2025-02-08, matching the manuscript. The offline fallback CSV bundled
with dynasir ends 2025-09-28; an offline re-run would shift the split and
desync the figures from the text. Regenerate with network access.

Usage:
    python regenerate_main_figures.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from dynasir import process_data_from_owid, DataContainer, Model
except ImportError:
    print("ERROR: dynasir package not installed.")
    raise SystemExit(1)

SPLIT_RATIO = 0.8
EVAL_STEPS = 12
MAX_LAG = 3

FIGURES_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (core, full): the canonical SIRD-ready frame (dropna on C/D/I/R/N,
    used for the 80/20 split) and the full container frame reindexed to it
    (carrying S and the recovered rate columns for the descriptive figures)."""
    print("Loading COVID-19 data from Our World in Data...")
    data = process_data_from_owid(include_vaccination=False)
    cd = DataContainer(data).data
    core = cd[["C", "D", "I", "R", "N"]].dropna()
    full = cd.reindex(core.index)
    return core, full


def fig_compartments(full: pd.DataFrame, n_fit: int) -> None:
    """Fig 1: SIRD compartments (S, I, R, D) over the full observed history."""
    print("Figure 1: SIRD compartments...")
    dates = full.index
    forecast_start = dates[n_fit]

    panels = [
        ("S", "Susceptible Population (S)", "#00B3B3"),
        ("I", "Infected Population (I)", "#FF6B6B"),
        ("R", "Recovered Population (R)", "#7ED321"),
        ("D", "Deceased Population (D)", "#2D3748"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("SIRD Model Compartments Over Time", fontsize=16, fontweight="bold")

    for idx, (col, title, color) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        ax.plot(dates, full[col].values, color=color, linewidth=2.0, label=title)
        ax.axvline(forecast_start, color="purple", linestyle="--", linewidth=1.5,
                   alpha=0.5, label="Forecast Start")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Population Count", fontsize=12)
        ax.set_xlabel("Date", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.margins(x=0.01)

    plt.tight_layout()
    out = FIGURES_DIR / "sird_compartments_visualization.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def fig_rates(full: pd.DataFrame, n_fit: int) -> None:
    """Fig 2: time-varying alpha/beta/gamma with 7-day average and +/-1 SD band."""
    print("Figure 2: time-dependent rates...")
    dates = full.index
    rate_info = {
        "alpha": ("#E74C3C", r"$\alpha(t)$ - Infection Rate",
                  "Rate of transmission from susceptible to infected"),
        "beta": ("#3498DB", r"$\beta(t)$ - Recovery Rate",
                 "Rate of recovery from infected to recovered"),
        "gamma": ("#9B59B6", r"$\gamma(t)$ - Mortality Rate",
                  "Rate of mortality from infected to deceased"),
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle("Time-Dependent Epidemiological Rates", fontsize=16, fontweight="bold")

    for i, (rate, (color, label, desc)) in enumerate(rate_info.items()):
        ax = axes[i]
        series = full[rate]
        ax.plot(dates, series.values, color=color, linewidth=1.5, alpha=0.55, label=label)
        rolling = series.rolling(window=7, center=True).mean()
        ax.plot(dates, rolling.values, color=color, linewidth=2.5, alpha=0.9,
                linestyle="--", label=f"{label} (7-day avg)")

        mean_val = series.mean()
        std_val = series.std()
        ax.axhline(mean_val, color=color, linestyle=":", alpha=0.5,
                   label=f"Mean: {mean_val:.4f}")
        ax.fill_between(dates, mean_val - std_val, mean_val + std_val,
                        color=color, alpha=0.1, label=r"$\pm 1$ SD")

        ax.set_title(f"{label}\n{desc}", fontsize=13, fontweight="bold")
        ax.set_ylabel("Rate Value", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.margins(x=0.01)
        if i < 2:
            ax.tick_params(axis="x", labelbottom=False)

    axes[2].set_xlabel("Date", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "time_dependent_rates_visualization.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def fig_forecast(core: pd.DataFrame, n_fit: int) -> None:
    """Fig 4: 12-step held-out forecast vs actual, on DAILY INCIDENCE.

    Plotting incidence (rather than cumulative compartments) matches the
    paper's primary metric (Table 1 reports incident MAPE) and the Methods
    argument that cumulative totals are nearly stationary over a 12-day
    horizon. It also removes the constant ~10k level offset that arises when
    the smoothed/inverted boundary reconstruction of the cumulative series is
    compared against the raw observed endpoint -- an offset that dominates the
    near-flat cumulative band but cancels under first differencing.

    Incident new cases (DeltaC) and new deaths (DeltaD) are first differences
    of the cumulative forecast/actual trajectories over the horizon (so the
    degenerate boundary increment, which mixes the reconstructed level with the
    observed anchor, is excluded). Active infections (I) are a prevalence and
    are shown at level.
    """
    print("Figure 4: forecasting analysis (12-step held-out, daily incidence)...")
    train = core.iloc[:n_fit].copy()
    test = core.iloc[n_fit:n_fit + EVAL_STEPS].copy()

    container = DataContainer(train)
    model = Model(container)
    model.create_model()
    model.fit_model(max_lag=MAX_LAG)
    model.forecast(steps=EVAL_STEPS)
    model.run_simulations(n_jobs=1)
    model.generate_result()

    # (compartment, colour, panel title, y-label, is_incident)
    panels = [
        ("C", "#E53E3E", "New Confirmed Cases", "New cases (count / day)", True),
        ("D", "#2D3748", "New Deaths", "New deaths (count / day)", True),
        ("I", "#D69E2E", "Active Infected", "Active infected (count)", False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(
        "COVID-19 12-Step Held-Out Forecast vs. Actual Data (daily incidence)",
        fontsize=20, fontweight="bold", y=0.98,
    )

    for idx, (comp, color, title, ylabel, is_incident) in enumerate(panels):
        ax = axes[idx]

        ensemble = np.asarray(model.results[comp].values, dtype=float)  # (12, n_scen)
        fc_dates = model.results[comp].index
        actual = test[comp].to_numpy()

        if is_incident:
            ensemble = np.diff(ensemble, axis=0)   # (11, n_scen) daily increments
            actual = np.diff(actual)               # (11,)
            dates = fc_dates[1:]
        else:
            dates = fc_dates

        y_min = np.nanmin(ensemble, axis=1)
        y_max = np.nanmax(ensemble, axis=1)
        y_q25 = np.nanpercentile(ensemble, 25, axis=1)
        y_q75 = np.nanpercentile(ensemble, 75, axis=1)
        y_mean = np.nanmean(ensemble, axis=1)

        ax.fill_between(dates, y_min, y_max, color=color, alpha=0.15,
                        label="Forecast Range", zorder=1)
        ax.fill_between(dates, y_q25, y_q75, color=color, alpha=0.30,
                        label="IQR (25-75%)", zorder=2)
        ax.plot(dates, y_mean, color=color, linewidth=3.0, linestyle="--",
                label="Mean Forecast", zorder=3)
        ax.plot(dates, actual, color="#1A202C", linewidth=3.5,
                marker="o", markersize=5, markerfacecolor="white",
                markeredgecolor="#1A202C", label="Actual", zorder=5)

        ax.set_title(f"{title} - 12-Step Forecast", fontsize=15, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=13, fontweight="bold")
        ax.set_xlabel("Date", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.margins(x=0.02)

    plt.tight_layout()
    out = FIGURES_DIR / "forecasting_analysis_visualization.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  saved {out.name}")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    core, full = load_data()
    n_fit = int(len(core) * SPLIT_RATIO)
    print(f"Observations: {len(core)} | split at index {n_fit} "
          f"(train ends {core.index[n_fit - 1].date()}, "
          f"test starts {core.index[n_fit].date()})")
    fig_compartments(full, n_fit)
    fig_rates(full, n_fit)
    fig_forecast(core, n_fit)
    print("Done.")


if __name__ == "__main__":
    main()
