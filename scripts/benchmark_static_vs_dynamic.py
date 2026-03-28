"""
Benchmarking Script: Multi-Baseline Comparison for Hybrid Dynamic SIRD.

Compares four forecasting approaches on the same held-out split:
1. Static SIRD (constant parameters)
2. Naive persistence baseline
3. Linear trend extrapolation baseline
4. Dynamic SIRD (dynasir)

Outputs:
- benchmark_multibaseline_results.csv
- benchmark_summary_ranked.csv
- benchmark_comparison.csv (backward-compatible dynamic vs static view)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

try:
    from dynasir import DataContainer, Model, process_data_from_owid
except ImportError:
    print("ERROR: dynasir package not installed.")
    print("Install with: pip install dynasir>=1.0.0")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1.0)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": _safe_mape(y_true, y_pred),
    }


def _to_numpy_1d(obj, target_len: int | None = None) -> np.ndarray | None:
    arr = None

    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        for candidate in ["point", "mean", "median", "value", "forecast"]:
            if candidate in obj.columns:
                arr = obj[candidate].to_numpy()
                break
        if arr is None and obj.shape[1] > 0:
            arr = obj.iloc[:, 0].to_numpy()
    elif isinstance(obj, pd.Series):
        arr = obj.to_numpy()
    elif isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.asarray(obj)
    else:
        return None

    if arr is None:
        return None

    arr = np.asarray(arr).astype(float).reshape(-1)
    if target_len is not None:
        # dynasir outputs may include history + forecast; use the trailing window.
        if len(arr) >= target_len:
            arr = arr[-target_len:]
        else:
            arr = arr[:target_len]
    return arr


def _extract_dynamic_series(model: Model, compartment: str, steps: int) -> np.ndarray | None:
    # Primary path: model.results[compartment]
    try:
        if model.results is not None and compartment in model.results:
            arr = _to_numpy_1d(model.results[compartment], target_len=steps)
            if arr is not None and len(arr) >= steps:
                return arr
    except Exception:
        pass

    # Fallback path: central simulation trajectory
    try:
        sim = model.simulation
        if sim is not None:
            central = sim["point"]["point"]["point"]
            if isinstance(central, pd.DataFrame) and compartment in central.columns:
                arr = central[compartment].to_numpy()[:steps]
                return arr.astype(float)
    except Exception:
        pass

    # Final fallback path: forecasting_box fields
    try:
        fb = model.forecasting_box
        for attr in [f"{compartment}_point", f"{compartment}_mean", compartment]:
            if hasattr(fb, attr):
                arr = _to_numpy_1d(getattr(fb, attr), target_len=steps)
                if arr is not None and len(arr) >= steps:
                    return arr
    except Exception:
        pass

    return None


def _extract_dynamic_interval(
    model: Model, compartment: str, steps: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    lower = None
    upper = None

    # Try forecasting_box columns first
    try:
        fb = model.forecasting_box
        low_attr = f"{compartment}_lower"
        up_attr = f"{compartment}_upper"
        if hasattr(fb, low_attr):
            lower = _to_numpy_1d(getattr(fb, low_attr), target_len=steps)
        if hasattr(fb, up_attr):
            upper = _to_numpy_1d(getattr(fb, up_attr), target_len=steps)
    except Exception:
        pass

    # Fallback: simulation quantiles across scenarios
    if lower is None or upper is None:
        try:
            sim = model.simulation
            if sim is not None:
                paths = []
                for l1 in ["lower", "point", "upper"]:
                    for l2 in ["lower", "point", "upper"]:
                        for l3 in ["lower", "point", "upper"]:
                            df = sim[l1][l2][l3]
                            if isinstance(df, pd.DataFrame) and compartment in df.columns:
                                paths.append(df[compartment].to_numpy()[:steps].astype(float))
                if paths:
                    stack = np.vstack(paths)
                    if lower is None:
                        lower = np.nanpercentile(stack, 2.5, axis=0)
                    if upper is None:
                        upper = np.nanpercentile(stack, 97.5, axis=0)
        except Exception:
            pass

    return lower, upper


def _calc_coverage(y_true: np.ndarray, lo: np.ndarray | None, hi: np.ndarray | None) -> float | None:
    if lo is None or hi is None:
        return None
    n = min(len(y_true), len(lo), len(hi))
    if n == 0:
        return None
    inside = (y_true[:n] >= lo[:n]) & (y_true[:n] <= hi[:n])
    return float(np.mean(inside) * 100.0)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def ode_sird_static(y, t, beta, gamma, mu, population):
    susceptible, infected, recovered, deaths = y
    d_s = -beta * susceptible * infected / population
    d_i = beta * susceptible * infected / population - (gamma + mu) * infected
    d_r = gamma * infected
    d_d = mu * infected
    return [d_s, d_i, d_r, d_d]


def simulate_static_sird(s0, i0, r0, d0, population, beta, gamma, mu, days):
    y0 = [s0, i0, r0, d0]
    t = np.arange(0, days, 1)
    solution = odeint(ode_sird_static, y0, t, args=(beta, gamma, mu, population))
    return solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]


def fit_static_sird(
    data_df: pd.DataFrame, split_ratio: float = 0.8, eval_steps: int = 12
) -> dict:
    n_train = int(len(data_df) * split_ratio)
    train = data_df.iloc[:n_train].copy()
    test = data_df.iloc[n_train:n_train + eval_steps].copy()

    population = float(train["N"].iloc[0])
    i0 = float(train["I"].iloc[0])
    r0 = float(train["R"].iloc[0])
    d0 = float(train["D"].iloc[0])
    s0 = float(population - i0 - r0 - d0)

    def objective(params):
        beta, gamma, mu = params
        if any(p <= 0 or p >= 1 for p in params):
            return 1e12
        try:
            _, i_pred, r_pred, d_pred = simulate_static_sird(
                s0, i0, r0, d0, population, beta, gamma, mu, len(train)
            )
            c_pred = i_pred + r_pred + d_pred
            return float(
                np.sum((c_pred - train["C"].to_numpy()) ** 2)
                + np.sum((d_pred - train["D"].to_numpy()) ** 2)
            )
        except Exception:
            return 1e12

    result = minimize(objective, x0=[0.5, 0.1, 0.01], method="Nelder-Mead", options={"maxiter": 5000})
    beta_fit, gamma_fit, mu_fit = result.x

    _, i_all, r_all, d_all = simulate_static_sird(
        s0, i0, r0, d0, population, beta_fit, gamma_fit, mu_fit, len(data_df)
    )
    c_all = i_all + r_all + d_all

    c_pred = c_all[n_train:n_train + len(test)]
    d_pred = d_all[n_train:n_train + len(test)]

    return {
        "name": "Static SIRD",
        "train": train,
        "test": test,
        "pred": {"C": c_pred, "D": d_pred},
        "params": {"beta": float(beta_fit), "gamma": float(gamma_fit), "mu": float(mu_fit)},
    }


def fit_naive_persistence(
    data_df: pd.DataFrame, split_ratio: float = 0.8, eval_steps: int = 12
) -> dict:
    n_train = int(len(data_df) * split_ratio)
    train = data_df.iloc[:n_train].copy()
    test = data_df.iloc[n_train:n_train + eval_steps].copy()

    c_last = float(train["C"].iloc[-1])
    d_last = float(train["D"].iloc[-1])

    c_pred = np.full(len(test), c_last)
    d_pred = np.full(len(test), d_last)

    return {
        "name": "Naive Persistence",
        "train": train,
        "test": test,
        "pred": {"C": c_pred, "D": d_pred},
    }


def fit_linear_trend(
    data_df: pd.DataFrame, split_ratio: float = 0.8, eval_steps: int = 12
) -> dict:
    n_train = int(len(data_df) * split_ratio)
    train = data_df.iloc[:n_train].copy()
    test = data_df.iloc[n_train:n_train + eval_steps].copy()

    x_train = np.arange(len(train), dtype=float)
    x_test = np.arange(len(train), len(train) + len(test), dtype=float)

    c_coef = np.polyfit(x_train, train["C"].to_numpy(), deg=1)
    d_coef = np.polyfit(x_train, train["D"].to_numpy(), deg=1)

    c_pred = np.polyval(c_coef, x_test)
    d_pred = np.polyval(d_coef, x_test)

    c_pred = np.maximum.accumulate(np.maximum(c_pred, 0.0))
    d_pred = np.maximum.accumulate(np.maximum(d_pred, 0.0))

    return {
        "name": "Linear Trend",
        "train": train,
        "test": test,
        "pred": {"C": c_pred, "D": d_pred},
        "params": {"cases_slope": float(c_coef[0]), "deaths_slope": float(d_coef[0])},
    }


def fit_dynamic_sird(
    data_df: pd.DataFrame,
    split_ratio: float = 0.8,
    max_lag: int = 3,
    eval_steps: int = 12,
) -> dict | None:
    n_train = int(len(data_df) * split_ratio)
    train = data_df.iloc[:n_train].copy()
    test = data_df.iloc[n_train:n_train + eval_steps].copy()

    container = DataContainer(train)
    model = Model(container)

    try:
        model.create_model()
        model.fit_model(max_lag=max_lag)
        model.forecast(steps=len(test))
        model.run_simulations(n_jobs=1)
        model.generate_result()
    except Exception as exc:
        print(f"Dynamic model error: {exc}")
        return None

    c_pred = _extract_dynamic_series(model, "C", len(test))
    d_pred = _extract_dynamic_series(model, "D", len(test))

    if c_pred is None or d_pred is None:
        print("Dynamic model error: could not extract C/D forecasts from dynasir outputs")
        return None

    c_lo, c_hi = _extract_dynamic_interval(model, "C", len(test))
    d_lo, d_hi = _extract_dynamic_interval(model, "D", len(test))

    return {
        "name": "Dynamic SIRD (dynasir)",
        "train": train,
        "test": test,
        "pred": {"C": c_pred, "D": d_pred},
        "coverage": {
            "cases_95": _calc_coverage(test["C"].to_numpy(), c_lo, c_hi),
            "deaths_95": _calc_coverage(test["D"].to_numpy(), d_lo, d_hi),
        },
        "model": model,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_results_table(results: list[dict]) -> pd.DataFrame:
    rows = []

    for res in results:
        test = res["test"]
        c_true = test["C"].to_numpy()
        d_true = test["D"].to_numpy()
        c_pred = np.asarray(res["pred"]["C"], dtype=float)[: len(test)]
        d_pred = np.asarray(res["pred"]["D"], dtype=float)[: len(test)]

        c_metrics = _compute_metrics(c_true, c_pred)
        d_metrics = _compute_metrics(d_true, d_pred)

        rows.append(
            {
                "model": res["name"],
                "horizon_days": len(test),
                "mae_cases": c_metrics["mae"],
                "rmse_cases": c_metrics["rmse"],
                "mape_cases": c_metrics["mape"],
                "mae_deaths": d_metrics["mae"],
                "rmse_deaths": d_metrics["rmse"],
                "mape_deaths": d_metrics["mape"],
                "coverage_cases_95": (res.get("coverage") or {}).get("cases_95"),
                "coverage_deaths_95": (res.get("coverage") or {}).get("deaths_95"),
            }
        )

    df = pd.DataFrame(rows)
    return df


def calc_improvement(reference: float, candidate: float) -> float:
    if reference == 0:
        return np.nan
    return (reference - candidate) / reference * 100.0


def main():
    print("=" * 78)
    print("BENCHMARK: Multi-Baseline Comparison for Hybrid Dynamic SIRD")
    print("=" * 78)

    print("\n[1/5] Loading COVID-19 data from Our World in Data...")
    data = process_data_from_owid(include_vaccination=False)
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    print("\n[2/5] Preparing SIRD-ready dataset...")
    container = DataContainer(data)
    assert container.data is not None, "DataContainer produced no data"
    data_fit = container.data[["C", "D", "I", "R", "N"]].dropna()
    print(f"Available rows: {len(data_fit)}")

    eval_steps = 12

    print(f"\n[3/5] Fitting baselines (evaluation horizon = {eval_steps} days)...")
    static_res = fit_static_sird(data_fit, split_ratio=0.8, eval_steps=eval_steps)
    naive_res = fit_naive_persistence(data_fit, split_ratio=0.8, eval_steps=eval_steps)
    trend_res = fit_linear_trend(data_fit, split_ratio=0.8, eval_steps=eval_steps)
    print("  - Static SIRD: done")
    print("  - Naive persistence: done")
    print("  - Linear trend: done")

    print("\n[4/5] Fitting dynamic model (dynasir)...")
    dynamic_res = fit_dynamic_sird(data_fit, split_ratio=0.8, max_lag=3, eval_steps=eval_steps)
    if dynamic_res is None:
        print("  - Dynamic SIRD failed; reporting baselines only")
        all_results = [static_res, naive_res, trend_res]
    else:
        print("  - Dynamic SIRD: done")
        all_results = [naive_res, trend_res, static_res, dynamic_res]

    print("\n[5/5] Building ranked summary...")
    results_df = build_results_table(all_results)

    ranked = results_df.sort_values(by=["mape_cases", "mape_deaths"], ascending=[True, True]).reset_index(drop=True)

    print("\nRANKING BY CASE MAPE (lower is better):")
    print(
        ranked[
            [
                "model",
                "mape_cases",
                "mape_deaths",
                "mae_cases",
                "mae_deaths",
                "coverage_cases_95",
            ]
        ].to_string(index=False)
    )

    # Backward-compatible static vs dynamic comparison table
    comparison_df = None
    if dynamic_res is not None:
        st = results_df[results_df["model"] == "Static SIRD"].iloc[0]
        dy = results_df[results_df["model"] == "Dynamic SIRD (dynasir)"].iloc[0]

        comparison_df = pd.DataFrame(
            {
                "Metric": [
                    "MAE (Cases)",
                    "RMSE (Cases)",
                    "MAPE (Cases) %",
                    "MAE (Deaths)",
                    "RMSE (Deaths)",
                    "MAPE (Deaths) %",
                ],
                "Static SIRD": [
                    st["mae_cases"],
                    st["rmse_cases"],
                    st["mape_cases"],
                    st["mae_deaths"],
                    st["rmse_deaths"],
                    st["mape_deaths"],
                ],
                "Dynamic SIRD": [
                    dy["mae_cases"],
                    dy["rmse_cases"],
                    dy["mape_cases"],
                    dy["mae_deaths"],
                    dy["rmse_deaths"],
                    dy["mape_deaths"],
                ],
                "Improvement": [
                    calc_improvement(st["mae_cases"], dy["mae_cases"]),
                    calc_improvement(st["rmse_cases"], dy["rmse_cases"]),
                    calc_improvement(st["mape_cases"], dy["mape_cases"]),
                    calc_improvement(st["mae_deaths"], dy["mae_deaths"]),
                    calc_improvement(st["rmse_deaths"], dy["rmse_deaths"]),
                    calc_improvement(st["mape_deaths"], dy["mape_deaths"]),
                ],
            }
        )

    # Save outputs
    out_dir = Path(__file__).resolve().parents[2] / "paper" / "companion_figures" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "benchmark_multibaseline_results.csv"
    ranked_path = out_dir / "benchmark_summary_ranked.csv"
    results_df.to_csv(results_path, index=False)
    ranked.to_csv(ranked_path, index=False)

    print(f"\nSaved: {results_path}")
    print(f"Saved: {ranked_path}")

    if comparison_df is not None:
        cmp_path = Path(__file__).resolve().parents[2] / "paper" / "benchmark_comparison.csv"
        comparison_df.to_csv(cmp_path, index=False)
        print(f"Saved: {cmp_path}")

        print("\nDynamic vs Static improvement (% reduction in error):")
        for _, row in comparison_df.iterrows():
            print(f"  - {row['Metric']}: {row['Improvement']:.2f}%")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
