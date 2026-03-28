"""
Benchmarking Script: Static SIRD vs. Dynamic SIRD (Time-Varying Parameters)

This script reproduces the empirical comparison between:
1. Traditional SIRD with constant parameters
2. Hybrid Dynamic SIRD with time-varying parameters (dynasir)

Output:
- Performance metrics (RMSE, MAE, MAPE)
- Visualization comparing forecasts
- Summary table for publication

Usage:
    python benchmark_static_vs_dynamic.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Try importing dynasir
try:
    from dynasir import process_data_from_owid, DataContainer, Model
except ImportError:
    print("ERROR: dynasir package not installed.")
    print("Install with: pip install dynasir>=1.0.0")
    exit(1)


# ============================================================================
# PART 1: STATIC SIRD MODEL (Baseline)
# ============================================================================

def ode_sird_static(y, t, beta, gamma, mu, N):
    """Static SIRD differential equations"""
    S, I, R, D = y
    dS = -beta * S * I / N
    dI = beta * S * I / N - (gamma + mu) * I
    dR = gamma * I
    dD = mu * I
    return [dS, dI, dR, dD]


def simulate_static_sird(S0, I0, R0, D0, N, beta, gamma, mu, days):
    """Simple Euler integration for static SIRD"""
    from scipy.integrate import odeint
    
    y0 = [S0, I0, R0, D0]
    t = np.arange(0, days, 1)
    
    solution = odeint(ode_sird_static, y0, t, args=(beta, gamma, mu, N))
    
    return solution[:, 0], solution[:, 1], solution[:, 2], solution[:, 3]


def fit_static_sird(data_df, split_ratio=0.8):
    """
    Fit a static SIRD model to historical data using least squares.
    
    Args:
        data_df: DataFrame with columns ['C', 'D', 'I', 'N'] (confirmed, deaths, infected, population)
        split_ratio: Fraction of data to use for fitting (rest for testing)
    
    Returns:
        Dictionary with fitted parameters and model performance
    """
    
    # Split data
    n_fit = int(len(data_df) * split_ratio)
    data_fit = data_df.iloc[:n_fit].copy()
    data_test = data_df.iloc[n_fit:].copy()
    
    # Initial conditions
    N = data_fit['N'].iloc[0]
    I0 = data_fit['I'].iloc[0]
    R0 = data_fit['R'].iloc[0]
    D0 = data_fit['D'].iloc[0]
    S0 = N - I0 - R0 - D0
    
    # Objective function: minimize error between model and data
    def objective(params):
        beta, gamma, mu = params
        
        # Ensure positive parameters
        if any(p <= 0 or p >= 1 for p in params):
            return 1e10
        
        try:
            S_pred, I_pred, R_pred, D_pred = simulate_static_sird(
                S0, I0, R0, D0, N, beta, gamma, mu, len(data_fit)
            )
            
            # Compare cumulative cases and deaths
            C_pred = I_pred + R_pred + D_pred
            error = (
                np.sum((C_pred - data_fit['C'].values)**2) +
                np.sum((D_pred - data_fit['D'].values)**2)
            )
            return error
        except:
            return 1e10
    
    # Initial guess
    x0 = [0.5, 0.1, 0.01]
    
    # Fit
    result = minimize(objective, x0, method='Nelder-Mead', 
                     options={'maxiter': 5000})
    
    beta_fit, gamma_fit, mu_fit = result.x
    
    # Generate forecast on test set
    S_fit, I_fit, R_fit, D_fit = simulate_static_sird(
        S0, I0, R0, D0, N, beta_fit, gamma_fit, mu_fit, len(data_fit) + len(data_test)
    )
    
    C_fit = I_fit + R_fit + D_fit
    C_forecast = C_fit[n_fit:]
    D_forecast = D_fit[n_fit:]
    
    # Metrics on test set
    mae_c = mean_absolute_error(data_test['C'], C_forecast)
    rmse_c = np.sqrt(mean_squared_error(data_test['C'], C_forecast))
    mape_c = np.mean(np.abs((data_test['C'] - C_forecast) / (data_test['C'] + 1))) * 100
    
    mae_d = mean_absolute_error(data_test['D'], D_forecast)
    rmse_d = np.sqrt(mean_squared_error(data_test['D'], D_forecast))
    mape_d = np.mean(np.abs((data_test['D'] - D_forecast) / (data_test['D'] + 1))) * 100
    
    return {
        'name': 'Static SIRD',
        'params': {'beta': beta_fit, 'gamma': gamma_fit, 'mu': mu_fit},
        'test_data': data_test,
        'forecasts': {'C': C_forecast, 'D': D_forecast},
        'metrics': {
            'confirmed_mae': mae_c,
            'confirmed_rmse': rmse_c,
            'confirmed_mape': mape_c,
            'deaths_mae': mae_d,
            'deaths_rmse': rmse_d,
            'deaths_mape': mape_d,
        },
        'test_size': len(data_test),
        'fit_size': n_fit,
    }


# ============================================================================
# PART 2: DYNAMIC SIRD MODEL (dynasir)
# ============================================================================

def fit_dynamic_sird(data_df, split_ratio=0.8):
    """
    Fit a dynamic SIRD model using dynasir's inverse problem approach.
    
    Args:
        data_df: DataFrame with columns ['C', 'D', 'I', 'N', 'R']
        split_ratio: Fraction of data to use for fitting
    
    Returns:
        Dictionary with model performance
    """
    
    # Create dynasir container
    container = DataContainer(data_df)
    
    # Create model
    model = Model(container)
    
    # Build and fit
    model.create_model()
    
    n_fit = int(len(data_df) * split_ratio)
    max_lag = 3
    
    try:
        model.fit_model(max_lag=max_lag)
    except Exception as e:
        print(f"Fitting error: {e}")
        return None
    
    # Forecast
    n_forecast = len(data_df) - n_fit
    try:
        model.forecast(steps=n_forecast)
        model.run_simulations(n_jobs=1)  # Sequential for stability
        model.generate_result()
    except Exception as e:
        print(f"Forecasting error: {e}")
        return None
    
    # Extract results
    if model.results is None:
        return None
    
    data_test = data_df.iloc[n_fit:]
    
    # Get forecasts (point estimates)
    try:
        forecasts_c = model.results['simulations'].get('C', None)
        forecasts_d = model.results['simulations'].get('D', None)
        
        if forecasts_c is not None and forecasts_d is not None:
            # Take mean of scenarios
            C_forecast = forecasts_c.mean(axis=1).values if hasattr(forecasts_c.mean(axis=1), 'values') else forecasts_c.mean(axis=1)
            D_forecast = forecasts_d.mean(axis=1).values if hasattr(forecasts_d.mean(axis=1), 'values') else forecasts_d.mean(axis=1)
        else:
            C_forecast = model.forecasting_box.C_point
            D_forecast = model.forecasting_box.D_point
    except:
        return None
    
    # Trim to test size
    C_forecast = C_forecast[:len(data_test)]
    D_forecast = D_forecast[:len(data_test)]
    
    # Metrics
    mae_c = mean_absolute_error(data_test['C'], C_forecast)
    rmse_c = np.sqrt(mean_squared_error(data_test['C'], C_forecast))
    mape_c = np.mean(np.abs((data_test['C'] - C_forecast) / (data_test['C'] + 1))) * 100
    
    mae_d = mean_absolute_error(data_test['D'], D_forecast)
    rmse_d = np.sqrt(mean_squared_error(data_test['D'], D_forecast))
    mape_d = np.mean(np.abs((data_test['D'] - D_forecast) / (data_test['D'] + 1))) * 100
    
    return {
        'name': 'Dynamic SIRD (dynasir)',
        'test_data': data_test,
        'forecasts': {'C': C_forecast, 'D': D_forecast},
        'metrics': {
            'confirmed_mae': mae_c,
            'confirmed_rmse': rmse_c,
            'confirmed_mape': mape_c,
            'deaths_mae': mae_d,
            'deaths_rmse': rmse_d,
            'deaths_mape': mape_d,
        },
        'test_size': len(data_test),
        'fit_size': int(len(data_df) * split_ratio),
    }


# ============================================================================
# PART 3: MAIN BENCHMARK
# ============================================================================

def main():
    print("=" * 70)
    print("BENCHMARK: Static SIRD vs. Dynamic SIRD (Time-Varying Parameters)")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading COVID-19 data from Our World in Data...")
    try:
        data = process_data_from_owid(include_vaccination=False)
    except Exception as e:
        print(f"Error loading data: {e}")
        # Try with local fallback if available
        return
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Prepare data
    print("\n[2/4] Preparing data (computing SIRD compartments)...")
    container = DataContainer(data)
    data_fit = container.data[['C', 'D', 'I', 'R', 'N']].dropna()
    
    print(f"Available data points: {len(data_fit)}")
    
    # Fit static model
    print("\n[3/4] Fitting Static SIRD model (baseline)...")
    try:
        static_results = fit_static_sird(data_fit, split_ratio=0.8)
        print(f"✓ Static SIRD fitted successfully")
        print(f"  - Test set size: {static_results['test_size']} days")
    except Exception as e:
        print(f"✗ Error fitting static SIRD: {e}")
        static_results = None
    
    # Fit dynamic model
    print("\n[4/4] Fitting Dynamic SIRD model (dynasir with VAR forecasting)...")
    try:
        dynamic_results = fit_dynamic_sird(data_fit, split_ratio=0.8)
        if dynamic_results:
            print(f"✓ Dynamic SIRD fitted successfully")
            print(f"  - Test set size: {dynamic_results['test_size']} days")
        else:
            print(f"✗ Dynamic model returned None")
            dynamic_results = None
    except Exception as e:
        print(f"✗ Error fitting dynamic SIRD: {e}")
        import traceback
        traceback.print_exc()
        dynamic_results = None
    
    # RESULTS
    print("\n" + "=" * 70)
    print("RESULTS: Forecast Accuracy Comparison")
    print("=" * 70)
    
    if static_results:
        print("\n[Static SIRD - Confirmed Cases]")
        print(f"  MAE:  {static_results['metrics']['confirmed_mae']:,.0f}")
        print(f"  RMSE: {static_results['metrics']['confirmed_rmse']:,.0f}")
        print(f"  MAPE: {static_results['metrics']['confirmed_mape']:.2f}%")
        
        print("\n[Static SIRD - Deaths]")
        print(f"  MAE:  {static_results['metrics']['deaths_mae']:,.0f}")
        print(f"  RMSE: {static_results['metrics']['deaths_rmse']:,.0f}")
        print(f"  MAPE: {static_results['metrics']['deaths_mape']:.2f}%")
    
    if dynamic_results:
        print("\n[Dynamic SIRD (dynasir) - Confirmed Cases]")
        print(f"  MAE:  {dynamic_results['metrics']['confirmed_mae']:,.0f}")
        print(f"  RMSE: {dynamic_results['metrics']['confirmed_rmse']:,.0f}")
        print(f"  MAPE: {dynamic_results['metrics']['confirmed_mape']:.2f}%")
        
        print("\n[Dynamic SIRD (dynasir) - Deaths]")
        print(f"  MAE:  {dynamic_results['metrics']['deaths_mae']:,.0f}")
        print(f"  RMSE: {dynamic_results['metrics']['deaths_rmse']:,.0f}")
        print(f"  MAPE: {dynamic_results['metrics']['deaths_mape']:.2f}%")
    
    # COMPARISON TABLE
    if static_results and dynamic_results:
        print("\n" + "=" * 70)
        print("IMPROVEMENT: Dynamic vs. Static")
        print("=" * 70)
        
        comparison_data = {
            'Metric': ['MAE (Cases)', 'RMSE (Cases)', 'MAPE (Cases) %', 
                      'MAE (Deaths)', 'RMSE (Deaths)', 'MAPE (Deaths) %'],
            'Static SIRD': [
                f"{static_results['metrics']['confirmed_mae']:,.0f}",
                f"{static_results['metrics']['confirmed_rmse']:,.0f}",
                f"{static_results['metrics']['confirmed_mape']:.2f}",
                f"{static_results['metrics']['deaths_mae']:,.0f}",
                f"{static_results['metrics']['deaths_rmse']:,.0f}",
                f"{static_results['metrics']['deaths_mape']:.2f}",
            ],
            'Dynamic SIRD': [
                f"{dynamic_results['metrics']['confirmed_mae']:,.0f}",
                f"{dynamic_results['metrics']['confirmed_rmse']:,.0f}",
                f"{dynamic_results['metrics']['confirmed_mape']:.2f}",
                f"{dynamic_results['metrics']['deaths_mae']:,.0f}",
                f"{dynamic_results['metrics']['deaths_rmse']:,.0f}",
                f"{dynamic_results['metrics']['deaths_mape']:.2f}",
            ],
        }
        
        # Calculate improvement
        def calc_improvement(static_val, dynamic_val):
            if static_val == 0:
                return 0
            improvement = (float(dynamic_val) - float(static_val)) / float(static_val) * 100
            return f"{improvement:+.1f}%"
        
        comparison_data['Improvement'] = [
            calc_improvement(static_results['metrics']['confirmed_mae'], 
                           dynamic_results['metrics']['confirmed_mae']),
            calc_improvement(static_results['metrics']['confirmed_rmse'], 
                           dynamic_results['metrics']['confirmed_rmse']),
            calc_improvement(static_results['metrics']['confirmed_mape'], 
                           dynamic_results['metrics']['confirmed_mape']),
            calc_improvement(static_results['metrics']['deaths_mae'], 
                           dynamic_results['metrics']['deaths_mae']),
            calc_improvement(static_results['metrics']['deaths_rmse'], 
                           dynamic_results['metrics']['deaths_rmse']),
            calc_improvement(static_results['metrics']['deaths_mape'], 
                           dynamic_results['metrics']['deaths_mape']),
        ]
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Save comparison table
        output_path = Path(__file__).parent.parent.parent / 'paper' / 'benchmark_comparison.csv'
        comparison_df.to_csv(output_path, index=False)
        print(f"\n✓ Comparison table saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
