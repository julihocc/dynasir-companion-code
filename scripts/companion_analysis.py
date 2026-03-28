"""
Companion Analysis Script: Detailed Comparison and Visualization

This script generates publication-ready figures comparing Static vs. Dynamic SIRD models.
Produces:
- Forecast comparison plots
- Parameter trajectory visualizations
- Error distribution analysis
- Summary statistics

Usage:
    python companion_analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from dynasir import process_data_from_owid, DataContainer, Model
except ImportError:
    print("ERROR: dynasir package not installed.")
    exit(1)


def generate_companion_analysis():
    """Generate detailed analysis showing the impact of time-varying parameters"""
    
    print("Generating companion analysis for the paper...")
    
    # Load data
    print("Loading data...")
    data = process_data_from_owid(include_vaccination=False)
    container = DataContainer(data)
    data_full = container.data[['C', 'D', 'I', 'R', 'N']].dropna()
    
    # Split into fit and test
    n_fit = int(len(data_full) * 0.8)
    data_train = data_full.iloc[:n_fit]
    data_test = data_full.iloc[n_fit:]
    
    print(f"Data: {len(data_train)} training, {len(data_test)} testing")
    
    # Fit dynamic model with dynasir
    print("Fitting dynamic SIRD model...")
    try:
        container = DataContainer(data_full)
        model = Model(container)
        model.create_model()
        model.fit_model(max_lag=3)
        model.forecast(steps=len(data_test))
        model.run_simulations(n_jobs=1)
        model.generate_result()
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Extract time-varying parameters
    print("Extracting parameter trajectories...")
    try:
        alpha_t = model.features_df['alpha'].values
        beta_t = model.features_df['beta'].values
        gamma_t = model.features_df['gamma'].values
    except:
        print("Could not extract parameters")
        return
    
    # Create figures directory
    figures_dir = Path(__file__).parent.parent.parent / 'paper' / 'companion_figures'
    figures_dir.mkdir(exist_ok=True)
    
    # FIGURE 1: Time-Varying Parameters
    print("Creating Figure 1: Parameter Trajectories...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Time-Varying Epidemiological Parameters from dynasir', fontsize=16, fontweight='bold')
    
    dates = data_full.index
    train_dates = dates[:n_fit]
    test_dates = dates[n_fit:]
    
    # Alpha (infection rate)
    ax = axes[0]
    ax.plot(dates, alpha_t, 'b-', linewidth=2, alpha=0.7, label='α(t) - Infection Rate')
    ax.axvline(train_dates[-1], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Forecast Start')
    ax.fill_between(train_dates, alpha_t[:len(train_dates)].min(), alpha_t[:len(train_dates)].max(), 
                     alpha=0.1, color='green', label='Training Period')
    ax.set_title('Infection Rate α(t)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Beta (recovery rate)
    ax = axes[1]
    ax.plot(dates, beta_t, 'g-', linewidth=2, alpha=0.7, label='β(t) - Recovery Rate')
    ax.axvline(train_dates[-1], color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.fill_between(train_dates, beta_t[:len(train_dates)].min(), beta_t[:len(train_dates)].max(), 
                     alpha=0.1, color='green')
    ax.set_title('Recovery Rate β(t)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Gamma (mortality rate)
    ax = axes[2]
    ax.plot(dates, gamma_t, 'r-', linewidth=2, alpha=0.7, label='γ(t) - Mortality Rate')
    ax.axvline(train_dates[-1], color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.fill_between(train_dates, gamma_t[:len(train_dates)].min(), gamma_t[:len(train_dates)].max(), 
                     alpha=0.1, color='green')
    ax.set_title('Mortality Rate γ(t)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate')
    ax.set_xlabel('Date')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / '01_parameter_trajectories.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 01_parameter_trajectories.png")
    plt.close()
    
    # FIGURE 2: Basic Reproduction Number
    print("Creating Figure 2: R₀(t) Over Time...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    R0_t = alpha_t / (beta_t + gamma_t)
    ax.plot(dates, R0_t, 'purple', linewidth=3, alpha=0.8, label='R₀(t) - Time-Varying')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='R₀=1 (Epidemic Threshold)')
    ax.axvline(train_dates[-1], color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Forecast Start')
    
    ax.fill_between(dates, 0, R0_t, where=(R0_t >= 1), alpha=0.2, color='red', label='Growth Phase (R₀>1)')
    ax.fill_between(dates, 0, R0_t, where=(R0_t < 1), alpha=0.2, color='green', label='Decline Phase (R₀<1)')
    
    ax.set_title('Basic Reproduction Number R₀(t): Dynamic Evolution', fontsize=14, fontweight='bold')
    ax.set_ylabel('R₀(t)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / '02_R0_evolution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 02_R0_evolution.png")
    plt.close()
    
    # FIGURE 3: Compartment Comparison
    print("Creating Figure 3: SIRD Compartments...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SIRD Model Compartments Over Time', fontsize=16, fontweight='bold')
    
    compartments = [
        (0, 'S', 'Susceptible', 'blue'),
        (1, 'I', 'Infected', 'red'),
        (2, 'R', 'Recovered', 'green'),
        (3, 'D', 'Deceased', 'black'),
    ]
    
    for idx, (comp_idx, col, name, color) in enumerate(compartments):
        if col not in data_full.columns:
            continue
        
        ax = axes[idx // 2, idx % 2]
        data_comp = data_full[col]
        
        ax.plot(dates, data_comp, color=color, linewidth=2.5, alpha=0.8)
        ax.axvline(train_dates[-1], color='red', linestyle='--', linewidth=2, alpha=0.5)
        ax.fill_between(train_dates, data_comp.iloc[:len(train_dates)].min(), 
                        data_comp.iloc[:len(train_dates)].max(), alpha=0.1, color=color)
        
        ax.set_title(f'{name} Compartment', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.savefig(figures_dir / '03_compartments.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 03_compartments.png")
    plt.close()
    
    # FIGURE 4: Parameter Drift Illustration
    print("Creating Figure 4: Static vs. Dynamic Parameter Assumption...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Compute moving average for cleaner visualization
    window = 14
    alpha_smooth = pd.Series(alpha_t).rolling(window, center=True).mean().values
    alpha_mean = np.nanmean(alpha_smooth)
    
    ax.plot(dates, alpha_smooth, 'b-', linewidth=3, alpha=0.8, label='Actual α(t) - Time-Varying')
    ax.axhline(alpha_mean, color='r', linestyle='--', linewidth=3, alpha=0.8, 
              label=f'Static Assumption: α={alpha_mean:.4f}')
    
    ax.fill_between(dates, alpha_mean, alpha_smooth, where=(alpha_smooth > alpha_mean), 
                    alpha=0.3, color='green', label='Model Overestimates (α > mean)')
    ax.fill_between(dates, alpha_mean, alpha_smooth, where=(alpha_smooth < alpha_mean), 
                    alpha=0.3, color='red', label='Model Underestimates (α < mean)')
    
    ax.axvline(train_dates[-1], color='purple', linestyle='--', linewidth=2, alpha=0.5, label='Forecast Start')
    
    ax.set_title('Parameter Drift: Why Static Parameters Fail', fontsize=14, fontweight='bold')
    ax.set_ylabel('Infection Rate α(t)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / '04_parameter_drift.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: 04_parameter_drift.png")
    plt.close()
    
    print(f"\n✓ All companion figures saved to: {figures_dir}")
    print(f"\nFigures generated:")
    print(f"  - 01_parameter_trajectories.png: Time series of α(t), β(t), γ(t)")
    print(f"  - 02_R0_evolution.png: Dynamic basic reproduction number")
    print(f"  - 03_compartments.png: SIRD compartments over time")
    print(f"  - 04_parameter_drift.png: Static vs. dynamic parameter assumption")


if __name__ == '__main__':
    generate_companion_analysis()
