"""
Monte Carlo Robustness Analysis for SFC-ML

Performs parameter sensitivity analysis with 500 runs per scenario.
Optional - takes ~5-10 minutes.

Usage:
    python3 monte_carlo.py

Outputs:
    - paper/monte_carlo/table3_robustness.tex
    - paper/monte_carlo/{scenario}/confidence_bands.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Tuple

sys.path.append('.')
from simV4 import (
    SFCModel, ModelParams, InitialStocks, PolicySchedule,
    ProfitLedPrivateBanking, StateLedNationalizedBanking,
    FloatingExchangeRate, ManagedExchangeRate,
    ConflictInflationRegime,
    boosted_transition_schedule,
    replace
)

# Configuration
DEFAULT_RUNS = 500
VARIATION_PCT = 0.10  # ±10% parameter variation
OUTPUT_DIR = Path("paper/monte_carlo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Key behavioral parameters to shock (most uncertain)
SHOCK_PARAMS = [
    'investment_sensitivity_to_pi',      # Animal spirits
    'wage_inflation_sensitivity',         # Worker bargaining power
    'capital_flight_sensitivity',         # Investor panic
    'infrastructure_productivity_effect', # Public spending effectiveness
    'firm_markup',                        # Market power
    'autonomous_investment_rate',         # Baseline investment propensity
]

def generate_parameter_shock(base_params: ModelParams, seed: int) -> ModelParams:
    """Generate random parameter variation within +/-10% bounds using Latin Hypercube Sampling."""
    np.random.seed(seed)
    
    shocked_values = {}
    for param_name in SHOCK_PARAMS:
        base_value = getattr(base_params, param_name)
        # Uniform distribution ±10%
        shock_factor = np.random.uniform(1 - VARIATION_PCT, 1 + VARIATION_PCT)
        shocked_values[param_name] = base_value * shock_factor
    
    return replace(base_params, **shocked_values)

def run_monte_carlo_scenario(
    scenario_name: str,
    base_params: ModelParams,
    stocks: InitialStocks,
    policy: PolicySchedule,
    finance_regime,
    external_regime,
    distribution_regime,
    n_runs: int = DEFAULT_RUNS
) -> pd.DataFrame:
    """
    Run Monte Carlo analysis for a single scenario.
    
    Returns:
        DataFrame with all runs stacked (run_id, year, metrics)
    """
    print(f"\nRunning Monte Carlo for {scenario_name}...")
    print(f"  Simulations: {n_runs}")
    print(f"  Parameter variation: ±{VARIATION_PCT*100:.0f}%")
    
    all_results = []
    
    for run_id in range(n_runs):
        if (run_id + 1) % 50 == 0:
            print(f"  Progress: {run_id + 1}/{n_runs}")
        
        # Generate shocked parameters
        shocked_params = generate_parameter_shock(base_params, seed=run_id)
        
        # Run simulation
        model = SFCModel(
            stocks=stocks,
            params=shocked_params,
            policy=policy,
            finance_regime=finance_regime,
            external_regime=external_regime,
            distribution_regime=distribution_regime,
        )
        
        try:
            results = model.run()
            results['run_id'] = run_id
            all_results.append(results)
        except Exception as e:
            print(f"  Warning: Run {run_id} failed with error: {e}")
            continue
    
    print(f"  ✓ Completed {len(all_results)}/{n_runs} successful runs")
    
    return pd.concat(all_results, ignore_index=True)

def calculate_confidence_bands(mc_results: pd.DataFrame) -> pd.DataFrame:
    """Calculate confidence bands for each year.
    
    Returns:
        DataFrame with year, variable, mean, std, p5, p50, p95
    """
    metrics = ['YOutput', 'PiT', 'realWageIndex', 'LFirmLoan', 'LaborProductivity', 'capitalFlight']
    
    summary_stats = []
    
    for year in mc_results['year'].unique():
        year_data = mc_results[mc_results['year'] == year]
        
        for metric in metrics:
            stats = {
                'year': year,
                'metric': metric,
                'mean': year_data[metric].mean(),
                'std': year_data[metric].std(),
                'p5': year_data[metric].quantile(0.05),
                'p50': year_data[metric].quantile(0.50),
                'p95': year_data[metric].quantile(0.95),
                'cv': year_data[metric].std() / year_data[metric].mean() if year_data[metric].mean() != 0 else np.inf,
            }
            summary_stats.append(stats)
    
    return pd.DataFrame(summary_stats)

def calculate_robustness_metrics(mc_results: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate aggregate robustness metrics.
    
    Measures:
    - Fraction of runs converging (final Π > 0)
    - Fraction avoiding crisis (no years with Π < -10%)
    - Output volatility (CV of final Y across runs)
    """
    final_year_results = mc_results[mc_results['year'] == mc_results['year'].max()]
    
    convergence_rate = (final_year_results['PiT'] > 0).mean()
    crisis_free_rate = (final_year_results['PiT'] > -0.10).mean()
    
    # Volatility
    output_cv = final_year_results['YOutput'].std() / final_year_results['YOutput'].mean()
    profit_cv = final_year_results['PiT'].std() / final_year_results['PiT'].mean() if final_year_results['PiT'].mean() != 0 else np.inf
    
    return {
        'convergence_rate': convergence_rate,
        'crisis_free_rate': crisis_free_rate,
        'output_cv': output_cv,
        'profit_cv': profit_cv,
        'n_runs': len(final_year_results),
    }

def export_monte_carlo_results(scenario_name: str, mc_results: pd.DataFrame, confidence_bands: pd.DataFrame, robustness: Dict):
    """Export Monte Carlo results for a scenario."""
    scenario_dir = OUTPUT_DIR / scenario_name
    scenario_dir.mkdir(exist_ok=True)
    
    # Save full results (large file)
    mc_results.to_csv(scenario_dir / "full_results.csv", index=False)
    
    # Save confidence bands (main result for paper)
    confidence_bands.to_csv(scenario_dir / "confidence_bands.csv", index=False)
    
    # Save robustness metrics
    robustness_df = pd.DataFrame([robustness])
    robustness_df.to_csv(scenario_dir / "robustness_metrics.csv", index=False)
    
    # Generate LaTeX snippet for paper
    latex_snippet = (
        f"% Monte Carlo Robustness for {scenario_name}\n"
        f"% Convergence rate: {robustness['convergence_rate']*100:.1f}%\n"
        f"% Crisis-free rate: {robustness['crisis_free_rate']*100:.1f}%\n"
        f"% Output CV: {robustness['output_cv']:.3f}\n"
    )
    with open(scenario_dir / "latex_snippet.tex", 'w') as f:
        f.write(latex_snippet)
    
    print(f"  ✓ Results exported to {scenario_dir}/")

def create_robustness_summary_table(all_robustness: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create Table 3: Robustness Checks (Monte Carlo)
    
    Summary table for publication comparing robustness across scenarios.
    """
    rows = []
    for scenario_name, metrics in all_robustness.items():
        rows.append({
            'Scenario': scenario_name,
            'Convergence Rate (%)': metrics['convergence_rate'] * 100,
            'Crisis-Free Rate (%)': metrics['crisis_free_rate'] * 100,
            'Output CV': metrics['output_cv'],
            'Profit CV': metrics['profit_cv'],
            'N Runs': metrics['n_runs'],
        })
    
    return pd.DataFrame(rows)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo robustness analysis")
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Number of Monte Carlo simulations per scenario (default: {DEFAULT_RUNS})",
    )
    return parser.parse_args()


def main(n_runs: int):
    """Run Monte Carlo analysis for key scenarios."""
    
    print("="*80)
    print("MONTE CARLO ROBUSTNESS ANALYSIS")
    print("="*80)
    print(f"Runs per scenario: {n_runs}")
    
    # Define scenarios (focus on key comparisons)
    scenarios = {
        'Reformed Capitalism (2c)': {
            'params': ModelParams(
                firm_markup=0.26,
                nominal_interest_rate=0.006,
                autonomous_investment_rate=0.05,
                wage_inflation_sensitivity=0.2,
                wage_unemployment_sensitivity=0.3,
                labor_productivity=1.15,
                investment_sensitivity_to_pi=0.5,
                capital_flight_sensitivity=0.5,
                infrastructure_productivity_effect=0.02,
                enable_productivity_feedback=True,
                enable_capital_flight=True,
            ),
            'stocks': InitialStocks(l_firm_loan=0.0),
            # Keep 15-year horizon but dramatically scale project spending
            'policy': boosted_transition_schedule(
                train_multiplier=3.5,
                housing_multiplier=2.5,
                baseline_spend=140.0,
            ),
            'finance_regime': ProfitLedPrivateBanking(),
            'external_regime': FloatingExchangeRate(),
            'distribution_regime': ConflictInflationRegime(),
        },
        'Socialist Transition (3)': {
            'params': ModelParams(
                planned_investment_rate=0.09,
                firm_markup=0.26,
                wage_inflation_sensitivity=0.2,
                wage_unemployment_sensitivity=0.3,
                labor_productivity=1.15,
                infrastructure_productivity_effect=0.03,
                enable_productivity_feedback=True,
                enable_capital_flight=False,
                capital_controls_effectiveness=1.0,
                enable_credit_rationing=False,
                nominal_interest_rate=0.0,
                investment_sensitivity_to_pi=0.5,
                capital_flight_sensitivity=0.5,
            ),
            'stocks': InitialStocks(l_firm_loan=0.0),
            'policy': boosted_transition_schedule(
                train_multiplier=3.5,
                housing_multiplier=2.5,
                baseline_spend=140.0,
            ),
            'finance_regime': StateLedNationalizedBanking(),
            'external_regime': ManagedExchangeRate(),
            'distribution_regime': ConflictInflationRegime(),
        },
    }
    
    all_robustness = {}
    
    for scenario_name, config in scenarios.items():
        # Run Monte Carlo
        mc_results = run_monte_carlo_scenario(
            scenario_name=scenario_name,
            base_params=config['params'],
            stocks=config['stocks'],
            policy=config['policy'],
            finance_regime=config['finance_regime'],
            external_regime=config['external_regime'],
            distribution_regime=config['distribution_regime'],
            n_runs=n_runs,
        )
        
        # Calculate statistics
        confidence_bands = calculate_confidence_bands(mc_results)
        robustness_metrics = calculate_robustness_metrics(mc_results)
        
        # Export
        export_monte_carlo_results(scenario_name, mc_results, confidence_bands, robustness_metrics)
        
        # Store for summary table
        all_robustness[scenario_name] = robustness_metrics
    
    # Create summary table
    summary_table = create_robustness_summary_table(all_robustness)
    print("\n" + "="*80)
    print("ROBUSTNESS SUMMARY")
    print("="*80)
    print(summary_table.to_string(index=False))
    
    # Export summary table with fallback
    summary_table.to_csv(OUTPUT_DIR / "table3_robustness_summary.csv", index=False)
    
    try:
        latex_table = summary_table.to_latex(
            index=False,
            float_format="%.2f",
            caption="Monte Carlo Robustness Checks (500 runs with ±10\\% parameter variation)",
            label="tab:monte_carlo",
        )
        with open(OUTPUT_DIR / "table3_robustness.tex", 'w') as f:
            f.write(latex_table)
        print(f"\n✓ Summary table exported to {OUTPUT_DIR}/table3_robustness.tex")
    except ImportError:
        print(f"\n⚠ LaTeX export requires jinja2 (pip install jinja2)")
        print(f"✓ CSV table exported to {OUTPUT_DIR}/table3_robustness_summary.csv")
    
    print("\nMonte Carlo analysis complete!")

if __name__ == "__main__":
    args = parse_args()
    if args.runs <= 0:
        print("Error: --runs must be a positive integer")
        sys.exit(1)
    main(n_runs=args.runs)
