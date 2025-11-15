"""
Comparative Analysis for SFC-ML

Generates comparative tables from simulation results.
Requires simV4.py to have been run first.

Usage:
    python3 comparative_analysis.py
    
Outputs:
    - paper/tables/table1_comparative_statics.tex
    - paper/tables/table2_transition_costs.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# Configuration
OUTPUT_DIR = Path("output")
PAPER_DIR = Path("paper")
TABLES_DIR = PAPER_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Scenario metadata
SCENARIO_NAMES = {
    'scenario_1': 'Baseline Capitalism',
    'scenario_2': 'Conflict Capitalism',
    'scenario_2b': 'Managed Conflict (Failed)',
    'scenario_2c': 'Reformed Capitalism',
    'scenario_3': 'Socialist Transition (Success)',
    'scenario_3b': 'Partial Planning (Failed)',
}

def load_scenario_results(scenario_id: str) -> pd.DataFrame:
    """
    Load results CSV for a specific scenario.
    
    Args:
        scenario_id: Identifier like 'scenario_1', 'scenario_2', etc.
    
    Returns:
        DataFrame with 15 rows (years) and columns for all model variables
    
    Raises:
        FileNotFoundError: If simV4.py hasn't been run yet
    """
    csv_path = OUTPUT_DIR / f"{scenario_id}_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found: {csv_path}")
    return pd.read_csv(csv_path)

def calculate_steady_state_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate steady-state metrics from final 3 years.
    
    Args:
        df: Simulation results (15 years)
    
    Returns:
        Dictionary with outcome metrics (output, profit, debt, wages, productivity, external)
    """
    # Use final 3 years as steady state
    steady_period = df.iloc[-3:]
    final_year = df.iloc[-1]
    
    # Calculate wage share (need to derive from model output)
    # W/Y is not directly in output, but we can approximate from realWageIndex and employment
    
    metrics = {
        # Growth & Stability
        'Output (Y*)': steady_period['YOutput'].mean(),
        'Output Growth (%)': ((df.iloc[-1]['YOutput'] / df.iloc[0]['YOutput']) ** (1/14) - 1) * 100,
        
        # Profitability
        'Profit Rate (Π*)': steady_period['PiT'].mean(),
        'Years with Π<0': (df['PiT'] < 0).sum(),
        
        # Financial Fragility
        'Final Debt (L)': final_year['LFirmLoan'],
        'Max Debt/K': (df['LFirmLoan'] / df['YOutput']).max() if df['YOutput'].min() > 0 else np.inf,
        
        # Distribution
        'Real Wage Index': steady_period['realWageIndex'].mean(),
        'Wage Growth (%)': ((df.iloc[-1]['realWageIndex'] / df.iloc[0]['realWageIndex']) ** (1/14) - 1) * 100,
        
        # External Balance
        'Net Foreign Assets': final_year['netForeignAssets'],
        'Cumulative Capital Flight': df['capitalFlight'].sum(),
        'Trade Balance': steady_period['tradeBalance'].mean(),
        
        # Productivity & Efficiency
        'Labor Productivity': final_year['LaborProductivity'],
        'Productivity Growth (%)': ((final_year['LaborProductivity'] / df.iloc[0]['LaborProductivity']) ** (1/14) - 1) * 100,
    }
    
    return metrics

def calculate_transition_costs(df: pd.DataFrame, baseline_df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate transition costs relative to baseline scenario.
    
    Metrics:
    - Cumulative output loss/gain
    - Peak crisis depth
    - Time to recovery
    """
    # Cumulative output gap
    output_gap = df['YOutput'] - baseline_df['YOutput']
    cumulative_gap = output_gap.sum()
    
    # Crisis metrics
    crisis_years = df[df['PiT'] < 0]
    peak_crisis = crisis_years['YOutput'].min() if len(crisis_years) > 0 else df['YOutput'].min()
    
    # Recovery time (when output exceeds baseline)
    recovery_year = None
    for idx, row in df.iterrows():
        if row['YOutput'] > baseline_df.iloc[idx]['YOutput']:
            recovery_year = row['year']
            break
    
    return {
        'Cumulative Output Gap': cumulative_gap,
        'Peak Crisis Output': peak_crisis,
        'Recovery Year': recovery_year if recovery_year else 'Never',
        'Crisis Duration (years)': len(crisis_years),
    }

def create_main_comparative_table() -> pd.DataFrame:
    """
    Generate Table 1: Comparative Statics Across Scenarios
    
    This is the CORE EMPIRICAL TABLE for the paper, answering:
    - Does socialist planning outperform capitalism? (compare scenarios 3 vs 1/2)
    - What happens with partial reforms? (scenario 2c vs 2b)
    - Can planning fail without full coordination? (scenario 3b)
    
    Process:
    1. Load results for 5 key scenarios (excludes scenario_2 for space)
    2. Calculate 13 metrics per scenario (output, profits, debt, etc.)
    3. Arrange as rows (scenarios) × columns (metrics)
    4. Format for LaTeX export
    
    Returns:
        DataFrame ready for publication (Table 1 in paper)
    """
    scenarios = ['scenario_1', 'scenario_2b', 'scenario_2c', 'scenario_3', 'scenario_3b']
    results = []
    
    # Loop through each scenario and calculate its steady-state outcomes
    for scenario_id in scenarios:
        df = load_scenario_results(scenario_id)  # Load the 15-year time series
        metrics = calculate_steady_state_metrics(df)  # Compute final-state metrics
        metrics['Scenario'] = SCENARIO_NAMES[scenario_id]  # Add human-readable name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results)
    
    # Reorder columns for readability
    column_order = [
        'Scenario',
        'Output (Y*)',
        'Output Growth (%)',
        'Profit Rate (Π*)',
        'Years with Π<0',
        'Final Debt (L)',
        'Real Wage Index',
        'Wage Growth (%)',
        'Labor Productivity',
        'Productivity Growth (%)',
        'Net Foreign Assets',
        'Cumulative Capital Flight',
    ]
    
    return comparison_df[column_order]

def create_transition_costs_table() -> pd.DataFrame:
    """
    Generate Table 2: Transition Costs
    
    Compares reform scenarios to baseline.
    """
    baseline_df = load_scenario_results('scenario_1')
    transition_scenarios = ['scenario_2b', 'scenario_2c', 'scenario_3', 'scenario_3b']
    
    results = []
    for scenario_id in transition_scenarios:
        df = load_scenario_results(scenario_id)
        costs = calculate_transition_costs(df, baseline_df)
        costs['Scenario'] = SCENARIO_NAMES[scenario_id]
        results.append(costs)
    
    return pd.DataFrame(results)

def export_latex_tables():
    """Export main comparative table to LaTeX format for publication."""
    main_table = create_main_comparative_table()
    
    # Try LaTeX export, fall back to CSV if jinja2 not available
    try:
        latex_main = main_table.to_latex(
            index=False,
            float_format="%.2f",
            caption="Comparative Statics Across Scenarios",
            label="tab:comparative_statics",
        )
        
        latex_path = PAPER_DIR / 'tables' / 'table1_comparative_statics.tex'
        latex_path.write_text(latex_main)
        print(f"\n✓ Exported LaTeX Table 1 to {latex_path}")
    except ImportError as e:
        print(f"\n⚠ LaTeX export requires jinja2: {e}")
        print("  Falling back to CSV export...")
        csv_path = PAPER_DIR / 'tables' / 'table1_comparative_statics.csv'
        main_table.to_csv(csv_path, index=False)
        print(f"✓ Exported Table 1 (CSV) to {csv_path}")
        print("  To enable LaTeX: pip install jinja2")
    
    # Export transition costs table
    trans_table = create_transition_costs_table()
    csv_path = PAPER_DIR / 'tables' / 'table2_transition_costs.csv'
    trans_table.to_csv(csv_path, index=False)
    print(f"✓ Exported Table 2 (CSV) to {csv_path}")

def print_summary_statistics():
    """Print human-readable summary for interpretation."""
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("="*80 + "\n")
    
    main_table = create_main_comparative_table()
    print("TABLE 1: Comparative Statics Across Scenarios")
    print("-" * 80)
    print(main_table.to_string(index=False))
    
    print("\n\n")
    
    transition_table = create_transition_costs_table()
    print("TABLE 2: Transition Costs (vs Baseline)")
    print("-" * 80)
    print(transition_table.to_string(index=False))
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Automated interpretation
    best_output = main_table.loc[main_table['Output (Y*)'].idxmax()]
    best_productivity = main_table.loc[main_table['Productivity Growth (%)'].idxmax()]
    least_fragile = main_table.loc[main_table['Final Debt (L)'].idxmin()]
    
    print(f"\n✓ Highest steady-state output: {best_output['Scenario']} (Y* = {best_output['Output (Y*)']:.1f})")
    print(f"✓ Fastest productivity growth: {best_productivity['Scenario']} ({best_productivity['Productivity Growth (%)']:.1f}% p.a.)")
    print(f"✓ Lowest financial fragility: {least_fragile['Scenario']} (L = {least_fragile['Final Debt (L)']:.1f})")
    
    # Crisis comparison
    crisis_free = main_table[main_table['Years with Π<0'] == 0]
    if len(crisis_free) > 0:
        print(f"\n✓ Scenarios without profit crises: {', '.join(crisis_free['Scenario'].tolist())}")
    
    print("\n")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
# This block only runs when called directly (not when imported as module)
# Usage: python3 comparative_analysis.py

if __name__ == "__main__":
    print("Generating comparative analysis tables for publication...")
    
    # SAFETY CHECK: Verify that simV4.py has been run
    # Without this, we have no data to analyze
    if not OUTPUT_DIR.exists():
        print(f"ERROR: {OUTPUT_DIR}/ not found. Run simV4.py first to generate results.")
        exit(1)
    
    # STEP 1: Generate and display tables in console
    # This shows the user what metrics were calculated
    print_summary_statistics()
    
    # STEP 2: Export tables for paper
    # Creates LaTeX (if jinja2 available) and CSV backups
    export_latex_tables()
    
    # STEP 3: Save additional CSV copies for convenience
    # Some users prefer CSV for data exploration in Excel/Python
    main_table = create_main_comparative_table()
    main_table.to_csv(TABLES_DIR / "table1_comparative_statics.csv", index=False)
    
    transition_table = create_transition_costs_table()
    transition_table.to_csv(TABLES_DIR / "table2_transition_costs.csv", index=False)
    
    print(f"✓ CSV tables exported to {TABLES_DIR}/")
    print("\nDone! Tables ready for paper.")

