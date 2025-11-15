"""
Phase Diagram Generation for SFC-ML

Generates state-space plots showing scenario trajectories.
Requires simV4.py to have been run first.

Usage:
    python3 phase_diagrams.py
    
Outputs:
    - paper/figures/figure2_phase_diagram.png
    - paper/figures/figure3_output_dynamics.png
    - paper/figures/figure4_productivity_growth.png
    - paper/figures/figure5_external_balance.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

# Configuration
OUTPUT_DIR = Path("output")
FIGURES_DIR = Path("paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Scenario metadata
SCENARIO_INFO = {
    'scenario_1': {'name': 'Baseline', 'color': 'gray', 'linestyle': '-'},
    'scenario_2b': {'name': 'Managed Conflict (Failed)', 'color': 'red', 'linestyle': '--'},
    'scenario_2c': {'name': 'Reformed Capitalism', 'color': 'blue', 'linestyle': '-'},
    'scenario_3': {'name': 'Socialist Success', 'color': 'green', 'linestyle': '-'},
    'scenario_3b': {'name': 'Partial Planning (Failed)', 'color': 'orange', 'linestyle': '--'},
}

def load_scenario(scenario_id: str) -> pd.DataFrame:
    """
    Load scenario results from CSV file.
    
    Args:
        scenario_id: Identifier like 'scenario_1', 'scenario_2c', etc.
    
    Returns:
        DataFrame with 15 years × all model variables
    
    Raises:
        FileNotFoundError: If simV4.py hasn't been run yet
    """
    csv_path = OUTPUT_DIR / f"{scenario_id}_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Results not found: {csv_path}. Run simV4.py first.")
    return pd.read_csv(csv_path)

def create_profit_debt_phase_diagram(scenarios: List[str], save_path: Path):
    """
    Figure 2: Phase Diagram - Profit Rate vs Debt Level
    
    Shows Minsky-style financial fragility dynamics.
    X-axis: Absolute debt level (financial exposure)
    Y-axis: Profitability (ability to service debt)
    
    Interpretation:
    - Top-left: Low debt + high profit = Robust finance
    - Bottom-right: High debt + low profit = Ponzi finance (crisis!)
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scenario_id in scenarios:
        df = load_scenario(scenario_id)
        info = SCENARIO_INFO[scenario_id]
        
        # Use absolute debt level (clearer than ratio)
        debt = df['LFirmLoan']
        profit = df['PiT']
        
        # Plot trajectory
        ax.plot(
            debt, 
            profit,
            label=info['name'],
            color=info['color'],
            linestyle=info['linestyle'],
            linewidth=2.5,
            alpha=0.8,
        )
        
        # Mark start (circle) and end (square)
        ax.scatter(debt.iloc[0], profit.iloc[0], 
                  color=info['color'], marker='o', s=150, zorder=5, edgecolors='black', linewidths=1.5)
        ax.scatter(debt.iloc[-1], profit.iloc[-1], 
                  color=info['color'], marker='s', s=150, zorder=5, edgecolors='black', linewidths=1.5)
        
        # Add year labels at key points
        for i in [0, 7, 14]:
            if i < len(df):
                ax.annotate(f'Y{int(df.iloc[i]["year"])}', 
                           (debt.iloc[i], profit.iloc[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color=info['color'], weight='bold')
    
    # Add crisis zones
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Crisis Threshold (Π=0)')
    ax.axhline(y=0.15, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target Profit Rate')
    
    # Annotate quadrants
    ax.text(0.05, 0.95, 'ROBUST FINANCE\n(Low debt, High profit)', 
            transform=ax.transAxes, fontsize=11, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.95, 0.05, 'PONZI FINANCE\n(High debt, Low profit)', 
            transform=ax.transAxes, fontsize=11, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    ax.set_xlabel('Firm Debt Level (L)', fontsize=14, weight='bold')
    ax.set_ylabel('Profit Rate (Π)', fontsize=14, weight='bold')
    ax.set_title('Minsky Dynamics: Profitability vs Financial Exposure\n(○ = Start, □ = End)', 
                 fontsize=16, weight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved phase diagram: {save_path}")
    plt.close()

def create_output_capital_phase_diagram(scenarios: List[str], save_path: Path):
    """
    Figure 3: Wage-Profit Trade-off (Distributive Conflict)
    
    Shows the classic Marxian/Kaleckian trade-off between wages and profits.
    X-axis: Real wage index (worker power)
    Y-axis: Profit rate (capitalist income)
    
    Interpretation:
    - Downward slope expected (conflict over distribution)
    - Socialist scenarios might break this trade-off via productivity growth
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scenario_id in scenarios:
        df = load_scenario(scenario_id)
        info = SCENARIO_INFO[scenario_id]
        
        wages = df['realWageIndex']
        profits = df['PiT']
        
        # Plot trajectory
        ax.plot(
            wages,
            profits,
            label=info['name'],
            color=info['color'],
            linestyle=info['linestyle'],
            linewidth=2.5,
            alpha=0.8,
        )
        
        # Mark start and end
        ax.scatter(wages.iloc[0], profits.iloc[0], 
                  color=info['color'], marker='o', s=150, zorder=5, edgecolors='black', linewidths=1.5)
        ax.scatter(wages.iloc[-1], profits.iloc[-1], 
                  color=info['color'], marker='s', s=150, zorder=5, edgecolors='black', linewidths=1.5)
        
        # Label final point
        ax.annotate(f'{info["name"][:15]}', 
                   (wages.iloc[-1], profits.iloc[-1]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=9, color=info['color'], weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=info['color'], alpha=0.7))
    
    # Reference lines
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Zero Profit')
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial Real Wage')
    
    ax.set_xlabel('Real Wage Index', fontsize=14, weight='bold')
    ax.set_ylabel('Profit Rate (Π)', fontsize=14, weight='bold')
    ax.set_title('Distributive Conflict: Wage-Profit Trade-off\n(○ = Start, □ = End)', 
                 fontsize=16, weight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved output trajectory: {save_path}")
    plt.close()

def create_productivity_growth_phase(scenarios: List[str], save_path: Path):
    """
    Figure 4: Productivity vs Output (Growth Dynamics)
    
    Shows whether productivity growth translates into output growth.
    X-axis: Labor productivity (technical progress)
    Y-axis: Output (material prosperity)
    
    Interpretation:
    - Upward slope: Productivity gains realize as growth
    - Flat trajectory: Productivity paradox (gains don't translate)
    - Socialist scenarios should show steeper slope
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scenario_id in scenarios:
        df = load_scenario(scenario_id)
        info = SCENARIO_INFO[scenario_id]
        
        productivity = df['LaborProductivity']
        output = df['YOutput']
        
        ax.plot(
            productivity,
            output,
            label=info['name'],
            color=info['color'],
            linestyle=info['linestyle'],
            linewidth=2.5,
            alpha=0.8,
            marker='o',
            markersize=5,
        )
        
        # Mark start and end
        ax.scatter(productivity.iloc[0], output.iloc[0], 
                  color=info['color'], marker='o', s=150, zorder=5, edgecolors='black', linewidths=1.5)
        ax.scatter(productivity.iloc[-1], output.iloc[-1], 
                  color=info['color'], marker='s', s=150, zorder=5, edgecolors='black', linewidths=1.5)
    
    ax.set_xlabel('Labor Productivity', fontsize=14, weight='bold')
    ax.set_ylabel('Output Level (Y)', fontsize=14, weight='bold')
    ax.set_title('Productivity-Led Growth: Does Technical Progress → Prosperity?\n(○ = Start, □ = End)', 
                 fontsize=16, weight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved productivity phase diagram: {save_path}")
    plt.close()

def create_external_balance_phase(scenarios: List[str], save_path: Path):
    """
    Figure 5: External Constraint (Capital Flight vs Exchange Rate)
    
    Shows balance-of-payments dynamics and capital flight.
    X-axis: Exchange rate (external competitiveness)
    Y-axis: Net foreign assets (external position)
    
    Interpretation:
    - Quadrant analysis of external sustainability
    - Capital flight episodes visible as sudden drops
    - Capital controls effectiveness shown by trajectory stability
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for scenario_id in scenarios:
        df = load_scenario(scenario_id)
        info = SCENARIO_INFO[scenario_id]
        
        exchange_rate = df['exchangeRate']
        net_foreign_assets = df['netForeignAssets']
        
        ax.plot(
            exchange_rate,
            net_foreign_assets,
            label=info['name'],
            color=info['color'],
            linestyle=info['linestyle'],
            linewidth=2.5,
            alpha=0.8,
        )
        
        # Mark start and end
        ax.scatter(exchange_rate.iloc[0], net_foreign_assets.iloc[0], 
                  color=info['color'], marker='o', s=150, zorder=5, edgecolors='black', linewidths=1.5)
        ax.scatter(exchange_rate.iloc[-1], net_foreign_assets.iloc[-1], 
                  color=info['color'], marker='s', s=150, zorder=5, edgecolors='black', linewidths=1.5)
        
        # Highlight capital flight episodes (large drops in NFA)
        capital_flight = df['capitalFlight']
        for i in range(1, len(df)):
            if capital_flight.iloc[i] > 50:  # Significant flight
                ax.scatter(exchange_rate.iloc[i], net_foreign_assets.iloc[i],
                          color='red', marker='x', s=100, zorder=6, linewidths=2)
    
    # Reference lines
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero NFA')
    ax.axvline(x=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Initial E')
    
    # Annotate quadrants
    ax.text(0.05, 0.95, 'SURPLUS\n(Strong E, +NFA)', 
            transform=ax.transAxes, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.95, 0.05, 'CRISIS\n(Weak E, -NFA)', 
            transform=ax.transAxes, fontsize=10, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    ax.set_xlabel('Exchange Rate (E)', fontsize=14, weight='bold')
    ax.set_ylabel('Net Foreign Assets', fontsize=14, weight='bold')
    ax.set_title('External Constraint: Capital Flight & Balance of Payments\n(○ = Start, □ = End, × = Capital Flight)', 
                 fontsize=16, weight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved external balance diagram: {save_path}")
    plt.close()

def main():
    """Generate all phase diagrams."""
    print(f"✓ Saved productivity phase diagram: {save_path}")
    plt.close()

def create_external_balance_phase(scenarios: List[str], save_path: Path):
    """
    Figure 5: External Balance Dynamics
    
    Shows capital flight vs net foreign assets trajectory.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for scenario_id in scenarios:
        df = load_scenario(scenario_id)
        info = SCENARIO_INFO[scenario_id]
        
        # Cumulative capital flight
        df['cumulative_flight'] = df['capitalFlight'].cumsum()
        
        ax.plot(
            df['cumulative_flight'],
            df['netForeignAssets'],
            label=info['name'],
            color=info['color'],
            linestyle=info['linestyle'],
            linewidth=2,
            alpha=0.8,
        )
        
        # Mark start and end
        ax.scatter(df.iloc[0]['cumulative_flight'], df.iloc[0]['netForeignAssets'], 
                  color=info['color'], marker='o', s=100, zorder=5)
        ax.scatter(df.iloc[-1]['cumulative_flight'], df.iloc[-1]['netForeignAssets'], 
                  color=info['color'], marker='s', s=100, zorder=5)
    
    ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Cumulative Capital Flight', fontsize=14)
    ax.set_ylabel('Net Foreign Assets', fontsize=14)
    ax.set_title('External Balance Dynamics\n(Capital Controls vs Open Capital Account)', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved external balance diagram: {save_path}")
    plt.close()

def main():
    """Generate all phase diagrams for paper."""
    
    print("="*80)
    print("GENERATING PHASE DIAGRAMS")
    print("="*80)
    
    # SAFETY CHECK: Verify that simV4.py has been run
    if not OUTPUT_DIR.exists():
        print(f"ERROR: {OUTPUT_DIR}/ not found. Run simV4.py first.")
        return
    
    # Define scenario sets for different figures
    all_scenarios = ['scenario_1', 'scenario_2b', 'scenario_2c', 'scenario_3', 'scenario_3b']
    key_comparison = ['scenario_2c', 'scenario_3', 'scenario_3b']  # Focus on transition scenarios
    
    print("\nGenerating figures...")
    
    # FIGURE 2: Main phase diagram (Π vs L/K)
    # This is the centerpiece visualization showing profit-debt dynamics
    create_profit_debt_phase_diagram(
        scenarios=all_scenarios,
        save_path=FIGURES_DIR / "figure2_phase_diagram.png"
    )
    
    # FIGURE 3: Output dynamics over time
    # Shows which scenarios grow vs stagnate
    create_output_capital_phase_diagram(
        scenarios=all_scenarios,
        save_path=FIGURES_DIR / "figure3_output_dynamics.png"
    )
    
    # FIGURE 4: Productivity-growth linkage
    # Demonstrates infrastructure feedback mechanism (v4.0 feature)
    create_productivity_growth_phase(
        scenarios=all_scenarios,
        save_path=FIGURES_DIR / "figure4_productivity_growth.png"
    )
    
    # FIGURE 5: External balance dynamics
    # Shows capital flight vs capital controls effectiveness
    create_external_balance_phase(
        scenarios=key_comparison,  # Focus on transition scenarios
        save_path=FIGURES_DIR / "figure5_external_balance.png"
    )
    
    print("\n" + "="*80)
    print(f"✓ All phase diagrams exported to {FIGURES_DIR}/")
    print("="*80)
    print("\nFigures ready for paper:")
    print("  - figure2_phase_diagram.png: Main (Π, L/K) trajectories")
    print("  - figure3_output_dynamics.png: Output time paths")
    print("  - figure4_productivity_growth.png: Productivity-led growth")
    print("  - figure5_external_balance.png: Capital flight dynamics")

# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
# This block only runs when called directly (not when imported as module)
# Usage: python3 phase_diagrams.py

if __name__ == "__main__":
    main()
