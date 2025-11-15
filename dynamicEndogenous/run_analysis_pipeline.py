"""
Master Analysis Pipeline for SFC-ML

Orchestrates complete analysis workflow:
1. Generate simulation results (simV4.py)
2. Create comparative tables (comparative_analysis.py)
3. Generate phase diagrams (phase_diagrams.py)
4. Run Monte Carlo robustness checks (monte_carlo.py) - optional

Usage:
    python3 run_analysis_pipeline.py
    
Options:
    1. Full pipeline (includes Monte Carlo, ~10 min)
    2. Quick pipeline (skip Monte Carlo, ~1 min)

WHAT IT DOES:
  This script automatically runs the following steps in sequence:
  
  STEP 1: python3 simV4.py
    - Runs all 6 scenarios (baseline, conflict, managed, reformed, socialist, partial)
    - Generates output/*.csv files with time series data
    - Creates output/figures/*.png diagnostic plots
  
  STEP 2: python3 comparative_analysis.py
    - Calculates steady-state metrics
    - Generates Table 1 (comparative statics)
    - Generates Table 2 (transition costs)
    - Exports to paper/tables/
  
  STEP 3: python3 phase_diagrams.py
    - Creates 4 phase space diagrams
    - Exports Figures 2-5 to paper/figures/
  
  STEP 4 (optional): python3 monte_carlo.py
    - Runs 500 simulations per scenario with parameter variation
    - Generates Table 3 (robustness checks)
    - Exports to paper/monte_carlo/
    - Takes ~5-10 minutes

DEPENDENCIES:
  - Python packages: pandas, numpy, matplotlib (from requirements.txt)
  - Optional: jinja2 for LaTeX table export
  - All analysis scripts must be in the same directory

OUTPUT STRUCTURE:
  output/                    # Raw simulation data
    ├── scenario_*.csv       # Time series for each scenario
    └── figures/*.png        # Diagnostic plots
  
  paper/                     # Publication-ready materials
    ├── tables/
    │   ├── table1_comparative_statics.tex
    │   └── table2_transition_costs.csv
    ├── figures/
    │   ├── figure2_phase_diagram.png
    │   ├── figure3_output_dynamics.png
    │   ├── figure4_productivity_growth.png
    │   └── figure5_external_balance.png
    └── monte_carlo/         # (if running full pipeline)
        └── table3_robustness.tex

WHO CALLS THIS:
  - Called by USER from command line
  - This script then calls: simV4.py → comparative_analysis.py → 
    phase_diagrams.py → monte_carlo.py (optional)
"""

import subprocess
import sys
from pathlib import Path

def run_command(description: str, command: list, cwd=None):
    """Run a command and report status."""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {' '.join(command)}")
    
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"✓ SUCCESS: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FAILED: {description}")
        print(f"Error: {e}")
        return False

def check_python_environment():
    """Verify Python environment has required packages."""
    print("Checking Python environment...")
    required_packages = ['pandas', 'numpy', 'matplotlib']
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"✗ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print("✓ All required packages installed")
    return True

def main():
    """Execute the full analysis pipeline."""
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                   SFC-ML PUBLICATION PIPELINE                         ║
    ║                                                                        ║
    ║  Generates all tables, figures, and statistics for CJE submission     ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Check environment
    if not check_python_environment():
        print("\nPlease install missing packages before continuing.")
        sys.exit(1)
    
    # Determine Python command (use venv if available)
    python_cmd = sys.executable
    print(f"\nUsing Python: {python_cmd}")
    
    cwd = Path.cwd()
    
    # Pipeline steps
    MONTE_CARLO_DEFAULT_RUNS = 500
    steps = [
        {
            'description': 'STEP 1: Generate Base Simulation Results',
            'command': [python_cmd, 'simV4.py'],
            'required': True,
        },
        {
            'description': 'STEP 2: Create Comparative Analysis Tables',
            'command': [python_cmd, 'comparative_analysis.py'],
            'required': True,
        },
        {
            'description': 'STEP 3: Generate Phase Diagrams',
            'command': [python_cmd, 'phase_diagrams.py'],
            'required': True,
        },
        {
            'description': 'STEP 4: Run Monte Carlo Robustness Analysis (500 runs, ~5-10 min)',
            'command': [python_cmd, 'monte_carlo.py'],
            'required': False,  # Optional - takes time
        },
    ]
    
    # Ask user if they want full pipeline or skip Monte Carlo
    print("\nOptions:")
    print("  1. Full pipeline (including Monte Carlo - slower)")
    print("  2. Quick pipeline (skip Monte Carlo)")
    choice = input("\nSelect option [1/2, default=2]: ").strip() or "2"
    
    if choice == "2":
        print("\n→ Skipping Monte Carlo analysis (can run separately later)")
        steps = [s for s in steps if 'Monte Carlo' not in s['description']]
    else:
        runs_input = input(f"\nMonte Carlo runs per scenario [default={MONTE_CARLO_DEFAULT_RUNS}]: ").strip()
        if runs_input:
            try:
                monte_carlo_runs = int(runs_input)
                if monte_carlo_runs <= 0:
                    raise ValueError
            except ValueError:
                print(f"Invalid input. Using default of {MONTE_CARLO_DEFAULT_RUNS} runs.")
                monte_carlo_runs = MONTE_CARLO_DEFAULT_RUNS
        else:
            monte_carlo_runs = MONTE_CARLO_DEFAULT_RUNS

        for step in steps:
            if 'Monte Carlo' in step['description']:
                step['description'] = step['description'].replace('500 runs', f"{monte_carlo_runs} runs")
                step['command'] = [python_cmd, 'monte_carlo.py', '--runs', str(monte_carlo_runs)]
                break
    
    # Execute pipeline
    success_count = 0
    total_steps = len(steps)
    
    for i, step in enumerate(steps, 1):
        print(f"\n\n{'#'*80}")
        print(f"# PIPELINE STEP {i}/{total_steps}")
        print(f"{'#'*80}")
        
        success = run_command(
            description=step['description'],
            command=step['command'],
            cwd=cwd
        )
        
        if success:
            success_count += 1
        elif step['required']:
            print(f"\n✗ Required step failed. Aborting pipeline.")
            sys.exit(1)
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("\n✓ All analysis complete!")
        print("\nGenerated outputs:")
        print("  • output/           - Raw simulation CSVs and metadata")
        print("  • paper/tables/     - LaTeX and CSV tables for publication")
        print("  • paper/figures/    - Phase diagrams and visualizations")
        if choice == "1":
            print("  • paper/monte_carlo/ - Robustness analysis results")
        
        print("\nNext steps:")
        print("  1. Review tables in paper/tables/")
        print("  2. Insert figures from paper/figures/ into manuscript")
        print("  3. Write results section around comparative table")
        if choice == "2":
            print("  4. (Optional) Run Monte Carlo: python monte_carlo.py")
    else:
        print("\n⚠ Some steps failed. Check errors above.")
    
    print("\n")

if __name__ == "__main__":
    main()
