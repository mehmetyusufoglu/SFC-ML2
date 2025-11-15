# SFC-ML: Stock-Flow Consistent Model for Development Policy Analysis

## Overview

Open-economy Stock-Flow Consistent (SFC) model implementing Godley-Lavoie methodology for analyzing heterodox development policy sequencing. Calibrated to Turkish economy (2010-2020).

**Framework:** Post-Keynesian SFC modeling  
**Application:** Semi-peripheral development policy  
**Language:** Python 3.x

## Quick Start

```bash
cd dynamicEndogenous
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run complete analysis pipeline (recommended)
python3 run_analysis_pipeline.py

# Or run individual components:
# python3 simV4.py                    # Core simulation only
# python3 comparative_analysis.py     # Generate tables
# python3 phase_diagrams.py           # Generate figures
# python3 monte_carlo.py --runs 500   # Robustness analysis
```

## Key Features

- Cost-of-living index responsive to infrastructure investment
- Endogenous capital flight based on profit-rate differentials
- Credit socialization mechanisms
- External balance constraints
- Calibrated parameters from authoritative sources (Penn World Tables, TurkStat, CBRT)

## Repository Structure

```
SFC-ML/
├── README.md
├── dynamicEndogenous/
│   ├── simV4.py                       # Main simulation engine
│   ├── turkish_calibration.py         # Parameter calibration module
│   ├── run_analysis_pipeline.py       # Master analysis script
│   ├── comparative_analysis.py        # Table generation
│   ├── phase_diagrams.py              # Figure generation
│   ├── monte_carlo.py                 # Robustness analysis
│   ├── TURKISH_DATA_SOURCES.md        # Complete data documentation
│   ├── DATA_SOURCES_QUICK_REF.md      # Quick parameter reference
│   ├── IMPLEMENTATION_SUMMARY.md      # Calibration implementation
│   ├── FOCUSED_ROADMAP.md             # Model overview
│   ├── confPaper/                     # Conference materials
│   │   ├── abstract_cassino2026.md
│   │   ├── abstract_cassino2026.tex
│   │   └── README.md
│   └── docs/                          # Technical documentation
│       ├── ANALYSIS_WORKFLOW.md
│       ├── QUICK_REFERENCE.md
│       ├── CRISIS_MECHANISMS_v3_2.md
│       └── DOCUMENTATION_UPDATE.md
└── staticPortablePolicy/              # Legacy code
    └── simV1.py
```

## Analysis Pipeline

The complete analysis workflow executes in 4 steps:

```bash
python3 run_analysis_pipeline.py
```

**Interactive Options:**
- Option 1: Full pipeline (includes Monte Carlo ~5-10 minutes)
- Option 2: Quick pipeline (skip Monte Carlo ~30 seconds)

**Pipeline Steps:**

1. **Simulation** (`simV4.py`): Generates 6 scenario time series
   - Scenario 1: Social Democracy (Fixed Share)
   - Scenario 2: PK Conflict (High Sensitivity)
   - Scenario 2B: Managed Conflict (Institutional Parameters)
   - Scenario 2C: Managed Conflict with Public Backstop
   - Scenario 3: Coordinated Socialist Transition (Successful)
   - Scenario 3B: Partial Planning (Failed Transition)

2. **Comparative Analysis** (`comparative_analysis.py`): Generates LaTeX tables
   - Table 1: Comparative statics across scenarios
   - Table 2: Transition costs vs baseline

3. **Phase Diagrams** (`phase_diagrams.py`): Generates publication figures
   - Figure 2: Phase diagram (Π, L/K)
   - Figure 3: Output dynamics
   - Figure 4: Productivity growth
   - Figure 5: External balance

4. **Monte Carlo** (optional, `monte_carlo.py`): Robustness analysis
   - 500 parameter variations per scenario
   - Confidence bands (5th, 50th, 95th percentiles)
   - Table 3: Robustness metrics

**Output:**
- `output/*.csv`: Raw simulation data (6 scenarios × 15 years)
- `paper/tables/*.tex`: LaTeX tables for manuscript
- `paper/figures/*.png`: Publication-ready figures
- `paper/monte_carlo/`: Robustness analysis results

## Core Model Parameters

All parameters empirically calibrated to Turkish data (2010-2020):

| Parameter | Value | Source |
|-----------|-------|--------|
| Labor productivity | 0.85 | Penn World Tables 10.0 |
| Depreciation rate | 0.048 | Penn World Tables 10.0 |
| Firm markup | 0.28 | TurkStat I-O Tables |
| Wage share | 0.62 | TurkStat National Accounts |
| Propensity to consume | 0.85 | OECD |
| Import propensity | 0.23 | World Bank WDI |

See `TURKISH_DATA_SOURCES.md` for complete documentation.

## Usage

### Run Complete Analysis Pipeline (Recommended)

```bash
cd dynamicEndogenous
python3 run_analysis_pipeline.py
```

This executes all 4 steps: simulations → tables → figures → Monte Carlo (optional)

### Core Simulation Engine (Individual Scenarios)

```python
from simV4 import SFCModel, ModelParams, InitialStocks, PolicySchedule
from simV4 import ProfitLedPrivateBanking, FloatingExchangeRate, FixedShareRegime

# Initialize model
params = ModelParams()
initial = InitialStocks()
policy = PolicySchedule()

# Run simulation
model = SFCModel(
    params=params,
    initial_stocks=initial,
    policy_schedule=policy,
    finance_regime=ProfitLedPrivateBanking(),
    external_regime=FloatingExchangeRate(),
    distribution_regime=FixedShareRegime()
)

# Simulate 15 years
for year in range(1, 16):
    model.step()

# Access results
print(model.results)

```

### Generate Publication Tables

```python
# Run after simulations complete
python3 comparative_analysis.py
```

### Generate Publication Figures

```python
# Run after simulations complete
python3 phase_diagrams.py
```

### Run Robustness Analysis

```bash
# 500 runs per scenario (~5-10 minutes)
python3 monte_carlo.py --runs 500

# Quick test with 10 runs
python3 monte_carlo.py --runs 10
```

### Parameter Calibration
```python
from turkish_calibration import TurkishCalibrationData
calibration = TurkishCalibrationData()
print(calibration.labor_productivity)  # 0.85
```

## Project Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  run_analysis_pipeline.py (Master Orchestrator)             │
│                                                               │
│  Step 1: Simulations                                          │
│    ├─> simV4.py (6 scenarios × 15 years)                     │
│    └─> output/*.csv                                           │
│                                                               │
│  Step 2: Comparative Analysis                                 │
│    ├─> comparative_analysis.py                                │
│    └─> paper/tables/table1_*.tex, table2_*.csv               │
│                                                               │
│  Step 3: Phase Diagrams                                       │
│    ├─> phase_diagrams.py                                      │
│    └─> paper/figures/figure2-5.png                            │
│                                                               │
│  Step 4: Monte Carlo (optional)                               │
│    ├─> monte_carlo.py --runs 500                              │
│    └─> paper/monte_carlo/table3_robustness.tex               │
└─────────────────────────────────────────────────────────────┘
```

### Analysis Pipeline Output Structure

```
dynamicEndogenous/
├── output/                           # Step 1 output
│   ├── scenario_1_results.csv
│   ├── scenario_1_metadata.txt
│   ├── scenario_2_results.csv
│   ├── scenario_2b_results.csv
│   ├── scenario_2c_results.csv
│   ├── scenario_3_results.csv
│   └── scenario_3b_results.csv
├── paper/
│   ├── tables/                       # Step 2 output
│   │   ├── table1_comparative_statics.tex
│   │   ├── table2_transition_costs.csv
│   │   └── table_turkish_calibration.tex
│   ├── figures/                      # Step 3 output
│   │   ├── figure2_phase_diagram.png
│   │   ├── figure3_output_dynamics.png
│   │   ├── figure4_productivity_growth.png
│   │   └── figure5_external_balance.png
│   └── monte_carlo/                  # Step 4 output
│       ├── table3_robustness.tex
│       ├── Reformed Capitalism (2c)/
│       │   ├── confidence_bands.csv
│       │   └── robustness_metrics.csv
│       └── Socialist Transition (3)/
│           ├── confidence_bands.csv
│           └── robustness_metrics.csv
```

## Core Model Architecture

**Simulation Engine:** `simV4.py`
```python
# Generates all tables and figures
import run_analysis_pipeline
```

**Simulation Engine:** `simV4.py`

- **SFCModel**: Main simulation class
- **ModelParams**: Calibrated parameters (Turkish data 2010-2020)
- **InitialStocks**: Starting values (K, L, NFA, etc.)
- **PolicySchedule**: Infrastructure investment schedule
- **Regimes**: Modular policy components
  - Finance: `ProfitLedPrivateBanking`, `StateLedNationalizedBanking`
  - External: `FloatingExchangeRate`, `ManagedExchangeRate`
  - Distribution: `FixedShareRegime`, `ConflictInflationRegime`

**Analysis Scripts:**
- `run_analysis_pipeline.py`: Master orchestrator (4-step workflow)
- `comparative_analysis.py`: Generates comparative statics tables
- `phase_diagrams.py`: Generates publication figures
- `monte_carlo.py`: Parameter sensitivity analysis (500 runs)
- `turkish_calibration.py`: Parameter calibration module

## Output Variables

- `YOutput`: Total output/GDP
- `PiT`: Profit rate
- `LFirmLoan`: Firm debt
- `LaborProductivity`: Labor productivity
- `realWageIndex`: Real wage level
- `exchangeRate`: Exchange rate
- `capitalFlight`: Capital outflows
- `tradeBalance`: Net exports

## Dependencies

- pandas
- numpy
- matplotlib

## Documentation

- `ANALYSIS_WORKFLOW.md`: Execution guide
- `TURKISH_DATA_SOURCES.md`: Data provenance
- `DATA_SOURCES_QUICK_REF.md`: Parameter quick reference
- `confPaper/README.md`: Conference materials

## License

GPL-3.0

## Citation

See `CITATION.cff` (if available) or `confPaper/abstract_cassino2026.md` for citation information.

## Contact

Repository: https://github.com/mehmetyusufoglu/sfc-sequencing-infrastructure

