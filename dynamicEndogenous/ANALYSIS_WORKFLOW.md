# Analysis Workflow

## Automated Pipeline

```bash
cd dynamicEndogenous
python3 run_analysis_pipeline.py
```

## Execution Order

### 1. Core Simulation
**`simV4.py`** - SFC model implementation
- Runs scenarios, generates time series data
- **Outputs:** `output/scenario_*.csv`, `output/figures/scenario_*.png`

### 2. Analysis Tools

**`comparative_analysis.py`** - Tables
- **Requires:** `output/scenario_*.csv`
- **Outputs:** `paper/tables/table1_comparative_statics.tex`, `paper/tables/table2_transition_costs.csv`

**`phase_diagrams.py`** - Figures
- **Requires:** `output/scenario_*.csv`
- **Outputs:** `paper/figures/figure*.png`

**`monte_carlo.py`** - Robustness (Optional)
- **Time:** ~5-10 minutes
- **Outputs:** `paper/monte_carlo/table3_robustness.tex`

### 3. Master Pipeline
**`run_analysis_pipeline.py`** - Orchestrates all scripts

## Manual Execution

```bash
python3 simV4.py
python3 comparative_analysis.py
python3 phase_diagrams.py
python3 monte_carlo.py  # optional
```

## Output Structure

```
dynamicEndogenous/
├── output/
│   ├── scenario_*_results.csv
│   └── figures/
└── paper/
    ├── tables/
    └── figures/
```

    │   ├── figure4_productivity_growth.png
    │   └── figure5_external_balance.png
    └── monte_carlo/                 # (optional)
        ├── scenario_1/
        │   ├── confidence_bands.csv
        │   └── robustness_metrics.csv
        ├── ...
        └── table3_robustness.tex
```

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Optional for LaTeX export:
```bash
pip install jinja2
```

## Workflow Summary

```
USER
  │
  ├─► run_analysis_pipeline.py ────┐
  │                                 │
  └─► (individual scripts)          │
                                    ▼
                            ┌───────────────┐
                            │  simV4.py     │  Generate raw data
                            │  (6 scenarios)│
                            └───────┬───────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌──────────────────────┐      ┌──────────────────────┐
        │ comparative_         │      │ phase_diagrams.py    │
        │ analysis.py          │      │ (Figures 2-5)        │
        │ (Tables 1-2)         │      └──────────────────────┘
        └──────────────────────┘
                    │
                    ▼
        ┌──────────────────────┐
        │ monte_carlo.py       │  (optional)
        │ (Table 3)            │
        └──────────────────────┘
                    │
                    ▼
            Publication materials
            ready in paper/
```

## Key Scenarios

1. **scenario_1**: Baseline Capitalism (fixed wage share)
2. **scenario_2**: Conflict Capitalism (high sensitivity)
3. **scenario_2b**: Managed Conflict (failed transition)
4. **scenario_2c**: Reformed Capitalism (public backstop)
5. **scenario_3**: Socialist Transition (successful)
6. **scenario_3b**: Partial Planning (failed)

## Questions?

- Check file headers for detailed documentation
- Each script has extensive comments explaining purpose and usage
- Run individual scripts to see their specific output
