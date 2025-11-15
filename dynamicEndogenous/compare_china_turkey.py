"""Generate side-by-side summary statistics for China and Turkey.

The script intentionally avoids coupling with simulation modules. It only
depends on CSV outputs produced by `fetch_china_data.py` and the existing
Turkish calibration table under `paper/tables/`.

Example
-------
$ python compare_china_turkey.py \
      --china data/chn_macro_merged.csv \
      --turkey paper/tables/table_turkish_calibration.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

PERIODS = {
    "china_1980_1993": (1980, 1993),  # Infrastructure build-up phase
    "china_1994_2005": (1994, 2005),  # Post-banking reform consolidation
    "china_1980_2005": (1980, 2005),  # Entire observation window
}

CHINA_METRICS = [
    "gross_capital_formation_pct_gdp",
    "public_capital_formation_pct_gdp",
    "paved_roads_pct",
    "electricity_use_kwh_per_capita",
    "energy_use_kg_oil_eq_per_capita",
    "inflation_cpi_pct",
    "gdp_growth_real_pct",
    "tfp_level",
    "capital_stock",
    "labor_share",
    "investment_share_gdp",
]

TURKEY_PARAM_MAP = {
    "Labor productivity": "labor_productivity",
    "Depreciation rate": "capital_depreciation",
    "Firm markup": "firm_markup",
    "Wage share": "labor_share",
    "Propensity to consume": "propensity_to_consume",
    "Tax rate": "tax_rate",
    "Real interest rate": "real_interest_rate",
    "Import propensity": "import_propensity",
    "Capital flight sensitivity": "capital_flight_sensitivity",
}

ALL_METRICS = CHINA_METRICS + sorted(set(TURKEY_PARAM_MAP.values()))


def compute_means(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, float]:
    return df[list(columns)].mean(skipna=True).to_dict()


def load_china(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "year" not in df.columns:
        raise ValueError("China dataset must contain a 'year' column")
    return df


def summarize_china(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for label, (start, end) in PERIODS.items():
        mask = (df["year"] >= start) & (df["year"] <= end)
        subset = df.loc[mask]
        if subset.empty:
            stats[label] = {metric: float("nan") for metric in CHINA_METRICS}
        else:
            stats[label] = compute_means(subset, CHINA_METRICS)
    return stats


def load_turkey(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["Model Value"] = pd.to_numeric(df["Model Value"], errors="coerce")
    values: Dict[str, float] = {}
    for param, metric in TURKEY_PARAM_MAP.items():
        match = df.loc[df["Parameter"] == param, "Model Value"]
        if not match.empty:
            values[metric] = match.iloc[0]
    return values


def build_summary(china_stats: Dict[str, Dict[str, float]], turkey_stats: Dict[str, float]) -> pd.DataFrame:
    columns = list(PERIODS.keys()) + ["turkey_reference"]
    rows: List[Dict[str, float]] = []
    for metric in ALL_METRICS:
        row = {"metric": metric}
        for label in PERIODS.keys():
            row[label] = china_stats.get(label, {}).get(metric)
        row["turkey_reference"] = turkey_stats.get(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--china",
        type=Path,
        default=Path("data/chn_macro_merged.csv"),
        help="Path to merged China dataset produced by fetch_china_data.py",
    )
    parser.add_argument(
        "--turkey",
        type=Path,
        default=Path("paper/tables/table_turkish_calibration.csv"),
        help="Path to Turkish calibration table (CSV)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/china_vs_turkey_summary.csv"),
        help="Destination CSV for the comparison table",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    china_df = load_china(args.china)
    china_stats = summarize_china(china_df)
    turkey_stats = load_turkey(args.turkey)

    summary = build_summary(china_stats, turkey_stats)
    summary.to_csv(args.output, index=False)
    print(f"Saved comparison table to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
