"""Utility to download core Chinese macro + infrastructure indicators from public APIs.

- World Bank WDI via `wbgapi`
- Penn World Table 10 via `pwt`

The output consists of:
1. data/raw/china_wdi_core.csv
2. data/raw/china_pwt10_core.csv
3. data/china_macro_merged.csv (outer join of the two sources)

Example:
    $ python fetch_china_data.py --start 1980 --end 2005
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

import pandas as pd
import wbgapi as wb

# --- Configuration ---------------------------------------------------------
WDI_SERIES: Dict[str, str] = {
    "NE.GDI.FTOT.ZS": "gross_capital_formation_pct_gdp",
    "NE.GDI.PUBC.ZS": "public_capital_formation_pct_gdp",
    "IS.ROD.PAVE.ZS": "paved_roads_pct",
    "EG.USE.ELEC.KH.PC": "electricity_use_kwh_per_capita",
    "EG.USE.PCAP.KG.OE": "energy_use_kg_oil_eq_per_capita",
    "FP.CPI.TOTL.ZG": "inflation_cpi_pct",
    "NY.GDP.MKTP.KD.ZG": "gdp_growth_real_pct",
    "GC.XPN.TOTL.GD.ZS": "gov_expenditure_pct_gdp",
}

PWT_COLUMNS: Dict[str, str] = {
    "rgdpna": "real_gdp_na",
    "ctfp": "tfp_level",
    "delta": "capital_depreciation",
    "labsh": "labor_share",
    "rkna": "capital_stock",
    "csh_i": "investment_share_gdp",
    "emp": "employment_millions",
}

DEFAULT_OUTPUT_DIR = Path("data")
RAW_DIR = DEFAULT_OUTPUT_DIR / "raw"
PWT_URL = "https://www.rug.nl/ggdc/docs/pwt100.xlsx"
PWT_CACHE_NAME = "pwt100.xlsx"


# --- World Bank WDI --------------------------------------------------------
def fetch_wdi(country: str, start_year: int, end_year: int) -> pd.DataFrame:
    time_range = range(start_year, end_year + 1)
    df = wb.data.DataFrame(list(WDI_SERIES.keys()), economy=country, time=time_range)
    if df.empty:
        raise RuntimeError("World Bank API returned no data. Check network or series codes.")

    df = df.T  # columns -> indicator codes, index -> YR####
    df.index = df.index.str.replace("YR", "").astype(int)
    df.index.name = "year"
    df.reset_index(inplace=True)
    desired_columns = ["year", *WDI_SERIES.keys()]
    df = df.reindex(columns=desired_columns)
    df.rename(columns=WDI_SERIES, inplace=True)
    df.sort_values("year", inplace=True)
    return df


# --- Penn World Table ------------------------------------------------------
def _download_pwt(cache_path: Path) -> Path:
    if cache_path.exists():
        return cache_path
    print("Downloading PWT 10.0 workbook (~6 MB)...")
    urllib.request.urlretrieve(PWT_URL, cache_path)
    return cache_path


def fetch_pwt(country: str, start_year: int, end_year: int, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / PWT_CACHE_NAME
    workbook_path = _download_pwt(cache_path)
    data = pd.read_excel(workbook_path, sheet_name="Data")
    subset = data.query("countrycode == @country and @start_year <= year <= @end_year")
    if subset.empty:
        raise RuntimeError("PWT returned no rows. Ensure the country code is valid.")

    keep_cols: List[str] = ["year", *PWT_COLUMNS.keys()]
    subset = subset[keep_cols].copy()
    subset.rename(columns=PWT_COLUMNS, inplace=True)
    subset.sort_values("year", inplace=True)
    return subset


# --- Output helpers --------------------------------------------------------
def ensure_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(parents=True, exist_ok=True)


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"Saved {path} ({len(df)} rows)")


# --- Main ------------------------------------------------------------------
def main(args: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default="CHN", help="ISO3 country code (default: CHN)")
    parser.add_argument("--start", type=int, default=1980, help="Start year (inclusive)")
    parser.add_argument("--end", type=int, default=2010, help="End year (inclusive)")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory root (default: data/)",
    )
    parsed = parser.parse_args(args=args)

    if parsed.start > parsed.end:
        raise ValueError("Start year must be <= end year")

    ensure_dirs(parsed.outdir)

    print("Fetching World Bank indicators...")
    wdi_df = fetch_wdi(parsed.country, parsed.start, parsed.end)
    save_dataset(wdi_df, parsed.outdir / "raw" / f"{parsed.country.lower()}_wdi_core.csv")

    print("Fetching Penn World Table indicators...")
    pwt_df = fetch_pwt(parsed.country, parsed.start, parsed.end, parsed.outdir / "raw")
    save_dataset(pwt_df, parsed.outdir / "raw" / f"{parsed.country.lower()}_pwt10_core.csv")

    merged = pd.merge(wdi_df, pwt_df, on="year", how="outer").sort_values("year")
    save_dataset(merged, parsed.outdir / f"{parsed.country.lower()}_macro_merged.csv")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
