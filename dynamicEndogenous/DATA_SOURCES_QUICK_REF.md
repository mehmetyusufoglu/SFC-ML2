# Turkish Calibration - Quick Reference

**Calibration Period:** 2010-2020

## Production (Penn World Tables 10.0)
- **Productivity (a=0.85):** Turkey TFP, variable `ctfp`
- **Depreciation (δ=0.048):** Capital depreciation, variable `delta`

## Distribution (TurkStat)
- **Markup (μ=0.28):** Input-Output Tables 2012/2017, manufacturing sector
- **Wage share (v=0.62):** National Accounts, compensation/GDP

## Behavior (OECD, IMF)
- **Consumption (c=0.85):** OECD household consumption/disposable income
- **Tax rate (τ=0.20):** IMF WEO total tax/GDP

## Finance (CBRT)
- **Interest (i=0.031):** CBRT Statistics, real funding cost 2013-2019
- **Investment (α₁=0.25):** Estimated from GFCF/profit correlation

## External (World Bank, Literature)
- **Imports (m=0.23):** World Bank WDI, excluding energy imports
- **Export elasticity (εₓ=0.50):** Gülmez & Yeldan (2014) study

## Crisis (CBRT)
- **Capital flight (κ=0.60):** 2018 crisis calibration
- **Global profit (r*=0.10):** OECD corporate ROIC average

## Data Access
- Penn World Tables: https://www.rug.nl/ggdc/productivity/pwt/
- TurkStat: https://data.tuik.gov.tr/
- CBRT EVDS: https://evds2.tcmb.gov.tr/
- World Bank: https://databank.worldbank.org/
- OECD: https://stats.oecd.org/
- IMF WEO: https://www.imf.org/en/Publications/WEO

See `TURKISH_DATA_SOURCES.md` for complete documentation.

