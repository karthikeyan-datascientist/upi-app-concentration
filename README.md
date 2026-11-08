# UPI App Concentration: Data Analytics Capstone Synopsis

Market concentration (HHI) and share dynamics among UPI apps in India, using NPCI's public monthly
"UPI Applications" tables (January 2022 - September 2026).

## 1. Get the data (manual download; NPCI does not allow automated access)
1. Open https://www.npci.org.in/product/ecosystem-statistics/upi
2. Choose the **UPI Applications** tab.
3. Select a month and year, click **Download**.
4. Save each file unchanged in `data/raw/` (any name containing `YYYY-Mon`, e.g. `upi_apps_2022-Jan.xlsx`).
5. Repeat for every month. Months that cannot be downloaded are left missing (not imputed).

## 2. Run
```bash
pip install -r requirements.txt
python code/combine_apps.py --raw data/raw --out data/processed   # parse, audit, harmonize -> panel CSV
python code/prelim_analysis.py                                    # sample-size inputs + preliminary estimates
```

## 3. Outputs
- `data/processed/upi_app_panel.csv` - month x app panel (volume Mn, value Cr, share, TPAP flag)
- `data/processed/file_audit.csv` - one row per source file with its status (used / excluded / relabeled)
- `outputs/results.json`, `outputs/system_series.csv`, `outputs/figures/fig1.png`, `fig2.png`

## 4. Data notes
- Three files were exact copies of a neighboring month under a wrong label (Jul 2022, Nov 2022, Jul 2026) and are excluded.
- Mar, May, Jun 2026 could not be downloaded. 51 of 57 calendar months are usable.
- The Apr 2025 file is titled "Dec'24" but its contents are April 2025; it is relabeled by file name.
- Paytm Payments Bank App and Paytm (OCL) are combined into one `Paytm` entity; FamApp variants are combined.
- TPAP flag: the "#" marker NPCI adds from 2024, extended to earlier months by name (assumption).

## 5. Planned scripts (weeks 3-9 of the work plan)
`rq1_its.py`, `rq2_peaks.py`, `rq3_persistence.py`, `rq4_forecast.py`, `make_tables.py`

Source: National Payments Corporation of India, UPI ecosystem statistics. Public aggregate data; no personal data.
