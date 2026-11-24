# MITT-CDA: cropland degradation screening for Akmola Oblast, Kazakhstan

Code and data for the manuscript *A Reproducible Multi-Index Threshold and Trend Framework for Cropland Degradation Screening from Landsat 8 Time Series: The Case of Akmola Oblast, Kazakhstan* (Akanova, Nurzhan, Bigaliyeva, Ospanova, Kubegenova).

MITT-CDA (Multi-Index Threshold and Trend Cropland Degradation Assessment) screens dryland cropland using five Landsat 8 indicators: NDVI, NDMI, LST, TGSI and SI. The anomalies are standardized against a 2013–2019 baseline whose scale is shrunk towards the 25 km neighbourhood. They are then adjusted for seasonal precipitation (ERA5-Land) and expressed relative to the cropland of the surrounding 25 km cell. Two separate rules flag pixels:

- the threshold branch flags an anomaly above 1.5 in at least 3 of the 5 years 2020–2024;
- the trend branch applies the Mann–Kendall test with exact critical values over 2013–2024.

The layers are compared with permutation benchmarks and evaluated on the hold-out years 2025 and 2026. They are screening priorities for field inspection and have not been validated against ground data.

## Repository structure

```
gee/        Google Earth Engine scripts (Code Editor, JavaScript)
analysis/   Python scripts that produce all sample-based numbers, tables and figures
data/       exported tables and rasters used by the Python scripts
results/    JSON outputs of the Python scripts, as used in the manuscript
legacy/     archived exploratory GEE script referencing the earlier uncorrected product
```

### Google Earth Engine (`gee/`)

| Script | What it does | Output |
|---|---|---|
| `01_diagnostics.js` | valid years per axis and per pixel, regional median of the anomalies | `diag_regional_median_A_v3.csv`, `diag_n_valid_years_by_axis.csv`, `diag_T_R_by_n_min.csv` |
| `02_adjusted_product_v3.js` | adjusted anomalies 2013–2024, 25 km cell medians, full-raster product with the per-pixel scale | asset `MITTCDA_v3_grid_25km_2013_2024_r2`; `ref_grid_25km_v3_*.csv`, `v3_areas*.csv` |
| `03_holdout_2025_2026.js` | adjusted anomalies of the hold-out years with the baseline and coefficients of 2013–2019; scale floors and composite observation dates | asset `A_holdout_2025_2026`; `sigma_floor_area.csv`, `composite_doy_by_year.csv` |
| `04_sample_export.js` | random sample of 47,451 cropland pixels (seed 20261111) and ERA5-Land covariates | `mittcda_sample_v3_holdout.csv`, `era5land_covariates_2013_2026.tif` |
| `05_baseline_stats.js` | baseline median, scale before and after the floor and number of valid years of every pixel; the same sample with these 20 columns added | asset `baseline_stats_2013_2019`; `mittcda_sample_v4_sigma.csv` |

The four assets listed in the manuscript (`A_stack_v2_2013_2024`, `MITTCDA_v3_grid_25km_2013_2024_r2`, `A_holdout_2025_2026`, `baseline_stats_2013_2019`) are under `projects/akmola/assets/` and are publicly readable (`A_stack_v2_2013_2024` is exported by `WorldCereal_GEE.ipynb`).

### Data (`data/`)

- `mittcda_sample_v4_sigma.csv.gz`: the sample with 121 columns. It holds the adjusted anomalies of 2013–2026, the flags of the stored product, the coordinates, the mask tier, the district code, the GLC_FCS30D classes of 2013–2022 and the baseline statistics. Decompress it before use: `gunzip data/mittcda_sample_v4_sigma.csv.gz`. It contains every column of `mittcda_sample_v3_holdout.csv`, row for row.
- `era5land_covariates_2013_2026.tif`: ERA5-Land precipitation of the window, pre-season precipitation and air temperature by year.
- `ref_grid_25km_v3_<axis>.csv`: medians of the adjusted anomalies in 25 km cells (full raster).
- `diag_*.csv`, `sigma_floor_area.csv`, `composite_doy_by_year.csv`, `v3_areas*.csv`: diagnostic tables from scripts 01–03.

### Analysis (`analysis/`)

Python 3.11 or later. Install the packages with `pip install -r requirements.txt`. Run every script from the `analysis/` folder. The release tagged `v1.0.0` reproduces the manuscript.

All layers of Sections 3.2–3.4 come from one procedure, `procedure.py`. First it recovers the indicator values from the sample (x = μ + σ·(δ·A + b·Z_P)). Then, for the given order of years, it re-estimates the baseline median, the scale (per-pixel, shrunk or pooled), the standardized precipitation and the 25 km reference, and applies the decision rules. The observed order of years and the permuted orders go through the same procedure.

| Step | Command | Manuscript |
|---|---|---|
| 1 | `python mittcda_v3_analysis.py ../data ../results` | Tables 4 and 5, Figure 4, year-effect statistics (Section 3.1) |
| 2 | `python compute_numbers.py ../data ../results` | Tables 7–11, the descriptive numbers of Sections 3.2–3.4, the composite index, the trend tests, maps |
| 3 | `python perm_benchmarks.py ../data ../results pixel` and `... shrunk` | Tables 6 and 12: the same procedure on permuted years (sizes and hold-out years); about 12 minutes each |
| 4 | `python perm_strata.py ../data ../results` | The trend layer by stratum on permuted years (Table 10); about 5 minutes |
| 5 | `python make_figures.py ../data ../figures` and `python make_figures_main.py ../results ../figures` | Figure 4; Figures 1, 3, 5, 6 and 7 |
| 6 | `python diagnostics_checks.py ../data ../results/diagnostics_checks.json` | Shrinkage weights by number of baseline values (Section 2.2), check of the PCA weights (Section 2.3), serial dependence of the final anomalies: lag-1 distribution, trend-free pre-whitening and the AR(1) stress scenario (Section 3.3) |

`stepb.py` holds the shared loader and the constants. Its own command line reproduces the pre-registered check of the shrunk scale from 15 November 2026: `results/stepb_V0.json` and `stepb_V2.json`, with the decision rule under `decision`. `check_numbers.py OLD.docx NEW.docx` lists the numbers that differ between two versions of the manuscript.

## Notes

- The stored full-raster product `MITTCDA_v3_grid_25km_2013_2024_r2` carries the per-pixel scale and the interpolated reference. The layers with the shrunk scale reported in the manuscript are computed on the sample by `procedure.py`, and their areas are sample estimates.
- The study-area map (Figure 2) and the correlation screening of Table 3 come from the original pipeline notebook, `WorldCereal_GEE.ipynb`.
- ERA5-Land values returned by Earth Engine for part of the region changed between computations. A re-run reproduces the stored anomalies within 0.013 σ.

## License

MIT (see `LICENSE`).
