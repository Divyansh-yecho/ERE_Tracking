# Data

No data files are stored in this repository. Put the inputs in this folder (`data/`), or point the scripts to another folder with the environment variable `ERE_DATA`. Intermediate tables are written to `data/derived/`, which git ignores.

**Common grid:** 5–40°N, 60–100°E at 0.25°, giving arrays of shape (time, 141, 161), latitude descending.
**Season:** June–September (JJAS); tracks are cut at 30 September.

## Inputs for the local scripts (CP6 catalogue, CP7, CP8)

| File | Content | Source | Used by |
|---|---|---|---|
| `imerg_india_025_jjas_2000_2024.nc` | Daily IMERG precipitation, variable `tp` in mm/day, JJAS 2000–2024 (3,050 days) | [NASA GPM IMERG](https://gpm.nasa.gov/data/imerg) daily, regridded to the ERA5 0.25° grid. Kaggle: `divyanshyecho/imerge-2000-2024-daily` | CP6, CP7 |
| `lps_daily_2000_2022.parquet` | One daily-mean centre per low-pressure system, JJAS 2000–2022: 2,232 system-days, 345 systems. Columns `track_id, date, lon, lat, lon_med, lat_med, n_hours, min_streamfn, rh850, span_km` | Vishnu et al. (2020) LPS track catalogue ([doi:10.5281/zenodo.3890646](https://doi.org/10.5281/zenodo.3890646), CC-BY-4.0), hourly centres averaged per day. Kaggle: `divyanshyecho/vorticity-center-2000-2022` | CP6, CP7 |
| `olr.cbo-2.5deg.day.mean.nc` | CPC blended OLR, 2.5°, daily, 1991–present | [NOAA PSL](https://downloads.psl.noaa.gov/Datasets/cpc_blended_olr-2.5deg/olr.cbo-2.5deg.day.mean.nc). The older NOAA interpolated OLR ends on 2022-12-31, so it cannot cover 2023–24. | CP8 |
| `BSISO.INDEX.NORM.LY.data` | APCC BSISO index (Lee et al. 2013), daily 1981–present: `YEAR DOY PC1 PC2 PC3 PC4 amp1 amp2`, missing = −999.9 | [APCC direct download](https://apcc21.org/clikapi/download/BSISO/BSISO.INDEX.NORM.LY.data) | CP8 |

## Inputs for the Kaggle notebooks (CP1–CP6)

The notebooks read their inputs from `/kaggle/input/...`. To run them elsewhere, change the paths at the top of each notebook.

| Dataset | Kaggle input used in the notebooks | Original source | Checkpoints |
|---|---|---|---|
| ERA5 daily total precipitation, India, 1°, JJAS 1998–2020 | `divyanshyecho/era5-india-daily-precip` (`era5_india_daily_correct.nc`) | Built in CP1 from ERA5 hourly `total_precipitation` via the [Copernicus CDS API](https://cds.climate.copernicus.eu/) | CP1, CP2 |
| ERA5 daily total precipitation, 0.25°, JJAS 1979–2024 (46 yearly files, mm/day) | `divyanshyecho/era5-0-25-daily-mmday` | ERA5 single levels, `tp`, hourly accumulations summed and ×1000 | CP3–CP5, CP6a |
| ERA5 850 hPa wind, daily mean | `divyanshyecho/era5-uv850-mean` (`wind_daily_uv850.nc`) | ERA5 pressure levels, mean of 00/06/12/18 UTC | CP6 |
| ERA5 700 / 750 / 800 hPa wind, daily mean | `wind_daily_uv700.nc` and similar | ERA5 pressure levels | CP6c |
| IMERG daily precipitation and the LPS catalogue | as in the table above | | CP6b, CP6c |
| ETOPO 2022 relief, India subset (60 arc-second) | uploaded GeoTIFF / NetCDF | [NOAA NCEI ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model), doi:10.25921/fd45-gt74 | CP6c |

## Files written to `data/derived/`

| Files | Written by |
|---|---|
| `imerg_setup.npz` (grid, India mask, thresholds, daily object labels), `objdays_{mean,min05}.parquet`, `tracks_{mean,min05}.parquet`, `merges_*.parquet`, `splits_*.parquet` | `checkpoints/CP6_imerg_lps_orography/cp6_build_imerg_catalogue.py` |
| `cp7_objdays.parquet`, `cp7_tracks.parquet`, `cmz_daily_index.csv` | `checkpoints/CP7_regional_trend_analysis/cp7_00_prepare_tables.py` |
| `spell_drivers_daily.csv` | `checkpoints/CP8_spells_olr_bsiso/cp8_00_build_drivers.py` |
