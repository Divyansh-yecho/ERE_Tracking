# Data

No raw data is stored in this repository. The notebooks read their inputs from Kaggle datasets (paths under `/kaggle/input/...`). This page lists every input and its original source.

| Dataset | Kaggle input used in the notebooks | Original source | Checkpoints |
|---|---|---|---|
| ERA5 daily total precipitation, India, 1°, JJAS 1998–2020 | `divyanshyecho/era5-india-daily-precip` (`era5_india_daily_correct.nc`) | Built in CP1 from ERA5 hourly `total_precipitation` via the [Copernicus CDS API](https://cds.climate.copernicus.eu/) | CP1, CP2 |
| ERA5 daily total precipitation, 0.25°, JJAS 1979–2024 (46 yearly files, mm/day) | `divyanshyecho/era5-0-25-daily-mmday` | ERA5 single levels, `tp`, hourly accumulations summed and ×1000 | CP3, CP4, CP5, CP6a |
| ERA5 850 hPa wind, daily mean | `divyanshyecho/era5-uv850-mean` (`wind_daily_uv850.nc`) | ERA5 pressure levels, mean of 00/06/12/18 UTC | CP6 |
| ERA5 700 / 750 / 800 hPa wind, daily mean | `wind_daily_uv700.nc` and similar | ERA5 pressure levels | CP6c |
| NASA IMERG daily precipitation, 0.25°, JJAS 2000–2024 | `divyanshyecho/imerge-2000-2024-daily` (`imerg_india_025_jjas_2000_2024.nc`) | [NASA GPM IMERG](https://gpm.nasa.gov/data/imerg), regridded to the ERA5 0.25° grid | CP6b, CP6c |
| Monsoon low-pressure-system tracks, 2000–2022 | uploaded CSV | Vishnu et al. (2020), [doi:10.5281/zenodo.3890646](https://doi.org/10.5281/zenodo.3890646) | CP6b, CP6c |
| ETOPO 2022 relief, India subset | uploaded GeoTIFF / NetCDF | [NOAA NCEI ETOPO 2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model) | CP6c |

**Common grid:** 5–40°N, 60–100°E at 0.25°, giving arrays of shape (time, 141, 161).
**Season:** June–September (JJAS); tracks are cut at 30 September.

To run locally, download the files and point the path variables at the top of each notebook or script to your copies.
