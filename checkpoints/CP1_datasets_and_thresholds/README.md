# Checkpoint 1 — Datasets and Extreme-Rainfall Thresholds

| Notebook | Script |
|---|---|
| [`cp1_datasets_thresholding.ipynb`](cp1_datasets_thresholding.ipynb) | [`cp1_datasets_thresholding.py`](cp1_datasets_thresholding.py) |

## Goal

Build the first precipitation dataset for the project and define what counts as *extreme* rainfall at each location.

## What this checkpoint does

1. **Download.** ERA5 hourly `total_precipitation` for June–September 1998–2020 over the Indian domain (5–40°N, 60–100°E, 1°), through the Copernicus CDS API. The loop skips years already downloaded, so an interrupted run can resume.
2. **Aggregate.** Hourly accumulations are summed to daily totals in mm/day (122 JJAS days per year), and the 23 yearly files are merged into one dataset.
3. **Sanity checks.** Per-year maximum and mean rainfall: daily maxima of 280–470 mm/day and JJAS means of about 5–6 mm/day.
4. **Thresholds** (following Nikumbh et al. 2019): for each grid cell, the 99th percentile of rainy-day rainfall. A sanity check confirms that wet regions such as the Western Ghats get higher thresholds than dry regions such as Pakistan.
5. **First detection.** Daily exceedance masks are grouped into connected objects (`scipy.ndimage.label`), giving the daily number of extreme cells (NT) and of ERE objects (NE).
6. **Known-event check.** The detection is compared against documented floods (Mumbai 2005, Uttarakhand 2013, Kerala 2018, Odisha 2001).

The resulting daily file (`era5_india_daily_correct.nc`) is the input to CP2.

## Notes

- Add your own CDS API key to `~/.cdsapirc` before running. The notebook uses the placeholder `YOUR-API-KEY-HERE`.
- The notebook ends with a note that the 99th percentile should be computed per month, because of high NT in June. This is taken up in CP2, and CP3 later adds a 50 mm/day floor and an India-only mask.
- This 1° dataset turned out to be too coarse for overlap-based tracking (see CP2). CP3 moves to 0.25°.

## Next

[CP2: first detection and tracking at 1°](../CP2_tracking_era5_1deg)
