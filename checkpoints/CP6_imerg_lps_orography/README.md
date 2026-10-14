# Checkpoint 6 — What Drives Multi-Day ERE Tracks? Wind, LPS and Orography

| Notebook | Script | Question |
|---|---|---|
| [`cp6a_wind_steering_test.ipynb`](cp6a_wind_steering_test.ipynb) | [`cp6a_wind_steering_test.py`](cp6a_wind_steering_test.py) | Does the 850 hPa wind steer the tracks? |
| [`cp6b_imerg_tracking_lps_colocation.ipynb`](cp6b_imerg_tracking_lps_colocation.ipynb) | [`cp6b_imerg_tracking_lps_colocation.py`](cp6b_imerg_tracking_lps_colocation.py) | Do multi-day tracks follow monsoon low-pressure systems (LPS)? |
| [`cp6c_orographic_attribution.ipynb`](cp6c_orographic_attribution.ipynb) | [`cp6c_orographic_attribution.py`](cp6c_orographic_attribution.py) | What drives the tracks that are far from any LPS? |

The tracker only links rain objects by overlap. It has no wind and no dynamics. That raises an obvious objection: are the multi-day tracks real weather systems, or chance chains of unrelated rain blobs? This checkpoint tests it.

---

## CP6a — Wind as a steering vector: rejected

The ERA5 850 hPa wind was first validated: it has no missing values, its grid matches the rainfall grid, and it shows the south-westerly monsoon flow over the Arabian Sea and Bay of Bengal. Its footprint-mean direction was then compared with each track step.

- **Multi-day tracks, 313 clean steps:**
  - mean cos(Δ) = −0.04, so the steps are not aligned with the wind;
  - only 48% of steps are within 90° of the wind direction;
  - the Rayleigh test cannot reject a uniform distribution (p = 0.26);
  - tracks move at only 0.16× the wind speed.
- **On IMERG (CP6b),** 69% of clean steps point *against* the footprint wind.

So the low-level wind does not steer the tracks. This motivated validating against independent weather systems instead. Note the separate question: wind failing to *steer* a track does not rule out wind being *lifted by terrain* under it (tested in CP6c).

## CP6b — IMERG tracking and LPS co-location

The CP5 MEAN tracker was re-run on **NASA IMERG** (0.25°, JJAS 2000–2024): **10,006 tracks, 71 merges, 51 splits, 10,877 object-days.**

Each rain object-day was joined to the nearest LPS centre on the same date, using the Vishnu et al. (2020) catalogue, which covers 2000–2022.

| Population | Median distance to nearest LPS | Within 500 km |
|---|---|---|
| All matchable object-days | 737 km | 35.1% |
| Multi-day tracks (> 2 days, n = 395) | **427 km** | **56.2%** |

- **Longer tracks sit closer to depressions:** median 370 km for tracks of 5+ days vs 763 km for 1–2-day tracks (Mann-Whitney p = 8×10⁻³).
- **Rain is offset from the depression centre** by a median of about 216 km west and 47 km south. This matches the south-west displacement reported in the literature.
- **Strict classification of 3+-day tracks** by mean nearest-LPS distance:
  - NEAR (≤ 500 km): 65
  - FAR (> 500 km): 60
  - NO-LPS (no matchable day; includes every track in 2023–24, after the catalogue ends): 31
- **FAR tracks move much more slowly than NEAR tracks:** 75 vs 124 km/day (p = 7.7×10⁻⁷, Cliff's δ = −0.50). That is what rain anchored to fixed terrain would do.

**Track-level shuffle null.** Each multi-day track is kept rigid and re-dated to a random other year, so the seasonal cycle is preserved but the real-year alignment is broken. The final run of this test found co-location at **3–5× chance**: 23.2% vs 5.0% within 300 km, p < 2×10⁻⁵.

> **Note:** in the notebook version committed here, the shuffle-null cell (the one that plots *Observed co-location vs track-level null*) stops with `ValueError: need at least one array to concatenate`. The 3–5× numbers come from the final run recorded in the project notes. Re-run that cell to regenerate them.

<p align="center"><img src="../../figures/cp6_rain_offset_from_lps.png" width="380" alt="Rain position relative to the LPS centre">
<img src="../../figures/cp6_far_vs_near_motion.png" width="520" alt="Daily step and directedness, FAR vs NEAR tracks"></p>

## CP6c — Orographic attribution of FAR tracks

- **Terrain.** ETOPO 2022 slope, roughness and fraction of steep cells, computed at native resolution, then aggregated to 0.25° over land only.
- **Above-ground wind.** Over high terrain the 850 hPa level is below ground, so each cell uses the nearest above-ground level:

  | Cell elevation | Wind level used |
  |---|---|
  | below 1400 m | 850 hPa |
  | up to 1950 m | 800 hPa |
  | up to 2470 m | 750 hPa |
  | above that | 700 hPa |

**Uprising test, 3–7-day tracks (FAR 60 vs NEAR 65):**

| Metric (footprint median) | FAR | NEAR | p (FAR > NEAR) | Cliff's δ |
|---|---|---|---|---|
| Max slope (m/km) | 217 | 116 | 7×10⁻⁵ | +0.39 |
| Roughness (m) | 112 | 38 | 4×10⁻⁶ | +0.46 |
| Fraction of steep cells | 0.48 | 0.07 | 8×10⁻⁶ | +0.45 |
| Wind-uphill alignment (−1..1) | 0.30 | 0.08 | 1×10⁻³ | +0.32 |
| Ascent on steep cells (cm/s) | 3.25 | −0.16 | 8×10⁻⁶ | +0.50 |
| Fraction of days ascending | 0.88 | 0.50 | 9×10⁻⁴ | +0.32 |

The median FAR slope of 217 m/km sits at the 87th percentile of all Indian land slopes.

**Spatial relocation null** (FAR tracks kept rigid and slid to random positions over India, 5,000 shuffles):

| Metric | Observed | Null mean | SDs above null |
|---|---|---|---|
| Slope (m/km) | 217 | 93 | 12.4 |
| Roughness (m) | 112 | 41 | 15.7 |
| Fraction steep | 0.48 | 0.05 | 18.7 |
| Uphill alignment (A_grad) | 0.28 | 0.00 | 4.3 |

**Honest caveat.** A separate *day-shuffle* null keeps each location but draws the wind from random days. It puts the uphill-wind alignment only 0.7 SD above shuffled days (p = 0.25). So the terrain affinity of FAR tracks is robust, but the uphill flow reflects the **climatological** monsoon state rather than event-specific forcing.

<p align="center"><img src="../../figures/cp6_near_far_tracks_relief.png" width="560" alt="NEAR vs FAR tracks over relief"></p>
<p align="center"><img src="../../figures/cp6_orographic_metrics_far_vs_near.png" width="720" alt="Orographic metrics, FAR vs NEAR"></p>

## Conclusions

1. **Multi-day ERE tracks are real weather systems.** They co-locate with independent depression centres far more than chance, with the expected south-west rain offset.
2. **Tracks far from any LPS are orographic.** They are slow and anchored to steep terrain (Western Ghats, Himalayan foothills, north-east hills), under climatological upslope monsoon flow.

## Open items

- Re-run the relocation null with the multi-level above-ground wind.
- Moisture-flux version of the uphill metric (q·V·∇h instead of the dry V·∇h).
