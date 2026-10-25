# Lagrangian Tracking of Indian Monsoon Extreme Rainfall Events

A Python pipeline that detects **extreme rainfall events (EREs)** over India during the summer monsoon (June–September), follows them from day to day as moving objects, including **merges and splits**, and then asks what physically produces the multi-day tracks and how they are organised in space and time.

*Summer research project, IIT Bombay, with Prof. Akshaya Nikumbh (May–Aug 2026).*

---

## Highlights

- **Tracker.** Per-cell 99th-percentile thresholds (floored at 50 mm/day) define the extremes. Connected objects are linked across days by spatial overlap, and merges, splits, continuations, births and deaths are recorded explicitly, with no distance-based "teleport" links.
- **ERA5, 1979–2024 (CP3–CP5).** At 1°, 97.7% of consecutive objects share no grid cell, so overlap tracking fails; at 0.25°, 44.6% of active records are continuations. The final mean-overlap tracker finds 10,965 tracks with 173 merges and 124 splits.
- **IMERG, 2000–2024 (CP6).** 10,006 tracks and 10,877 object-days. A stricter two-sided merge/split gate (CP7) keeps 28 merges and 17 splits of the 71 and 51 found by the mean-overlap rule.
- **Multi-day tracks are real weather systems.** They sit much closer to independent monsoon **low-pressure system (LPS)** centres than rain objects overall (median 427 vs 737 km), at **3–4× chance** in a track-level re-dating null, with the rain ~200 km west of the depression centre, as reported in the literature.
- **Tracks far from any LPS** sit on the steep monsoon margins (Western Ghats, Himalayan foothills, north-east hills) and move slowly (75 vs 124 km/day). This is consistent with terrain-anchored rain, but the terrain contrast is largely geographic and the uphill wind is the climatological monsoon flow, so orographic forcing is not proven.
- **Regional patterns and life cycles (CP7).** Size controls survival beyond overlap geometry (59% of 50+ cell objects survive to the next day vs 11% by chance); bigger objects are rainier per cell; active monsoon spells carry ~3× the ERE activity and ~6× the multi-day births of breaks, and only the Northeast gains extreme rain during breaks; near-LPS tracks are twice as large but not more intense; merges grow rainfall (1.34×) and splits halve it (0.50×).
- **Spells validated (CP8).** The rainfall-defined active/break spells are visible in independent satellite OLR (r = −0.76) and active spells cluster at BSISO1 phase 5.

<p align="center">
  <img src="figures/cp6_near_far_tracks_relief.png" width="620" alt="3-7 day ERE tracks over terrain: NEAR (blue) vs FAR (red) from LPS">
</p>

---

## Repository structure

```
ERE_Tracking/
├── README.md                     ← this file
├── requirements.txt
├── data/README.md                ← datasets, where to get them, files the scripts write
├── figures/                      ← all figures; the cpN_ prefix names the checkpoint
└── checkpoints/                  ← one folder per research checkpoint, in order
    ├── CP1_datasets_and_thresholds/
    ├── CP2_tracking_era5_1deg/
    ├── CP3_tracking_era5_0p25deg/
    ├── CP4_connectivity_and_size_filter/
    ├── CP5_merge_split_tracking/
    ├── CP6_imerg_lps_orography/
    ├── CP7_regional_trend_analysis/
    └── CP8_spells_olr_bsiso/
```

Every checkpoint folder has its own `README.md`.
- **CP1–CP6:** each Kaggle notebook (`.ipynb`, with all outputs and figures) sits next to a `.py` script converted from it, with the cells in the same order (`# %%` markers).
- **CP6–CP8:** standalone scripts rebuild the IMERG catalogue locally and regenerate every IMERG figure and number from four input files.

---

## Checkpoints

| # | Folder | What it does | Key outcome |
|---|---|---|---|
| CP1 | [`CP1_datasets_and_thresholds`](checkpoints/CP1_datasets_and_thresholds) | ERA5 download (CDS API), daily JJAS precipitation, per-cell extreme thresholds | Daily 1° dataset 1998–2020 |
| CP2 | [`CP2_tracking_era5_1deg`](checkpoints/CP2_tracking_era5_1deg) | First detection and tracking at 1°; NT / NE / S-bar trends | 97.7% of day-to-day pairs share no cells at 1°, so resolution is the bottleneck |
| CP3 | [`CP3_tracking_era5_0p25deg`](checkpoints/CP3_tracking_era5_0p25deg) | 0.25°, 1979–2024, India-only mask, search-box fallback, gap tolerance | Continuation rate 44.6%; June anomaly resolved; known events detected |
| CP4 | [`CP4_connectivity_and_size_filter`](checkpoints/CP4_connectivity_and_size_filter) | 4- vs 8-connectivity, ≥16-cell filter, Mann-Kendall trends | With the size filter, trends match Nikumbh et al. (2019) in direction |
| CP5 | [`CP5_merge_split_tracking`](checkpoints/CP5_merge_split_tracking) | Unified tracker with explicit merges and splits; MAX vs MIN vs MEAN overlap; fallback removed | Final MEAN tracker: 10,965 tracks, 173 merges, 124 splits |
| CP6 | [`CP6_imerg_lps_orography`](checkpoints/CP6_imerg_lps_orography) | Wind-steering test, IMERG tracking, LPS co-location, orographic attribution | Multi-day tracks follow LPS (3–4× chance); tracks far from LPS sit on steep terrain and move slowly |
| CP7 | [`CP7_regional_trend_analysis`](checkpoints/CP7_regional_trend_analysis) | Survival, intensity scaling, active/break spells, regions and trends, life cycles, merge/split budget, temporal clustering | Size controls survival; active spells triple ERE activity; near-LPS tracks are larger, not more intense |
| CP8 | [`CP8_spells_olr_bsiso`](checkpoints/CP8_spells_olr_bsiso) | Spells tested against CPC OLR and the APCC BSISO index | Spells confirmed in OLR; active spells phase-locked to the BSISO, breaks only weakly |

---

## Method in brief

1. **Threshold.** For each 0.25° land cell, the 99th percentile of wet-day (>1 mm) rainfall, with a 50 mm/day floor. Cells outside India never exceed.
2. **Detect.** Daily exceedance cells grouped into connected objects (4-connectivity in CP3–CP4, 8-connectivity from CP5).
3. **Link.** Objects on consecutive days are linked by overlap. The order each day is merges → splits → continuations → deaths → births. Merges and splits need both overlap fractions to clear 5% from CP7 on (mean of the two ≥ 0.10 in CP5–CP6). A track ends when most of it leaves India, and seasons reset on 30 September.
4. **Validate.** Same-date nearest-LPS distance (Vishnu et al. LPS catalogue), a track-level re-dating null and a NEAR / FAR / NO-LPS classification (CP6); spells checked against OLR and the BSISO (CP8).
5. **Attribute.** ETOPO 2022 slope, roughness and steep fraction under each track footprint, an above-ground per-cell wind for uphill-flow metrics, Mann-Whitney / Cliff's δ tests, spatial-relocation and day-shuffle nulls (CP6c).
6. **Characterise.** Survival vs size against a geometric null, threshold-relative intensity, active/break composites, regions, life cycles, merge/split rainfall budgets, temporal clustering (CP7).

---

## Getting started

```bash
git clone https://github.com/Divyansh-yecho/ERE_Tracking.git
cd ERE_Tracking
pip install -r requirements.txt
```

**Notebooks (CP1–CP6)** were run on **Kaggle** (inputs under `/kaggle/input/...`). To run them elsewhere, download the inputs listed in [`data/README.md`](data/README.md) and change the input paths at the top of each notebook or script. CP1 needs your own Copernicus CDS API key in `~/.cdsapirc`.

**Local scripts (CP6–CP8).** Put the four input files listed in [`data/README.md`](data/README.md) into `data/` (or set `ERE_DATA` to their folder), then run from the repository root:

```bash
python checkpoints/CP6_imerg_lps_orography/cp6_build_imerg_catalogue.py
python checkpoints/CP6_imerg_lps_orography/cp6_lps_figures.py
python checkpoints/CP6_imerg_lps_orography/cp6_lps_null.py
for f in checkpoints/CP7_regional_trend_analysis/cp7_0*.py; do python "$f"; done
for f in checkpoints/CP8_spells_olr_bsiso/cp8_0*.py; do python "$f"; done
```

Each script prints its results and writes its figures to `figures/`; intermediate tables go to `data/derived/`. The whole chain takes about 10 minutes on a laptop.

---

## Data

| Dataset | Used in | Purpose |
|---|---|---|
| ERA5 total precipitation (1°, 1998–2020) | CP1–CP2 | First detection and tracking |
| ERA5 total precipitation (0.25°, 1979–2024) | CP3–CP5, CP6a | Main ERA5 tracking |
| NASA IMERG daily precipitation (0.25°, 2000–2024) | CP6–CP8 | Satellite-based tracking |
| ERA5 winds at 850 / 800 / 750 / 700 hPa | CP6 | Steering test and uphill-flow metrics |
| Vishnu et al. monsoon LPS track catalogue | CP6–CP7 | Independent depression centres |
| ETOPO 2022 relief | CP6c | Terrain metrics |
| CPC blended OLR (2.5°) and APCC BSISO index | CP8 | Independent check of the active/break spells |

Full details are in [`data/README.md`](data/README.md).

---

## References

- Nikumbh, A. C., Chakraborty, A. & Bhat, G. S. (2019). Recent spatial aggregation tendency of rainfall extremes over India. *Scientific Reports*, 9, 10321.
- Sullivan et al. (2019): the 15% overlap-tracking criterion adopted in CP2–CP3.
- Vishnu, S., Boos, W. R., Ullrich, P. A. & O'Brien, T. A. (2020). Global track catalogue of monsoon low-pressure systems. Zenodo, doi:10.5281/zenodo.3890646.
- Boos et al. (2023), *npj Climate and Atmospheric Science*, and Hunt, K. M. R. et al. (2016), *Monthly Weather Review*: rainfall displaced south-west of monsoon-depression centres.
- Rajeevan, M., Gadgil, S. & Bhate, J. (2010). Active and break spells of the Indian summer monsoon. *Journal of Earth System Science*, 119, 229–247.
- Lee, J.-Y., Wang, B., Wheeler, M. C., Fu, X., Waliser, D. E. & Kang, I.-S. (2013). Real-time multivariate indices for the boreal summer intraseasonal oscillation over the Asian summer monsoon region. *Climate Dynamics*, 40, 493–509.
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146, 1999–2049.
- Huffman, G. J. et al. NASA GPM Integrated Multi-satellite Retrievals for GPM (IMERG).
- NOAA NCEI (2022). ETOPO 2022 15 Arc-Second Global Relief Model.

---

**Author:** Divyansh Yecho, BS-MS (Physics major, Data Science minor), IISER Pune · [GitHub](https://github.com/Divyansh-yecho)
