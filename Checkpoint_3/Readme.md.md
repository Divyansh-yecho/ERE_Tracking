# Checkpoint 3 — ERE Detection and Tracking at 0.25° Resolution

**ERA5 Reanalysis | Indian Monsoon EREs | 1979–2024**

---

## Overview

Checkpoint 3 is the most significant upgrade in the ERE Detection and Tracking Pipeline. It addresses the two critical failures identified in Checkpoint 2 — the 1° resolution overlap problem that made tracking nearly impossible, and the June NT anomaly that produced physically incorrect seasonal cycles. It introduces a new 46-year ERA5 dataset at 0.25° resolution, a spatially constrained India-only detection domain, and a physically grounded tracking algorithm that achieves a 44.6% continuation rate compared to CP2's 9.9%.

---

## What CP3 Solves

### Problem 1 — 97.7% Zero Overlap in Tracking (CP2)

**The problem:** At 1° resolution, a monsoon system covering 300×300 km spans roughly 9 grid cells. Moving 200 km overnight produces a 2-cell shift — zero spatial overlap between consecutive days. CP2's tracking was almost entirely dependent on a distance-based fallback that linked EREs by centroid proximity alone, with no spatial grounding.

**The fix:** Moving to 0.25° resolution means the same 300×300 km system now spans ~121 grid cells. Moving 200 km overnight produces a 7-cell shift with ~80 shared cells — 66% overlap. The 97.7% zero-overlap rate essentially disappears.

**Result:**
| Metric | CP2 (1°) | CP3 (0.25°) |
|---|---|---|
| Continuation rate | 9.9% | 44.6% |
| Single-day tracks | 70% | 5.7% |
| Mean track duration | 1.43 days | 2.69 days |
| Max track duration | 10 days | 16 days |

---

### Problem 2 — June NT Anomaly

**The problem:** In CP1 and CP2, June had higher NT (total extreme grid cells per day) than July despite July being the peak Indian monsoon month. This was initially attributed to lower rainy-day counts in June producing a lower 99th percentile threshold.

**The investigation:** At 0.25° with a single annual threshold and the full ERA5 domain (5N–40N, 60E–100E), the June anomaly persisted even with the correct threshold. A systematic investigation revealed:

- Overall exceedance rates: June 1.06%, July 0.93% — June was genuinely higher
- Mid-threshold cells (13.99–57.13 mm/day, 60% of domain): June exceedance 1.199% vs July 0.905%
- At June-winning cells, mean precip on exceeding days: June 58.27 mm/day vs July 56.15 mm/day

The root cause was the heterogeneous ERA5 domain — ocean cells, Pakistan desert, and arid fringe cells all had low thresholds (floored at 50 mm/day). The Indian monsoon onset in June activates these low-threshold cells across the entire domain, inflating June NT above July NT even though July is more intense over Indian land.

**The fix:** Restricting the detection domain to Indian land cells only (using regionmask Natural Earth country boundaries, India index 98). This removes ocean, Pakistan, and desert fringe cells entirely.

**Result:** July NT (18.3) > June NT (13.4) — anomaly fully resolved.

---

### Problem 3 — Short Record for Trend Significance (CP2: 23 years)

**The problem:** CP2 used 1998–2020 (23 years). NT, NE, and S-bar trends were directionally correct but statistically insignificant (p > 0.05) — record too short.

**The fix:** New ERA5 dataset covers 1979–2024 (46 years) — nearly double the record length. 1979 is the standard ERA5 start year due to satellite data assimilation.

---

## Dataset

| Property | CP2 | CP3 |
|---|---|---|
| Source | ERA5 reanalysis | ERA5 reanalysis |
| Resolution | 1° (~111 km) | 0.25° (~27 km) |
| Period | 1998–2020 | 1979–2024 |
| JJAS days | 2,806 | 5,612 |
| Grid shape | (2806, 36, 41) | (5612, 141, 161) |
| Domain | 5N–40N, 60E–100E | 5N–40N, 60E–100E |
| Detection domain | Full domain | India-only (4,452 cells) |

**Download:** 46 yearly NetCDF files (~11 MB each), Kaggle dataset `era5-0-25-daily-mmday`. Each file contains JJAS daily total precipitation in mm/day (ERA5 `tp` variable, hourly accumulation summed × 1000).

---

## Detection Pipeline

### Step 1 — India Mask
```python
countries  = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
india_raw  = countries.mask(lon, lat)
india_mask = (india_raw.values == 98)   # 4,452 cells
```
Restricts all threshold computation and ERE detection to Indian land cells. Eliminates ocean, Pakistan, and arid fringe cells that caused the June anomaly.

### Step 2 — Threshold Computation (Nikumbh method)
```
- Rainy day definition  : precipitation > 1 mm/day
- Percentile            : 99th (computed on India cells only)
- Floor                 : max(99th percentile, 50 mm/day)
- Non-India cells       : threshold set to 999 mm/day (never exceed)
```

**Threshold statistics over India:**
- Min: 50.0 mm/day (rain shadow zones at floor)
- Max: 217.8 mm/day (Western Ghats windward)
- Mean: 70.5 mm/day
- Cells at floor: 1,166 / 4,452 (26.2%) — rain shadow zones

**Note on percentile choice:** Nikumbh et al. (2019) used the 99.5th percentile. CP3 uses the 99th percentile to produce a sufficient number of ERE objects for meaningful tracking (NE = 2.65/day at 99th vs 1.75/day at 99.5th). This is a documented methodological divergence.

### Step 3 — Exceedance Mask
```
c = (data_precip > threshold_3d)    # (5612, 141, 161) boolean
```
Only India cells can ever be True. Non-India cells have threshold = 999 mm/day.

**Exceedance rates (India cells only):**
| Month | Rate |
|---|---|
| June | 0.531% |
| July | 0.787% |
| August | 0.627% |
| September | 0.377% |
| Overall | 0.582% |

July correctly highest — June anomaly resolved.

### Step 4 — Connected Component Labelling
```python
struct4 = ndimage.generate_binary_structure(2, 1)   # 4-connectivity
labeled, nr_objects = ndimage.label(img, structure=struct4)
```
4-connectivity (no diagonal connections) — matches Nikumbh et al. (2019). At 0.25° the diagonal distance between cell centres is ~38 km, which is close to cell width (~27 km), but we maintain 4-connectivity for direct methodological comparability with Nikumbh.

Each connected component = one ERE object. `Loc_ERE` stores the size (cell count) of each ERE at every cell. `Label_ERE` stores integer labels for tracking.

**Detection statistics:**
- Mean NE (ERE objects/day): 2.65
- Mean NT (extreme cells/day): 25.93
- Mean S-bar: 10.66 cells
- Max NE one day: 19
- Zero-ERE days: 19.7%

---

## ERE Size Categories

Nikumbh et al. (2019) defined size categories by cell count at 1° resolution. At 0.25° these must be rescaled by physical area to be comparable:

| Category | Nikumbh (1°) | CP3 equivalent (0.25°) | Physical area |
|---|---|---|---|
| Small | 1 cell | 1–15 cells | < 12,300 km² |
| Medium | 2–5 cells | 16–90 cells | 12,300–70,000 km² |
| Large | ≥6 cells | ≥91 cells | ≥70,000 km² |

**Important:** Direct cell-count comparison between CP3 and Nikumbh is invalid. Only physical area comparisons are meaningful.

**CP3 size distribution (by physical area, at ERE peak size):**
| Category | Count | % of tracks |
|---|---|---|
| Small (<12,300 km²) | 6,472 | 78.6% |
| Medium (12,300–70,000 km²) | 1,657 | 20.1% |
| Large (≥70,000 km²) | 110 | 1.3% |

---

## Tracking Algorithm

### Parameters
| Parameter | CP2 | CP3 | Justification |
|---|---|---|---|
| Resolution | 1° | 0.25° | Fixes overlap problem |
| Overlap threshold | 15% | 15% | Sullivan et al. (2019) |
| Overlap denominator | min(A,B) | min(A,B) | Unchanged |
| Fallback | Distance ≤ 2° | Search box 2.5° + closest centroid | Data-driven: 95th pct of actual displacements = 2.14° |
| Fallback tiebreak | Random (sentinel 1e-6) | Nearest by centroid distance | Physically grounded |
| Gap tolerance | 0 days | 1 day | Brief threshold excursions |
| Centre of mass | Unweighted | Unweighted geometric | Consistent, simple |
| Season break | Sept 30 | Sept 30 | No cross-year tracks |
| Domain exit | Index-based boundary | India mask boundary cells | Physically correct |

### Search Box Derivation
The 2.5° search box was derived from actual data — not assumed:
```
Day-to-day centroid displacement (overlapping EREs only):
  mean     : 0.795°
  median   : 0.613°
  90th pct : 1.676°
  95th pct : 2.140°   ← search box set to 2.5° (95th pct + buffer)
  max      : 4.806°
```

### Algorithm Flow (per day pair t, t+1)
```
1. OVERLAP — build n×m overlap matrix between active EREs at t and EREs at t+1
             assign greedily (strongest overlap first, 1-to-1)
             threshold: shared_cells / min(|A|,|B|) ≥ 0.15

2. FALLBACK — for unmatched EREs from t, search within 2.5° for closest
              unmatched ERE at t+1 by centroid distance

3. GAP REVIVAL — check dormant tracks (1-day gap) against unmatched t+1 EREs
                 within 2.5° search box

4. RECORD — continuations (matched), births (unmatched t+1),
            deaths/domain_exits (unmatched t after gap expiry)

5. SEPT 30 — hard season break, all active and dormant tracks terminated
```

### Domain Exit Detection
An ERE is classified as `domain_exit` (not `death`) if any of its cells touches an India mask boundary cell — defined as an India cell with at least one non-India neighbour in 4-connectivity. This correctly identifies systems leaving the Indian domain rather than simply dissipating.

---

## Tracking Results

| Metric | Value |
|---|---|
| Total ERE-day records | 23,102 |
| Unique tracks | 8,239 |
| Active records (births + continuations) | 14,864 |
| Single-day tracks | 470 (5.7%) |
| 2-day tracks | 4,237 |
| 3–5 day tracks | 3,247 |
| ≥6 day tracks | 285 |
| Max track duration | 16 days |
| Mean track duration | 2.69 days |
| Continuation rate | 44.6% |
| Domain exits | 2,794 records |
| Track fragmentation index | 0.554 |

---

## Validation

### Category 1 — Internal Consistency
| Check | Result | Target | Status |
|---|---|---|---|
| Continuation rate | 44.6% | > 20% | ✓ PASS |
| Mean track duration | 2.69d | > 2.0d | ✓ PASS |
| Single-day fraction | 5.7% | < 60% | ✓ PASS |
| Birth count = unique tracks | 8,239 = 8,239 | Equal | ✓ PASS |
| No season boundary crossings | 0 | 0 | ✓ PASS |

### Category 2 — Physical Consistency
| Check | Result | Target | Status |
|---|---|---|---|
| Active/weak year ratio | 1.64 | ≥ 1.2 | ✓ PASS |
| Peak continuation month | August | July or August | ✓ PASS |
| Highest long-track region | Central India (26.2%) | Central/NE India | ✓ PASS |
| Lowest long-track region | Pakistan/Desert (0.0%) | Pakistan/Desert | ✓ PASS |

### Category 3 — Known Event Validation
| Event | Detected | Multi-day tracked | Max track duration |
|---|---|---|---|
| Uttarakhand 2013 | YES ✓ | YES ✓ | 3 days |
| Kerala 2018 | YES ✓ | YES ✓ | 4 days |
| Assam 2012 | YES ✓ | YES ✓ | 5 days |
| Bihar floods 2007 | YES ✓ | YES ✓ | — |
| Mumbai 2005 | YES ✓ | YES ✓ | — |
| Leh cloudburst 2010 | NO ✗ | — | Expected: ERA5 cannot resolve localised mountain cloudbursts |

---

## Trend Analysis (Detection Level)

NT, NE, and S-bar computed annually over Central India (15–25N, 75–85E) following Nikumbh et al. (2019). An ERE belongs to Central India if at least one cell lies within the box.

### With All EREs (including small)
| Metric | Slope | p-value | Significant | Nikumbh finding |
|---|---|---|---|---|
| NT | +5.17/yr | 0.22 | ✗ | Significant increase |
| NE | +0.77/yr | 0.0015 | ✓ | Flat |
| S-bar | -0.046/yr | 0.13 | ✗ | Significant increase |

NE is significantly increasing — **opposite of Nikumbh**. Investigation shows this is driven entirely by small EREs (<12,300 km²) proliferating at +1.43/year (p=0.0019). Medium and large NE are completely flat.

### Medium + Large EREs Only (≥12,300 km²)
| Metric | Slope | p-value | Significant | Nikumbh finding |
|---|---|---|---|---|
| NT | +2.60/yr | 0.49 | ✗ | Significant increase |
| NE | +0.038/yr | 0.66 | ✗ | Flat ✓ |
| S-bar | +0.063/yr | 0.41 | ✗ | Significant increase |

Directional agreement with Nikumbh: NT increasing, NE nearly flat, S-bar increasing. Statistical significance not reached over 46 years.

### Interpretation
ERA5 at 0.25° resolves small convective EREs (<12,300 km²) that are invisible to IMD gauge data at 1°. These small systems are proliferating significantly — likely a combination of real atmospheric change and ERA5 data quality improvement over time (better observational constraint in recent decades). When physically comparable size ranges are used (medium+large), the Nikumbh finding is directionally replicated but significance requires a longer record or lower noise.

---

## Known Problems and Limitations

| # | Problem | Severity | Fix planned |
|---|---|---|---|
| 1 | Small ERE proliferation masks S-bar trend | Critical | Use medium+large only for CP5 trend analysis |
| 2 | Statistical significance not reached | High | More years + medium+large only |
| 3 | ERA5 spatial smoothing limits large ERE detection | High | Document as ERA5 limitation |
| 4 | ERA5 inhomogeneity 1979–2024 | High | Sensitivity test by sub-period |
| 5 | Size categories not comparable to Nikumbh directly | Medium | Physical area rescaling documented |
| 6 | Merges and splits not handled | Medium | CP4 |
| 7 | 2024 outlier (NT=5590 vs mean ~3000) | Medium | Sensitivity test without 2024 |
| 8 | Domain exit direction check fails (39.2% vs target 50%) | Low | Check invalid for irregular India boundary |
| 9 | Centre of mass unweighted | Low | CP4 rainfall-weighted CoM |

---

## File Outputs

| File | Description |
|---|---|
| `Loc_ERE.npy` | (5612, 141, 161) float32 — ERE size at each cell, NaN = no ERE |
| `Label_ERE.npy` | (5612, 141, 161) int32 — integer ERE labels per day |
| `NT_daily.npy` | (5612,) int32 — total extreme cells per day |
| `NE_daily.npy` | (5612,) int32 — number of ERE objects per day |
| `per_grid_threshold.npy` | (141, 161) float32 — detection threshold per cell |
| `india_mask.npy` | (141, 161) bool — True = India land cell |
| `lat.npy` / `lon.npy` | (141,) / (161,) float32 — coordinate arrays |
| `time.npy` | (5612,) datetime64 — time coordinate |
| `ere_tracks_cp3.csv` | Full tracking dataframe — all events |
| `track_summary_cp3.csv` | One row per track — duration, max size, distance |

---

## Dependencies

```
numpy
pandas
xarray
scipy
regionmask
cartopy
matplotlib
h5netcdf
```

---

## Next Steps

- **CP4** — Merge and split detection using bipartite graph on overlap matrix
- **CP5** — Formal trend analysis: NT, NE, S-bar over 1979–2024, Mann-Kendall test, Nikumbh replication
- **CP6** — Presentation: ERA5 vs IMD comparison, resolution effects, scientific findings

---

## Reference

Nikumbh, A.C., Chakraborty, A. & Bhat, G.S. (2019). Recent spatial aggregation tendency of rainfall extremes over India. *Scientific Reports*, 9, 10321. https://doi.org/10.1038/s41598-019-46719-2
