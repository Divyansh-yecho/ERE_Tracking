# Checkpoint 2 — Known Limitations and Open Problems

This document records all known limitations and open problems in the current
ERE detection and tracking pipeline as of Checkpoint 2. These are not bugs
(except where noted) but fundamental constraints of the current design that
will be addressed in future checkpoints.

---

## 1. Spatial Resolution Mismatch (Critical)

**What:** ERA5 data was downloaded at 1° resolution (~111 km per grid cell).
At this resolution, a rainfall system moving even 100 km overnight completely
vacates its previous grid cell, producing zero spatial overlap between
consecutive-day ERE footprints.

**Impact:** 97.7% of all consecutive ERE pairs share zero grid cells.
The primary overlap criterion — the physically meaningful linking rule —
fires in only 2.3% of cases. The entire tracking result therefore rests on
the fallback distance criterion rather than genuine spatial continuity.

**Root cause:** Deliberate choice during data acquisition to reduce file size
and computation time. ERA5's native resolution is 0.25° (~28 km per cell).

**Planned fix:** Re-download ERA5 at 0.25° resolution. At native resolution
each system spans ~16× more grid cells and consecutive-day overlap becomes
far more common, making the overlap criterion physically meaningful.

---

## 2. Fallback Criterion Not Physically Grounded (High)

**What:** When cell overlap is zero, the tracker falls back to linking two
EREs if their centres of mass are within 2° and their size ratio is ≥ 0.5.
This is a distance-only check with no spatial relationship requirement.

**Impact:** Two completely unrelated EREs anywhere within ~220 km that happen
to be similar in size will be linked as the same track. Unlike Sullivan et al.
(2019) where the search box defined candidates but overlap was still required,
the current fallback requires no spatial overlap at all.

**Planned fix:** Replace distance fallback with an adjacency check — link two
zero-overlap EREs only if any cell of ERE A directly shares an edge (4-connectivity)
with any cell of ERE B on the next day. This ensures footprints are literally
touching rather than merely nearby.

---

## 3. Overlap Threshold Not Calibrated for This Grid (Medium)

**What:** The 15% overlap threshold was inherited from Sullivan et al. (2019)
who worked at ~30 km resolution with large MCS covering hundreds of pixels.
At 1° resolution the threshold behaves differently — a size-2 ERE needs 50%
overlap to pass, while a size-20 ERE needs only 5%.

**Impact:** The threshold is simultaneously too strict for large EREs and
potentially too loose for small ones. It was never calibrated for a coarse
discrete grid.

**Planned fix:** Sensitivity analysis — rerun tracking with thresholds of
10%, 15%, and 20% and compare track statistics. Select the value that
produces the most physically consistent results.

---

## 4. No Gap Tolerance (Medium)

**What:** The tracker requires EREs to overlap or be adjacent on strictly
consecutive days. A system that produces extreme rainfall on day 1 and day 3
but dips below the 99th percentile threshold on day 2 is recorded as two
separate single-day tracks rather than one continuous 3-day track.

**Impact:** Genuine multi-day rainfall systems are artificially fragmented
when they briefly fall below the detection threshold. This inflates birth and
death counts and underestimates true track durations.

**Planned fix:** Allow a gap tolerance of 1 day — if an active track finds
no match on day t+1 but finds a valid match on day t+2, keep the track alive
across the gap and record the gap day explicitly.

---

## 5. Merge and Split Events Deferred and Currently Misclassified (Medium)

**What:** The greedy 1-to-1 assignment forces all N-to-1 (merge) and 1-to-N
(split) cases into either a single continuation link or independent birth/death
events. Real merge and split events are silently absorbed or broken into
fragments.

**Impact:** Track counts are inflated (splits create false births) and track
durations are underestimated (merges terminate one track prematurely).
The physical lifecycle of EREs — particularly the role of merging during
active monsoon phases — cannot be studied until this is resolved.

**Planned fix:** Checkpoint 4 will implement explicit merge and split
detection using the existing bipartite overlap matrix infrastructure.
N-to-1 patterns will be classified as merges, 1-to-N as splits, and
N-to-N as complex events.

---

## 6. Unweighted Centre of Mass (Low)

**What:** The centre of mass of each ERE is computed as the simple unweighted
mean of the lat/lon indices of its member cells. For irregularly shaped EREs
this can place the centre outside the actual ERE footprint.

**Impact:** Centre-of-mass distance metrics (step_dist_deg, track_distance,
mean_velocity) are less physically meaningful than they would be with a
rainfall-weighted centre. The intense rain core may be far from the geometric
centre for asymmetric EREs.

**Planned fix:** Compute rainfall-weighted centre of mass using daily
precipitation values at each ERE cell as weights.

---

## 7. Domain Exit Detection is Index-Based (Low)

**What:** Domain exits are flagged when any cell of a dying ERE touches the
boundary indices of the lat/lon arrays (index 0 or max index). All boundary
contact is treated equally regardless of ERE size or direction of movement.

**Impact:** A large ERE that spans most of the domain and incidentally touches
a boundary is classified the same as a small ERE genuinely propagating off the
grid. No distinction is made between EREs moving toward versus away from the
boundary.

**Planned fix:** Add a directional check — compute the step_dist_deg vector
and flag domain_exit only when the ERE centre is moving toward the boundary
it is touching.

---

## 8. Known Event Validation Not Repeated (Low)

**What:** Chapter 1 validated the detection pipeline against 4 known extreme
rainfall events (Mumbai 2005, Uttarakhand 2013, Kerala 2018, Odisha 2001).
Only 1 of 4 was detected. Chapter 2 introduced monthly thresholds and tracking
but did not repeat this validation.

**Impact:** It is unknown whether the corrected monthly thresholds improve
detection of the 3 previously missed events. Uttarakhand 2013 was detected in
Chapter 1 — whether it now forms a valid multi-day track has not been verified.

**Planned fix:** Re-run known event validation in Checkpoint 3 with the
corrected monthly thresholds and check whether detected events form
physically plausible tracks.

---

## 9. Nikumbh Replication Statistically Underpowered (Expected)

**What:** NT, NE, and S-bar all trend in the correct direction (rising) over
1998–2020 but none reach p < 0.05 significance.

**Impact:** The core Nikumbh et al. (2019) finding — that mean ERE size S-bar
has significantly increased — cannot be independently confirmed from the
current 23-year record.

**Why expected:** Nikumbh established the trend over a 64-year record
(1951–2015). A 23-year window starting in 1998 captures only the continuation
of the trend, not its full extent. Statistical significance requires a longer
baseline. This is a dataset limitation, not a methodological failure.

**No fix planned:** This limitation is inherent to ERA5 availability
(1979 onwards) and the project scope (1998–2020 monsoon seasons).
Results are reported as directionally consistent with Nikumbh.

---

## Summary Table

| # | Problem | Severity | Status |
|---|---------|----------|--------|
| 1 | 1° resolution — overlap almost never fires | Critical | Planned — re-download at 0.25° |
| 2 | Fallback not spatially grounded | High | Planned — replace with adjacency |
| 3 | Overlap threshold not calibrated | Medium | Planned — sensitivity analysis |
| 4 | No gap tolerance | Medium | Planned — allow 1-day gap |
| 5 | Merges and splits misclassified | Medium | Checkpoint 4 |
| 6 | Unweighted centre of mass | Low | Planned — rainfall weighting |
| 7 | Domain exit detection index-based | Low | Planned — directional check |
| 8 | Known event validation not repeated | Low | Checkpoint 3 |
| 9 | Nikumbh trend not significant | Expected | Dataset limitation |

---

*Last updated: Checkpoint 2 — Detection and Tracking*
*Next: Checkpoint 3 — Validation and resolution fix*
