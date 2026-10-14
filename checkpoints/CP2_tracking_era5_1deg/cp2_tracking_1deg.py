"""
CP2 - ERE detection and first tracker (ERA5, 1 deg)

Project : Lagrangian tracking of Indian monsoon extreme rainfall events (ERE_Tracking)
Source  : cp2_tracking_1deg.ipynb (converted cell by cell; figures and printed results are in the notebook)

Detects daily ERE objects (connected exceedance cells), links them across days with an
overlap rule plus a distance fallback, and computes NT / NE / S-bar trends (Nikumbh et al. 2019).

Originally run on Kaggle: input paths point to /kaggle/input/... (see data/README.md at the
repository root). Notebook shell commands such as `!pip install` are kept as comments.
"""


# %% Cell 1
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# %% Cell 2
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from geopy.distance import geodesic
import warnings
warnings.filterwarnings("ignore")

print("imports done")


# %% Cell 3
ds = xr.open_dataset('/kaggle/input/datasets/divyanshyecho/era5-india-daily-precip/era5_india_daily_correct.nc')

data_precip = np.array(ds['precipitation'][:])
lat         = ds['lat'].values
lon         = ds['lon'].values
time        = ds['time'].values

print(f"shape      : {data_precip.shape}")
print(f"lat        : {lat.min()} to {lat.max()}")
print(f"lon        : {lon.min()} to {lon.max()}")
print(f"time       : {str(time[0])[:10]} to {str(time[-1])[:10]}")
print(f"max precip : {np.nanmax(data_precip):.2f} mm/day")
print(f"mean precip: {np.nanmean(data_precip):.2f} mm/day")


# %% Cell 4
times_pd = pd.to_datetime(time)

# mask dry days
data_precip_masked = np.copy(data_precip)
data_precip_masked[data_precip_masked <= 0.0] = np.nan

# separate threshold per grid per calendar month
# shape: (4, 36, 41) — one per month June/July/Aug/Sept
months     = [6, 7, 8, 9]
per_grid_monthly = np.full((4, len(lat), len(lon)), np.nan, dtype='float32')

for m_idx, month in enumerate(months):
    month_mask = (times_pd.month == month)
    month_data = data_precip_masked[month_mask, :, :]  # (days_in_month, 36, 41)

    for x in range(len(lat)):
        for y in range(len(lon)):
            col = month_data[:, x, y]
            col = col[~np.isnan(col)]
            if len(col) > 0:
                per_grid_monthly[m_idx, x, y] = np.percentile(col, 99)

print("monthly threshold shape :", per_grid_monthly.shape)
for m_idx, month in enumerate(months):
    mn = np.nanmin(per_grid_monthly[m_idx])
    mx = np.nanmax(per_grid_monthly[m_idx])
    mu = np.nanmean(per_grid_monthly[m_idx])
    print(f"  month {month} — min {mn:.1f}  max {mx:.1f}  mean {mu:.1f} mm/day")


# %% Cell 5
data_precip_safe = np.copy(data_precip)
data_precip_safe[np.isnan(data_precip_safe)] = 0.0

# broadcast monthly threshold to every day
threshold_3d = np.full_like(data_precip, np.nan, dtype='float32')

for m_idx, month in enumerate(months):
    month_mask = (times_pd.month == month)
    threshold_3d[month_mask, :, :] = per_grid_monthly[m_idx, :, :]

threshold_3d[np.isnan(threshold_3d)] = 999.0

c = (data_precip_safe > threshold_3d)

print(f"exceedance mask shape   : {c.shape}")
print(f"total extreme grid-days : {c.sum()}")
print(f"fraction exceeding      : {c.mean()*100:.2f}%")

# quick monthly sanity check — should be close to 1% each month
for month in months:
    mask = (times_pd.month == month)
    frac = c[mask].mean() * 100
    print(f"  month {month} exceedance rate : {frac:.2f}%")


# %% Cell 6
# Loc_ERE  — size stamped (existing design, used for analysis)
# Label_ERE — integer label per connected component (new, needed for tracking)

Loc_ERE   = np.full((len(time), len(lat), len(lon)), np.nan, dtype='float32')
Label_ERE = np.zeros((len(time), len(lat), len(lon)), dtype='int32')

nt_per_day    = []
n_ere_per_day = []

for k in range(len(time)):
    img = c[k, :, :].astype(int)

    labeled, nr_objects = ndimage.label(img)
    Label_ERE[k, :, :] = labeled

    nt_per_day.append(int(c[k, :, :].sum()))
    n_ere_per_day.append(nr_objects)

    if nr_objects >= 1:
        counts = np.bincount(labeled.flat)[1:]   # size of each label
        for i in range(nr_objects):
            sz  = counts[i]
            ind = np.where(labeled == i + 1)
            Loc_ERE[k, ind[0], ind[1]] = sz

    # zero → nan in Loc_ERE
    day_slice = Loc_ERE[k, :, :]
    day_slice[day_slice == 0.0] = np.nan
    Loc_ERE[k, :, :] = day_slice

    if k % 500 == 0:
        print(f"  day {k}/{len(time)}")

nt_per_day    = np.array(nt_per_day)
n_ere_per_day = np.array(n_ere_per_day)

print(f"\ndetection complete")
print(f"mean EREs per day : {n_ere_per_day.mean():.2f}")
print(f"max EREs one day  : {n_ere_per_day.max()}")
print(f"mean NT per day   : {nt_per_day.mean():.2f}")


# %% Cell 7
def get_cell_sets(label_slice):
    """
    Given a 2D Label_ERE slice for one day,
    return dict { label_id : frozenset of (lat_idx, lon_idx) }
    Label 0 = no ERE, skipped.
    """
    cell_sets = {}
    labels_present = np.unique(label_slice)
    labels_present = labels_present[labels_present > 0]
    for lbl in labels_present:
        idxs = np.argwhere(label_slice == lbl)
        cell_sets[lbl] = frozenset(map(tuple, idxs))
    return cell_sets


def centre_of_mass(cell_set, lat, lon):
    """
    Unweighted centre of mass in degrees for a frozenset of (lat_idx, lon_idx).
    """
    lat_idxs = [c[0] for c in cell_set]
    lon_idxs = [c[1] for c in cell_set]
    return lat[lat_idxs].mean(), lon[lon_idxs].mean()


def compute_overlap(cells_a, cells_b):
    """
    Overlap percentage using min denominator.
    Returns value in [0, 1].
    """
    shared = len(cells_a & cells_b)
    if shared == 0:
        return 0.0
    return shared / min(len(cells_a), len(cells_b))


def check_fallback(cells_a, cells_b, lat, lon,
                   max_dist_deg=3.0, min_size_ratio=0.5):
    """
    Fallback criterion when overlap is exactly zero.
    Returns True if centre distance <= max_dist_deg
    AND size ratio >= min_size_ratio.
    """
    size_a = len(cells_a)
    size_b = len(cells_b)

    # size similarity guard
    size_ratio = min(size_a, size_b) / max(size_a, size_b)
    if size_ratio < min_size_ratio:
        return False

    # centre of mass distance
    lat_a, lon_a = centre_of_mass(cells_a, lat, lon)
    lat_b, lon_b = centre_of_mass(cells_b, lat, lon)

    # approximate degree distance (no need for geodesic here,
    # we are comparing against a degree threshold)
    dist_deg = np.sqrt((lat_a - lat_b)**2 + (lon_a - lon_b)**2)

    return dist_deg <= max_dist_deg


def greedy_assignment(overlap_matrix, threshold=0.15):
    """
    Given an n x m overlap matrix (values in [0,1]),
    return a list of (row_idx, col_idx) pairs representing
    valid 1-to-1 assignments.

    Strategy: sort all entries above threshold descending,
    assign greedily, mark row and column as used.
    """
    n, m = overlap_matrix.shape
    entries = []
    for i in range(n):
        for j in range(m):
            if overlap_matrix[i, j] >= threshold:
                entries.append((overlap_matrix[i, j], i, j))

    # sort by overlap descending
    entries.sort(key=lambda x: x[0], reverse=True)

    used_rows = set()
    used_cols = set()
    assignments = []

    for val, i, j in entries:
        if i not in used_rows and j not in used_cols:
            assignments.append((i, j))
            used_rows.add(i)
            used_cols.add(j)

    return assignments


print("helper functions defined")


# %% Cell 8
OVERLAP_THRESHOLD   = 0.15
FALLBACK_DIST_DEG   = 2.0
FALLBACK_SIZE_RATIO = 0.5

lat_boundary = {0, len(lat) - 1}
lon_boundary = {0, len(lon) - 1}

def is_domain_exit(cell_set):
    for (li, lj) in cell_set:
        if li in lat_boundary or lj in lon_boundary:
            return True
    return False

def make_row(date, track_id, lbl, cells, k, event_type,
             overlap_pct, step_dist_deg, data_precip, lat, lon):
    lat_c, lon_c = centre_of_mass(cells, lat, lon)
    mean_rain = float(np.nanmean(
        [data_precip[k, ci, cj] for (ci, cj) in cells]))
    return {
        'date'          : date,
        'year'          : date.year,
        'month'         : date.month,
        'track_id'      : track_id,
        'day_label'     : int(lbl) if lbl is not None else -1,
        'size'          : len(cells),
        'centre_lat'    : round(float(lat_c), 2),
        'centre_lon'    : round(float(lon_c), 2),
        'mean_rain'     : round(mean_rain, 2),
        'overlap_pct'   : overlap_pct,
        'step_dist_deg' : step_dist_deg,
        'event_type'    : event_type,
    }

# ── main tracking loop ──────────────────────────────────────────
track_id_counter = 1
active_tracks    = {}
results          = []
times_pd         = pd.to_datetime(time)

print("starting tracking loop...")

for k in range(len(time)):

    current_date = times_pd[k]

    # ── set prev_date ──
    if k > 0:
        prev_date = times_pd[k - 1]
    else:
        prev_date = None

    # ── detect season break ──
    if k > 0 and prev_date.month == 9 and current_date.month == 6:
        season_break = True
        # active_tracks was already cleared on Sept 30 of prev year
        # nothing to record here — deaths were recorded on Sept 30
        active_tracks = {}
    else:
        season_break = False

    # ── get today's EREs ──
    cells_today = get_cell_sets(Label_ERE[k, :, :])

    # ── no EREs today ──
    if len(cells_today) == 0:
        if k > 0 and prev_date is not None:
            cells_yesterday = get_cell_sets(Label_ERE[k - 1, :, :])
            for lbl_y, cells_y in cells_yesterday.items():
                if (k - 1, lbl_y) in active_tracks:
                    tid   = active_tracks[(k - 1, lbl_y)]
                    etype = 'domain_exit' if is_domain_exit(cells_y) else 'death'
                    results.append(make_row(
                        prev_date, tid, None, cells_y, k - 1,
                        etype, np.nan, np.nan, data_precip, lat, lon))
        active_tracks = {}
        continue

    # ── get active EREs from yesterday ──
    if k == 0 or season_break:
        active_yesterday = {}
    else:
        cells_yesterday  = get_cell_sets(Label_ERE[k - 1, :, :])
        active_yesterday = {
            lbl: cells
            for lbl, cells in cells_yesterday.items()
            if (k - 1, lbl) in active_tracks
        }

    labels_yesterday = list(active_yesterday.keys())
    labels_today     = list(cells_today.keys())
    n = len(labels_yesterday)
    m = len(labels_today)

    # ── if nothing active yesterday — all today are births ──
    if n == 0:
        for lbl, cells in cells_today.items():
            results.append(make_row(
                current_date, track_id_counter, lbl, cells, k,
                'birth', np.nan, np.nan, data_precip, lat, lon))
            active_tracks[(k, lbl)] = track_id_counter
            track_id_counter += 1
        continue

    # ── build overlap matrix [n x m] ──
    overlap_matrix = np.zeros((n, m), dtype='float32')

    for i, lbl_y in enumerate(labels_yesterday):
        for j, lbl_t in enumerate(labels_today):
            cells_y = active_yesterday[lbl_y]
            cells_t = cells_today[lbl_t]
            ov = compute_overlap(cells_y, cells_t)
            if ov > 0:
                overlap_matrix[i, j] = ov
            else:
                if check_fallback(cells_y, cells_t, lat, lon,
                                  FALLBACK_DIST_DEG, FALLBACK_SIZE_RATIO):
                    overlap_matrix[i, j] = 1e-6

    # ── greedy 1-to-1 assignment ──
    assignments   = greedy_assignment(overlap_matrix, threshold=OVERLAP_THRESHOLD)
    assigned_rows = {i for i, j in assignments}
    assigned_cols = {j for i, j in assignments}

    # add fallback links not already assigned
    for i in range(n):
        for j in range(m):
            if overlap_matrix[i, j] == 1e-6:
                if i not in assigned_rows and j not in assigned_cols:
                    assignments.append((i, j))
                    assigned_rows.add(i)
                    assigned_cols.add(j)

    assigned_today_cols = {j for i, j in assignments}

    # ── record continuations ──
    for i, j in assignments:
        lbl_y   = labels_yesterday[i]
        lbl_t   = labels_today[j]
        cells_y = active_yesterday[lbl_y]
        cells_t = cells_today[lbl_t]
        tid     = active_tracks[(k - 1, lbl_y)]

        ov_pct = float(overlap_matrix[i, j])
        if ov_pct == 1e-6:
            ov_pct = 0.0

        lat_c, lon_c = centre_of_mass(cells_t, lat, lon)
        lat_p, lon_p = centre_of_mass(cells_y, lat, lon)
        dist_deg = float(np.sqrt((lat_c - lat_p)**2 + (lon_c - lon_p)**2))

        results.append(make_row(
            current_date, tid, lbl_t, cells_t, k,
            'continuation', round(ov_pct, 3), round(dist_deg, 3),
            data_precip, lat, lon))
        active_tracks[(k, lbl_t)] = tid

    # ── record births for unmatched today EREs ──
    for j, lbl_t in enumerate(labels_today):
        if j not in assigned_today_cols:
            cells_t = cells_today[lbl_t]
            results.append(make_row(
                current_date, track_id_counter, lbl_t, cells_t, k,
                'birth', np.nan, np.nan, data_precip, lat, lon))
            active_tracks[(k, lbl_t)] = track_id_counter
            track_id_counter += 1

    # ── record deaths for unmatched yesterday EREs ──
    for i, lbl_y in enumerate(labels_yesterday):
        if i not in assigned_rows:
            cells_y = active_yesterday[lbl_y]
            tid     = active_tracks[(k - 1, lbl_y)]
            etype   = 'domain_exit' if is_domain_exit(cells_y) else 'death'
            results.append(make_row(
                prev_date, tid, None, cells_y, k - 1,
                etype, np.nan, np.nan, data_precip, lat, lon))

    # ── Sept 30 hard break — terminate everything still active today ──
    if current_date.month == 9 and current_date.day == 30:
        for lbl_t, cells_t in cells_today.items():
            if (k, lbl_t) in active_tracks:
                tid   = active_tracks[(k, lbl_t)]
                etype = 'domain_exit' if is_domain_exit(cells_t) else 'death'
                results.append(make_row(
                    current_date, tid, None, cells_t, k,
                    etype, np.nan, np.nan, data_precip, lat, lon))
        active_tracks = {}

    if k % 500 == 0:
        print(f"  day {k}/{len(time)}  |  tracks so far: {track_id_counter - 1}")

print(f"\ntracking complete")
print(f"total track IDs assigned : {track_id_counter - 1}")
print(f"total ERE-day records    : {len(results)}")


# %% Cell 9
df_tracks = pd.DataFrame(results)
df_tracks['date'] = pd.to_datetime(df_tracks['date'])

# ── track-level summaries ──
track_summary = df_tracks.groupby('track_id').agg(
    track_duration  = ('date', 'count'),
    track_distance  = ('step_dist_deg', 'sum'),
).reset_index()

track_summary['mean_velocity'] = (
    track_summary['track_distance'] / track_summary['track_duration']
)

df_tracks = df_tracks.merge(track_summary, on='track_id', how='left')

print(f"df_tracks shape  : {df_tracks.shape}")
print(f"unique tracks    : {df_tracks['track_id'].nunique()}")
print(f"event type counts:")
print(df_tracks['event_type'].value_counts())


# %% Cell 10
df_tracks.to_csv('/kaggle/working/ere_tracks.csv', index=False)

np.save('/kaggle/working/Loc_ERE.npy',            Loc_ERE)
np.save('/kaggle/working/Label_ERE.npy',           Label_ERE)
np.save('/kaggle/working/NT_daily.npy',            nt_per_day)
np.save('/kaggle/working/NE_daily.npy',            n_ere_per_day)
np.save('/kaggle/working/per_grid_monthly.npy',    per_grid_monthly)
np.save('/kaggle/working/lat.npy',                 lat)
np.save('/kaggle/working/lon.npy',                 lon)
np.save('/kaggle/working/time.npy',                time)

print("saved:")
print("  ere_tracks.csv")
print("  Loc_ERE.npy")
print("  Label_ERE.npy")
print("  NT_daily.npy  NE_daily.npy")
print("  per_grid_monthly.npy")
print("  lat.npy  lon.npy  time.npy")


# %% Cell 11
# ── summary statistics ──
print("=" * 50)
print("TRACKING SUMMARY")
print("=" * 50)
print(f"  total ERE-day records   : {len(df_tracks)}")
print(f"  unique tracks           : {df_tracks['track_id'].nunique()}")
print(f"  single-day tracks       : {(track_summary['track_duration'] == 1).sum()}")
print(f"  tracks >= 2 days        : {(track_summary['track_duration'] >= 2).sum()}")
print(f"  tracks >= 3 days        : {(track_summary['track_duration'] >= 3).sum()}")
print(f"  max track duration      : {track_summary['track_duration'].max()} days")
print(f"  mean track duration     : {track_summary['track_duration'].mean():.2f} days")
print()

# ── event type breakdown ──
print("=" * 50)
print("EVENT TYPE BREAKDOWN")
print("=" * 50)
event_counts = df_tracks['event_type'].value_counts()
for etype, count in event_counts.items():
    pct = count / len(df_tracks) * 100
    print(f"  {etype:<15} : {count:>6}  ({pct:.1f}%)")
print()

# ── annual stats ──
print("=" * 50)
print("ANNUAL STATS  (mean per day)")
print("=" * 50)
print(f"  {'year':<6} {'NT':>6} {'NE':>6} {'tracks':>8} {'mean_dur':>10}")
print("  " + "-" * 40)

annual_nt = pd.DataFrame({
    'date' : pd.to_datetime(time),
    'NT'   : nt_per_day,
    'NE'   : n_ere_per_day,
})
annual_nt['year'] = annual_nt['date'].dt.year
annual_nt_g = annual_nt.groupby('year')[['NT','NE']].mean()

tracks_per_year = (
    df_tracks[df_tracks['event_type'] == 'birth']
    .groupby('year')['track_id'].count()
)
mean_dur_per_year = (
    df_tracks.groupby('year')['track_duration'].mean()
)
for yr in range(1998, 2021):
    nt  = annual_nt_g.loc[yr, 'NT']  if yr in annual_nt_g.index  else np.nan
    ne  = annual_nt_g.loc[yr, 'NE']  if yr in annual_nt_g.index  else np.nan
    tr  = tracks_per_year.loc[yr]     if yr in tracks_per_year.index else np.nan
    md  = mean_dur_per_year.loc[yr]   if yr in mean_dur_per_year.index else np.nan
    print(f"  {yr:<6} {nt:>6.2f} {ne:>6.2f} {tr:>8.0f} {md:>10.2f}")

print()

# ── clean sample rows ──
print("=" * 50)
print("SAMPLE ROWS (first 5 continuations)")
print("=" * 50)
sample = df_tracks[df_tracks['event_type'] == 'continuation'].head(5)[
    ['date', 'track_id', 'size', 'mean_rain',
     'overlap_pct', 'step_dist_deg', 'event_type', 'track_duration']
]
print(sample.to_string(index=False))


# %% [markdown]
# ## Tracking Results
#
# ### Summary
# | Metric | Value |
# |--------|-------|
# | Total ERE-day records | 16,789 |
# | Unique tracks | 11,705 |
# | Single-day tracks | 8,221 (70%) |
# | Tracks >= 2 days | 3,484 (30%) |
# | Tracks >= 3 days | 1,046 (9%) |
# | Max track duration | 10 days |
# | Mean track duration | 1.43 days |
#
# ### Event Types
# | Event | Count | % |
# |-------|-------|---|
# | death | 10,029 | 59.7% |
# | birth | 3,639 | 21.7% |
# | continuation | 1,667 | 9.9% |
# | domain_exit | 1,454 | 8.7% |
#
# ---
#
# ## Key Observations
#
# **Single-day dominance (70%):**
# Most tracks last one day. Expected — the 99th percentile threshold is very high
# and many monsoon rainfall events are episodic. Also a known limitation of 1° daily resolution.
#
# **Active vs weak years (PASS):**
# 2007 and 2010 have the highest track counts (197, 198) and NT (20.48, 18.69).
# Weak years 2002 and 2004 have the lowest (143, 138). Signal is clean.
#
# **Active years produce more tracks, not longer ones:**
# Mean duration is similar across all years (1.75–2.17 days).
# Active monsoon spells generate more simultaneous ERE systems rather than longer-lived ones.
#
# **Domain exits (8.7%):**
# ~1 in 12 tracks exits the domain boundary rather than dying naturally.
# Expected — monsoon systems propagate westward and northward out of the India grid.
#
# ---
#
# ## Open Questions for Validation
# 1. Is 70% single-day rate physical or a threshold artefact?
# 2. Do long tracks concentrate in Central India and the monsoon trough?
# 3. Does S-bar show an increasing trend over 1998–2020


# %% Cell 12
import os

for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# %% Cell 13
print("=" * 55)
print("GROUP 2 — TRACKING INTERNAL CONSISTENCY")
print("=" * 55)

# 2a. birth count == unique track count
print("\n2a. Birth count == unique track count")
birth_count  = (df_tracks['event_type'] == 'birth').sum()
unique_tracks = df_tracks['track_id'].nunique()
status = "PASS" if birth_count == unique_tracks else "FAIL"
print(f"  births        : {birth_count}")
print(f"  unique tracks : {unique_tracks}")
print(f"  status        : {status}")

# 2b. all continuations have valid overlap or step_dist
print("\n2b. Continuations have valid overlap_pct or step_dist_deg")
cont = df_tracks[df_tracks['event_type'] == 'continuation']
both_nan = cont['overlap_pct'].isna() & cont['step_dist_deg'].isna()
status = "PASS" if both_nan.sum() == 0 else "FAIL"
print(f"  continuations with both NaN : {both_nan.sum()}")
print(f"  status                      : {status}")

# 2c. no track crosses season boundary
print("\n2c. No track crosses Sept 30 → June 1 boundary")
df_tracks['year']  = df_tracks['date'].dt.year
df_tracks['month'] = df_tracks['date'].dt.month
cross = []
for tid, grp in df_tracks.groupby('track_id'):
    grp = grp.sort_values('date')
    years_in_track = grp['year'].unique()
    if len(years_in_track) > 1:
        cross.append(tid)
status = "PASS" if len(cross) == 0 else "FAIL"
print(f"  tracks crossing year boundary : {len(cross)}")
print(f"  status                        : {status}")

# 2d. domain exits cluster near boundaries
print("\n2d. Domain exits near expected boundaries")
exits = df_tracks[df_tracks['event_type'] == 'domain_exit']
near_boundary = (
    (exits['centre_lat'] <= 6)  |
    (exits['centre_lat'] >= 39) |
    (exits['centre_lon'] <= 61) |
    (exits['centre_lon'] >= 99)
)
pct_near = near_boundary.mean() * 100
status = "PASS" if pct_near >= 70 else "CHECK"
print(f"  domain exits near boundary : {pct_near:.1f}%")
print(f"  status                     : {status}")


# %% Cell 14
print("=" * 55)
print("GROUP 3 — PHYSICAL CHECKS")
print("=" * 55)

# 3a. active vs weak year track counts
print("\n3a. Active vs weak monsoon years")
weak_years   = [2002, 2004, 2009, 2014, 2015]
active_years = [2007, 2010, 2011, 2019, 2020]

tracks_per_year = (
    df_tracks[df_tracks['event_type'] == 'birth']
    .groupby('year')['track_id'].count()
)
mean_dur_year = df_tracks.groupby('year')['track_duration'].mean()

print(f"\n  {'year':<6} {'tracks':>8} {'mean_dur':>10} {'type':<8}")
print("  " + "-" * 36)
for yr in sorted(weak_years + active_years):
    if yr in tracks_per_year.index:
        ytype = "active" if yr in active_years else "weak"
        print(f"  {yr:<6} {tracks_per_year[yr]:>8} "
              f"{mean_dur_year[yr]:>10.2f} {ytype:<8}")

weak_mean   = tracks_per_year[
    [y for y in weak_years if y in tracks_per_year.index]].mean()
active_mean = tracks_per_year[
    [y for y in active_years if y in tracks_per_year.index]].mean()
status = "PASS" if active_mean > weak_mean else "FAIL"
print(f"\n  mean tracks — weak: {weak_mean:.0f}  active: {active_mean:.0f}")
print(f"  status : {status}")

# 3b. single-day rate investigation
print("\n3b. Single-day track rate investigation")
print("  checking consecutive ERE pairs that share at least one cell...")

shared_any = 0
total_pairs = 0

for k in range(1, len(time)):
    prev_date = times_pd[k-1]
    curr_date = times_pd[k]

    # skip season boundaries
    if prev_date.month == 9 and curr_date.month == 6:
        continue
    if prev_date.year != curr_date.year and not (
        prev_date.month == 9 and curr_date.month == 6):
        continue

    labels_prev = np.unique(Label_ERE[k-1])
    labels_prev = labels_prev[labels_prev > 0]
    labels_curr = np.unique(Label_ERE[k])
    labels_curr = labels_curr[labels_curr > 0]

    if len(labels_prev) == 0 or len(labels_curr) == 0:
        continue

    for lp in labels_prev:
        cells_p = frozenset(map(tuple, np.argwhere(Label_ERE[k-1] == lp)))
        for lc in labels_curr:
            cells_c = frozenset(map(tuple, np.argwhere(Label_ERE[k] == lc)))
            total_pairs += 1
            if len(cells_p & cells_c) > 0:
                shared_any += 1

pct_sharing = shared_any / total_pairs * 100 if total_pairs > 0 else 0
print(f"  total consecutive ERE pairs  : {total_pairs}")
print(f"  pairs sharing >= 1 cell      : {shared_any} ({pct_sharing:.1f}%)")
print(f"  pairs sharing zero cells     : {total_pairs - shared_any} "
      f"({100 - pct_sharing:.1f}%)")
print(f"\n  interpretation: of all possible t→t+1 ERE pairs,")
print(f"  {pct_sharing:.1f}% share at least one grid cell.")
print(f"  pairs below 15% overlap threshold are broken tracks.")


# %% Cell 15
print("=" * 55)
print("GROUP 3 — SPATIAL DISTRIBUTION OF LONG TRACKS")
print("=" * 55)

long_tracks = df_tracks[df_tracks['track_duration'] >= 3]

regions = {
    'Western Ghats'  : {'lat_min':  8, 'lat_max': 20, 'lon_min': 72, 'lon_max': 76},
    'Central India'  : {'lat_min': 15, 'lat_max': 25, 'lon_min': 75, 'lon_max': 85},
    'Northeast India': {'lat_min': 22, 'lat_max': 30, 'lon_min': 88, 'lon_max': 97},
    'Northwest India': {'lat_min': 25, 'lat_max': 35, 'lon_min': 68, 'lon_max': 76},
    'Pakistan/Desert': {'lat_min': 25, 'lat_max': 35, 'lon_min': 60, 'lon_max': 68},
}

print(f"\n  long track ERE-days (duration >= 3) by region")
print(f"  {'region':<20} {'ERE-days':>10} {'% of long':>10}")
print("  " + "-" * 44)

total_long = len(long_tracks)
for name, b in regions.items():
    subset = long_tracks[
        (long_tracks['centre_lat'] >= b['lat_min']) &
        (long_tracks['centre_lat'] <= b['lat_max']) &
        (long_tracks['centre_lon'] >= b['lon_min']) &
        (long_tracks['centre_lon'] <= b['lon_max'])
    ]
    pct = len(subset) / total_long * 100 if total_long > 0 else 0
    print(f"  {name:<20} {len(subset):>10} {pct:>9.1f}%")

print(f"\n  expected: Central India and Western Ghats highest")
print(f"  expected: Pakistan/Desert lowest")


# %% Cell 16
print("=" * 55)
print("GROUP 4 — NIKUMBH REPLICATION")
print("=" * 55)

df_daily = pd.DataFrame({
    'date'  : times_pd,
    'year'  : times_pd.year,
    'month' : times_pd.month,
    'NT'    : nt_per_day,
    'NE'    : n_ere_per_day,
})
df_daily['S_bar'] = df_daily['NT'] / df_daily['NE'].replace(0, np.nan)

annual = df_daily.groupby('year')[['NT', 'NE', 'S_bar']].mean().reset_index()

# linear trends
def linear_trend(years, values):
    mask = ~np.isnan(values)
    slope, intercept, r, p, se = stats.linregress(
        years[mask], values[mask])
    return slope, p, r**2

nt_slope,    nt_p,    nt_r2    = linear_trend(
    annual['year'].values, annual['NT'].values)
ne_slope,    ne_p,    ne_r2    = linear_trend(
    annual['year'].values, annual['NE'].values)
sbar_slope,  sbar_p,  sbar_r2  = linear_trend(
    annual['year'].values, annual['S_bar'].values)

print("\n  annual mean trends over 1998–2020")
print(f"  {'metric':<8} {'slope/yr':>10} {'p-value':>10} {'R²':>6}  status")
print("  " + "-" * 50)

for name, slope, p, r2 in [
    ('NT',    nt_slope,   nt_p,   nt_r2),
    ('NE',    ne_slope,   ne_p,   ne_r2),
    ('S_bar', sbar_slope, sbar_p, sbar_r2),
]:
    sig = "significant" if p < 0.05 else "not significant"
    print(f"  {name:<8} {slope:>+10.4f} {p:>10.4f} {r2:>6.3f}  {sig}")

print(f"\n  Nikumbh finding: NT rising, NE flat → S_bar increasing")
print(f"  NT  trend : {'rising' if nt_slope > 0 else 'falling'}")
print(f"  NE  trend : {'rising' if ne_slope > 0 else 'flat/falling'}")
print(f"  S_bar trend: {'rising ✓' if sbar_slope > 0 else 'falling ✗'}")


# %% Cell 17
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

df_daily = pd.DataFrame({
    'date'  : times_pd,
    'year'  : times_pd.year,
    'NT'    : nt_per_day,
    'NE'    : n_ere_per_day,
})
df_daily['S_bar'] = df_daily['NT'] / df_daily['NE'].replace(0, np.nan)
annual = df_daily.groupby('year')[['NT', 'NE', 'S_bar']].mean().reset_index()
years  = annual['year'].values

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.suptitle('Annual Mean NT, NE and S̄ — ERA5 India 1998–2020', fontsize=13)

metrics = [
    ('NT',    'Mean extreme grids/day',  'steelblue'),
    ('NE',    'Mean EREs/day',           'darkorange'),
    ('S_bar', 'Mean ERE size (NT/NE)',   'seagreen'),
]

for ax, (col, ylabel, color) in zip(axes, metrics):
    vals = annual[col].values
    slope, intercept, r, p, _ = stats.linregress(years, vals)
    trend = slope * years + intercept
    sig   = f'p={p:.3f}' + (' *' if p < 0.05 else '')

    ax.bar(years, vals, color=color, alpha=0.7, width=0.7)
    ax.plot(years, trend, color='red', linewidth=2,
            label=f'trend {slope:+.4f}/yr  {sig}')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(1997, 2021)

axes[-1].set_xlabel('Year')
plt.tight_layout()
plt.savefig('/kaggle/working/nt_ne_sbar_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print("saved → nt_ne_sbar_trends.png")


# %% [markdown]
# CROSS MODEL COMPARE VARIABLES


# %% Cell 18
print("=" * 55)
print("CATEGORY 1 — INTERNAL CONSISTENCY METRICS")
print("=" * 55)

total_records    = len(df_tracks)
births           = (df_tracks['event_type'] == 'birth').sum()
continuations    = (df_tracks['event_type'] == 'continuation').sum()
deaths           = (df_tracks['event_type'] == 'death').sum()
domain_exits     = (df_tracks['event_type'] == 'domain_exit').sum()

# active rows only (birth + continuation)
active_rows      = df_tracks[~df_tracks['event_type'].isin(['death','domain_exit'])]
track_dur        = active_rows.groupby('track_id')['date'].count()

total_tracks     = df_tracks['track_id'].nunique()
single_day       = (track_dur == 1).sum()
multi_day        = (track_dur >= 2).sum()

# continuation rate = continuations / (births + continuations)
cont_rate        = continuations / (births + continuations) * 100

# track fragmentation index = total tracks / total ERE-days
frag_index       = total_tracks / total_records

# orphan rate = tracks that only have birth, never a continuation
tracks_with_cont = set(df_tracks[df_tracks['event_type']=='continuation']['track_id'])
orphan_tracks    = total_tracks - len(tracks_with_cont)
orphan_rate      = orphan_tracks / total_tracks * 100

print(f"\n  {'metric':<35} {'value':>12}")
print("  " + "-" * 50)
print(f"  {'total ERE-day records':<35} {total_records:>12,}")
print(f"  {'unique tracks':<35} {total_tracks:>12,}")
print(f"  {'continuation rate':<35} {cont_rate:>11.1f}%")
print(f"  {'mean track duration (days)':<35} {track_dur.mean():>12.2f}")
print(f"  {'median track duration (days)':<35} {track_dur.median():>12.1f}")
print(f"  {'max track duration (days)':<35} {track_dur.max():>12}")
print(f"  {'single-day track fraction':<35} {single_day/total_tracks*100:>11.1f}%")
print(f"  {'multi-day tracks (>= 2 days)':<35} {multi_day:>12,}")
print(f"  {'track fragmentation index':<35} {frag_index:>12.4f}")
print(f"  {'orphan rate (never continued)':<35} {orphan_rate:>11.1f}%")

print(f"\n  track duration distribution:")
print(f"  {'duration':<12} {'count':>8} {'%':>8}")
print("  " + "-" * 30)
for d in range(1, min(track_dur.max()+1, 12)):
    count = (track_dur == d).sum()
    pct   = count / total_tracks * 100
    bar   = '█' * int(pct / 2)
    print(f"  {str(d)+' day':<12} {count:>8,} {pct:>7.1f}%  {bar}")
if track_dur.max() >= 12:
    count = (track_dur >= 12).sum()
    pct   = count / total_tracks * 100
    print(f"  {'>=12 days':<12} {count:>8,} {pct:>7.1f}%")

# summary score
print(f"\n  MODEL SCORE — CATEGORY 1")
print(f"  continuation rate   : {cont_rate:.1f}%   (target > 20%)")
print(f"  mean duration       : {track_dur.mean():.2f}d  (target > 2.0d)")
print(f"  single-day fraction : {single_day/total_tracks*100:.1f}%  (target < 60%)")


# %% Cell 19
print("=" * 55)
print("CATEGORY 2 — PHYSICAL CONSISTENCY METRICS")
print("=" * 55)

weak_years   = [2002, 2004, 2009, 2014, 2015]
active_years = [2007, 2010, 2011, 2019, 2020]

birth_rows   = df_tracks[df_tracks['event_type'] == 'birth'].copy()
birth_rows['year']  = pd.to_datetime(birth_rows['date']).dt.year
birth_rows['month'] = pd.to_datetime(birth_rows['date']).dt.month

# ── 2a. active vs weak year track count ratio ──
print("\n2a. Active vs weak year track count ratio")
tracks_yr    = birth_rows.groupby('year')['track_id'].count()
active_mean  = tracks_yr[[y for y in active_years if y in tracks_yr.index]].mean()
weak_mean    = tracks_yr[[y for y in weak_years   if y in tracks_yr.index]].mean()
ratio        = active_mean / weak_mean
status       = "PASS" if ratio >= 1.2 else "FAIL"

print(f"  active year mean tracks : {active_mean:.0f}")
print(f"  weak year mean tracks   : {weak_mean:.0f}")
print(f"  active / weak ratio     : {ratio:.2f}  (target >= 1.20)")
print(f"  status                  : {status}")

# ── 2b. peak track activity month ──
print("\n2b. Peak track activity month (expected July or August)")
cont_rows = df_tracks[df_tracks['event_type'] == 'continuation'].copy()
cont_rows['month'] = pd.to_datetime(cont_rows['date']).dt.month
month_map  = {6:'June', 7:'July', 8:'August', 9:'September'}
cont_monthly = cont_rows.groupby('month')['track_id'].count()

print(f"  {'month':<12} {'continuations':>15} {'%':>8}")
print("  " + "-" * 38)
for m in [6,7,8,9]:
    count = cont_monthly.get(m, 0)
    pct   = count / cont_monthly.sum() * 100
    bar   = '█' * int(pct / 3)
    print(f"  {month_map[m]:<12} {count:>15,} {pct:>7.1f}%  {bar}")

peak_month = cont_monthly.idxmax()
status     = "PASS" if peak_month in [7, 8] else "FAIL"
print(f"\n  peak month : {month_map[peak_month]}")
print(f"  status     : {status}  (expected July or August)")

# ── 2c. spatial concentration of long tracks ──
print("\n2c. Spatial concentration of long tracks (duration >= 3)")
long_tracks  = df_tracks[df_tracks['track_duration'] >= 3]
total_long   = len(long_tracks)

regions = {
    'Central India'  : dict(lat_min=15, lat_max=25, lon_min=75, lon_max=85),
    'Northeast India': dict(lat_min=22, lat_max=30, lon_min=88, lon_max=97),
    'Western Ghats'  : dict(lat_min= 8, lat_max=20, lon_min=72, lon_max=76),
    'Northwest India': dict(lat_min=25, lat_max=35, lon_min=68, lon_max=76),
    'Pakistan/Desert': dict(lat_min=25, lat_max=35, lon_min=60, lon_max=68),
}

print(f"\n  {'region':<20} {'ERE-days':>10} {'%':>8}  expected")
print("  " + "-" * 55)
region_pcts = {}
for name, b in regions.items():
    subset = long_tracks[
        (long_tracks['centre_lat'] >= b['lat_min']) &
        (long_tracks['centre_lat'] <= b['lat_max']) &
        (long_tracks['centre_lon'] >= b['lon_min']) &
        (long_tracks['centre_lon'] <= b['lon_max'])
    ]
    pct = len(subset) / total_long * 100 if total_long > 0 else 0
    region_pcts[name] = pct
    expected = 'highest' if name in ['Central India','Northeast India'] else \
               'lowest'  if name == 'Pakistan/Desert' else 'medium'
    print(f"  {name:<20} {len(subset):>10,} {pct:>7.1f}%  {expected}")

top_region = max(region_pcts, key=region_pcts.get)
bot_region = min(region_pcts, key=region_pcts.get)
status_top = "PASS" if top_region in ['Central India','Northeast India'] else "FAIL"
status_bot = "PASS" if bot_region == 'Pakistan/Desert' else "FAIL"
print(f"\n  highest region : {top_region}  —  {status_top}")
print(f"  lowest region  : {bot_region}  —  {status_bot}")

# ── domain exit direction ──
print("\n2d. Domain exit direction (expected westward + northward)")
exit_tracks = df_tracks[df_tracks['event_type'].isin(['birth','continuation','death','domain_exit'])]
exit_ids    = df_tracks[df_tracks['event_type'] == 'domain_exit']['track_id'].unique()

directions  = {'west':0, 'east':0, 'north':0, 'south':0}
for tid in exit_ids:
    trk = df_tracks[df_tracks['track_id'] == tid].sort_values('date')
    trk = trk[~trk['event_type'].isin(['death','domain_exit'])]
    if len(trk) < 2:
        continue
    dlon = trk.iloc[-1]['centre_lon'] - trk.iloc[0]['centre_lon']
    dlat = trk.iloc[-1]['centre_lat'] - trk.iloc[0]['centre_lat']
    if abs(dlon) >= abs(dlat):
        directions['west' if dlon < 0 else 'east'] += 1
    else:
        directions['north' if dlat > 0 else 'south'] += 1

total_dir = sum(directions.values())
print(f"  {'direction':<10} {'count':>8} {'%':>8}")
print("  " + "-" * 28)
for d, count in directions.items():
    pct = count / total_dir * 100 if total_dir > 0 else 0
    print(f"  {d:<10} {count:>8} {pct:>7.1f}%")
west_north = (directions['west'] + directions['north']) / total_dir * 100
status = "PASS" if west_north >= 50 else "FAIL"
print(f"\n  westward + northward : {west_north:.1f}%  (target >= 50%)")
print(f"  status               : {status}")


# %% Cell 20
print("=" * 55)
print("CATEGORY 3 — KNOWN EVENT VALIDATION")
print("=" * 55)

known_events = {
    'Uttarakhand 2013': {
        'dates'    : pd.date_range('2013-06-14', '2013-06-18'),
        'lat_range': (28.0, 32.0),
        'lon_range': (77.0, 81.0),
        'expected_size': 10,
        'note'     : 'Catastrophic flooding, ~5700 deaths',
    },
    'Kerala 2018': {
        'dates'    : pd.date_range('2018-08-14', '2018-08-20'),
        'lat_range': (8.0,  12.0),
        'lon_range': (74.0, 78.0),
        'expected_size': 8,
        'note'     : 'Worst floods since 1924',
    },
}

df_tracks['date'] = pd.to_datetime(df_tracks['date'])

for event_name, ev in known_events.items():
    print(f"\n{'─'*50}")
    print(f"  {event_name}")
    print(f"  {ev['note']}")
    print(f"{'─'*50}")

    # find EREs in the region during the event dates
    window = df_tracks[
        (df_tracks['date'].isin(ev['dates']))          &
        (df_tracks['centre_lat'] >= ev['lat_range'][0]) &
        (df_tracks['centre_lat'] <= ev['lat_range'][1]) &
        (df_tracks['centre_lon'] >= ev['lon_range'][0]) &
        (df_tracks['centre_lon'] <= ev['lon_range'][1])
    ].sort_values('date')

    if len(window) == 0:
        print(f"  FAIL — no EREs detected in region during event window")
        print(f"  region : {ev['lat_range']}N  {ev['lon_range']}E")
        print(f"  dates  : {ev['dates'][0].date()} to {ev['dates'][-1].date()}")
        continue

    print(f"\n  EREs detected in event region:")
    print(f"  {'date':<14} {'track_id':>10} {'size':>6} "
          f"{'mean_rain':>11} {'event_type':<15} {'duration':>10}")
    print("  " + "-" * 70)

    for _, row in window.iterrows():
        print(f"  {str(row['date'])[:10]:<14} "
              f"{int(row['track_id']):>10} "
              f"{int(row['size']):>6} "
              f"{row['mean_rain']:>10.1f} "
              f"{row['event_type']:<15} "
              f"{int(row['track_duration']):>9}d")

    # check if any track spans multiple days in the window
    track_ids_found  = window['track_id'].unique()
    multi_day_tracks = []
    for tid in track_ids_found:
        full_track = df_tracks[df_tracks['track_id'] == tid]
        active     = full_track[~full_track['event_type'].isin(['death','domain_exit'])]
        if len(active) >= 2:
            multi_day_tracks.append({
                'track_id' : tid,
                'duration' : int(full_track['track_duration'].iloc[0]),
                'max_size' : int(full_track['size'].max()),
                'max_rain' : float(full_track['mean_rain'].max()),
            })

    print(f"\n  summary:")
    print(f"  ERE-day records found  : {len(window)}")
    print(f"  unique tracks found    : {len(track_ids_found)}")
    print(f"  multi-day tracks       : {len(multi_day_tracks)}")

    if multi_day_tracks:
        print(f"\n  multi-day track details:")
        for t in multi_day_tracks:
            print(f"    track {t['track_id']:>6}  "
                  f"duration {t['duration']}d  "
                  f"max size {t['max_size']} cells  "
                  f"max rain {t['max_rain']:.1f} mm/day")
        status = "PASS"
    else:
        status = "FAIL — event detected but not tracked as multi-day"

    print(f"\n  status : {status}")

print(f"\n{'='*55}")
print(f"  CATEGORY 3 SUMMARY")
print(f"{'='*55}")
print(f"  Uttarakhand 2013 : detected and multi-day tracked?")
print(f"  Kerala 2018      : detected and multi-day tracked?")
print(f"  Note: detection failure at 1° resolution is expected")
print(f"  for localised events. Improvement expected at 0.25°.")


# %% Cell 21
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── find the best 5 consecutive days — most EREs + most continuations ──
df_tracks['date'] = pd.to_datetime(df_tracks['date'])
times_pd          = pd.to_datetime(time)

lon_arr = np.array(ds['lon'].values) if 'ds' in dir() else np.arange(60, 101, 1, dtype=float)
lat_arr = np.array(ds['lat'].values) if 'ds' in dir() else np.arange(5,  41, 1, dtype=float)

# score each possible 5-day window by number of continuations
best_score = 0
best_start = '2007-07-10'

for year in range(1998, 2021):
    for month in [6, 7, 8, 9]:
        month_days = df_tracks[
            (df_tracks['date'].dt.year  == year) &
            (df_tracks['date'].dt.month == month)
        ]
        dates_in_month = sorted(month_days['date'].unique())
        for i in range(len(dates_in_month) - 4):
            window_dates = dates_in_month[i:i+5]
            if (window_dates[-1] - window_dates[0]).days != 4:
                continue
            sub = month_days[month_days['date'].isin(window_dates)]
            score = (sub['event_type'] == 'continuation').sum() * 3 + \
                    len(sub['track_id'].unique())
            if score > best_score:
                best_score = score
                best_start = str(window_dates[0])[:10]

days     = pd.date_range(best_start, periods=5, freq='D')
day_strs = [str(d)[:10] for d in days]
print(f"best 5-day window : {day_strs[0]} to {day_strs[-1]}  (score={best_score})")

# ── get all active rows in this window ──
window = df_tracks[
    df_tracks['date'].isin(days) &
    ~df_tracks['event_type'].isin(['death', 'domain_exit'])
].copy()

# ── assign consistent colours per track ──
track_ids = sorted(window['track_id'].unique())
cmap      = plt.cm.get_cmap('tab20', max(len(track_ids), 20))
color_map = {tid: cmap(i % 20) for i, tid in enumerate(track_ids)}

# ── figure ──
fig, axes = plt.subplots(1, 5, figsize=(22, 6))
fig.patch.set_facecolor('#0a0f1e')
fig.suptitle(
    f'ERE Tracking — Best 5-Day Window  |  {day_strs[0]}  →  {day_strs[-1]}',
    fontsize=13, color='#aad4ff', y=1.01,
    fontfamily='monospace', fontweight='bold'
)

LON_MIN, LON_MAX = 60, 100
LAT_MIN, LAT_MAX =  5,  40

for ax_idx, (ax, date_str) in enumerate(zip(axes, day_strs)):

    ax.set_facecolor('#0d1525')
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect('equal')

    # grid lines
    for glon in range(60, 101, 5):
        ax.axvline(glon, color='#141e33', linewidth=0.4, zorder=1)
    for glat in range(5, 41, 5):
        ax.axhline(glat, color='#141e33', linewidth=0.4, zorder=1)

    for spine in ax.spines.values():
        spine.set_edgecolor('#1e3a5f')
        spine.set_linewidth(1.2)

    today_data = window[
        window['date'].dt.strftime('%Y-%m-%d') == date_str
    ]
    today_tids = set(today_data['track_id'].unique())

    # ── trails ──
    for tid in today_tids:
        tcolor = color_map[tid]
        trail  = window[
            (window['track_id'] == tid) &
            (window['date'] <= pd.Timestamp(date_str))
        ].sort_values('date')

        if len(trail) >= 2:
            lons = trail['centre_lon'].values
            lats = trail['centre_lat'].values
            for seg in range(len(lons) - 1):
                alpha = 0.15 + 0.3 * (seg / len(lons))
                ax.plot([lons[seg], lons[seg+1]],
                        [lats[seg], lats[seg+1]],
                        color=tcolor, linewidth=1.5,
                        alpha=alpha, zorder=2)
            if len(lons) >= 2:
                ax.annotate('',
                    xy=(lons[-1], lats[-1]),
                    xytext=(lons[-2], lats[-2]),
                    arrowprops=dict(
                        arrowstyle='->', color=tcolor,
                        lw=1.2, alpha=0.6),
                    zorder=3)

    # ── ERE grid cells from Label_ERE ──
    time_idx = np.where(
        pd.to_datetime(time) == pd.Timestamp(date_str)
    )[0]

    if len(time_idx) > 0:
        label_day = Label_ERE[time_idx[0]]
        for li in range(label_day.shape[0]):
            for lj in range(label_day.shape[1]):
                lbl = label_day[li, lj]
                if lbl == 0:
                    continue
                match = today_data[today_data['day_label'] == lbl]
                if len(match) == 0:
                    cell_color = '#1a2a3a'
                    alpha      = 0.4
                else:
                    tid        = match.iloc[0]['track_id']
                    cell_color = color_map[tid]
                    alpha      = 0.75

                rect = mpatches.Rectangle(
                    (lon_arr[lj] - 0.5, lat_arr[li] - 0.5),
                    1.0, 1.0,
                    linewidth=0.3,
                    edgecolor='#0a0f1e',
                    facecolor=cell_color,
                    alpha=alpha,
                    zorder=4
                )
                ax.add_patch(rect)

    # ── centre of mass dots ──
    for _, row in today_data.iterrows():
        tid    = row['track_id']
        tcolor = color_map[tid]
        r      = max(4, min(14, 3 + row['size'] * 2))

        # glow
        ax.plot(row['centre_lon'], row['centre_lat'],
                'o', color=tcolor,
                markersize=r * 2.2,
                alpha=0.12, zorder=5)
        # dot
        ax.plot(row['centre_lon'], row['centre_lat'],
                'o', color=tcolor,
                markersize=r,
                markeredgecolor='white',
                markeredgewidth=0.5,
                zorder=6)
        # track id
        ax.text(row['centre_lon'] + 0.4,
                row['centre_lat'] + 0.4,
                str(row['track_id']),
                fontsize=5.5, color='white',
                fontfamily='monospace',
                zorder=7, alpha=0.9)

    # ── axis styling ──
    ax.set_title(date_str, fontsize=9, color='#7aaed4',
                 fontfamily='monospace', pad=5)
    ax.set_xticks(range(60, 101, 10))
    ax.set_xticklabels(
        [f'{l}°E' for l in range(60, 101, 10)],
        color='#334466', fontsize=6)
    if ax_idx == 0:
        ax.set_yticks(range(5, 41, 5))
        ax.set_yticklabels(
            [f'{l}°N' for l in range(5, 41, 5)],
            color='#334466', fontsize=6)
    else:
        ax.set_yticks([])

    # ── info badge ──
    n_cont = (today_data['event_type'] == 'continuation').sum()
    n_born = (today_data['event_type'] == 'birth').sum()
    ax.text(61, 37.5,
            f'EREs: {len(today_data)}\nCont: {n_cont}  Born: {n_born}',
            fontsize=6.5, color='#aad4ff',
            fontfamily='monospace', linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#0d1525',
                      edgecolor='#1e3a5f',
                      alpha=0.85),
            zorder=8)

    ax.text(98.5, 6.5, f'D{ax_idx+1}',
            fontsize=9, color='#4af',
            fontfamily='monospace',
            ha='right', zorder=8)

# ── legend — multi-day tracks only ──
multi_tids = window.groupby('track_id')['date'].nunique()
multi_tids = multi_tids[multi_tids >= 2].index.tolist()[:10]

legend_handles = []
for tid in multi_tids:
    tcolor = color_map[tid]
    dur    = int(df_tracks[df_tracks['track_id'] == tid]['track_duration'].iloc[0])
    sz     = int(df_tracks[df_tracks['track_id'] == tid]['size'].max())
    legend_handles.append(
        mpatches.Patch(color=tcolor,
                       label=f'Track {tid}  |  {dur}d  |  max {sz} cells')
    )

if legend_handles:
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=min(len(legend_handles), 5),
        fontsize=7,
        framealpha=0.2,
        facecolor='#0d1525',
        edgecolor='#1e3a5f',
        labelcolor='white',
        title='Multi-day tracks  (track id  |  duration  |  max size)',
        title_fontsize=7.5,
        bbox_to_anchor=(0.5, -0.10)
    )

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig('/kaggle/working/ere_best5day_snapshot.png',
            dpi=180, bbox_inches='tight',
            facecolor='#0a0f1e')
plt.show()
print(f"saved → ere_best5day_snapshot.png")
