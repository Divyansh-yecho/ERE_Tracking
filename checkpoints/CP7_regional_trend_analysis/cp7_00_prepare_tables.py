# cp7_00_prepare_tables.py: enriched object-day and track tables for the regional analysis.
# Reads the IMERG catalogue built by CP6 (MEAN 0.10 gate; object-days, durations and LPS classes
# are identical under the MIN 0.05 gate) and adds: region labels, the core-monsoon-zone (CMZ)
# active/break index, LPS class (near / far / no_lps) and per-track summaries.
#
# CMZ index: daily mean IMERG rain over India land cells in 18-28N, 73-88E, standardised per
# calendar day against a 15-day-smoothed 25-season climatology. A spell is active (break) when
# z >= +1 (z <= -1) for >= 3 consecutive days. `phase` classifies all of JJAS; `phase_ja` is the
# standard July-August-only classification used for every result (June/September excluded).
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ERE_DATA", REPO / "data"))
DER = REPO / "data" / "derived"

stats = pd.read_parquet(DER / "objdays_mean.parquet")
summary = pd.read_parquet(DER / "tracks_mean.parquet")
z = np.load(DER / "imerg_setup.npz")
lat, lon, india_mask = z["lat"], z["lon"], z["india_mask"]
times = pd.to_datetime(z["times"], format="%Y%m%d")
date_str = times.strftime("%Y%m%d")
assert len(stats) == 10877 and len(summary) == 10006, "run cp6_build_imerg_catalogue.py first"


# ---------- regions (by object centroid; priority order) ----------
def region_of(la, lo):
    if lo >= 88.5 and 21.5 <= la <= 29.5:
        return "northeast"
    if lo <= 77.5 and 8.0 <= la <= 21.0:
        return "west_ghats"
    if la >= 26.5:
        return "himalayan_foothills"
    if 17.0 <= la < 26.5 and 74.0 <= lo <= 88.5:
        return "central_india"
    return "other"


stats["region"] = [region_of(la, lo) for la, lo in zip(stats.centroid_lat, stats.centroid_lon)]

# ---------- track-level LPS class (mean nearest-LPS distance over matchable days) ----------
per = stats[stats.matchable].groupby("track_num").nearest_lps_km.mean()
track_class = pd.DataFrame({"track_num": summary.track_num})
track_class["mean_lps_km"] = track_class.track_num.map(per)
track_class["lps_class"] = np.where(track_class.mean_lps_km.isna(), "no_lps",
                                    np.where(track_class.mean_lps_km <= 500, "near", "far"))

# ---------- CMZ active/break index ----------
ds = xr.open_dataset(DATA / "imerg_india_025_jjas_2000_2024.nc")
tp = ds["tp"].values.astype("float32"); ds.close()
box = india_mask & ((lat >= 18) & (lat <= 28))[:, None] & ((lon >= 73) & (lon <= 88))[None, :]
cmz_rain = tp[:, box].mean(axis=1)
del tp

doy_key = times.strftime("%m%d")
cal = pd.DataFrame({"rain": cmz_rain, "key": doy_key})
clim_mean, clim_std = cal.groupby("key").rain.mean(), cal.groupby("key").rain.std()
cm = pd.Series(clim_mean.to_numpy()).rolling(15, center=True, min_periods=1).mean().to_numpy()
cs = pd.Series(clim_std.to_numpy()).rolling(15, center=True, min_periods=1).mean().to_numpy()
cm_map, cs_map = dict(zip(clim_mean.index, cm)), dict(zip(clim_std.index, cs))
z_cmz = np.array([(r - cm_map[k]) / cs_map[k] for r, k in zip(cmz_rain, doy_key)])


def classify(zv, months, thr=1.0):
    """active/break where |z| >= thr holds >= 3 consecutive days inside one season."""
    mo, yr = times.month.to_numpy(), times.year.to_numpy()
    inm = np.isin(mo, months)
    ph = np.array(["excluded"] * len(zv), dtype=object)
    ph[inm] = "neutral"
    for cond, name in [(zv >= thr, "active"), (zv <= -thr, "break")]:
        c = cond & inm
        i = 0
        while i < len(c):
            if c[i]:
                j = i
                while j + 1 < len(c) and c[j + 1] and yr[j + 1] == yr[i]:
                    j += 1
                if j - i + 1 >= 3:
                    ph[i:j + 1] = name
                i = j + 1
            else:
                i += 1
    return ph


season = pd.DataFrame({"date": date_str, "cmz_rain_mm": cmz_rain, "cmz_z": z_cmz,
                       "phase": classify(z_cmz, [6, 7, 8, 9]),
                       "phase_ja": classify(z_cmz, [7, 8])})
print("Jul-Aug spell days:", season.phase_ja.value_counts().to_dict())
stats = stats.merge(season[["date", "cmz_z", "phase", "phase_ja"]], on="date", how="left")


# ---------- track-level enrichment ----------
def hav(a, b):
    p1, p2 = np.radians(a[0]), np.radians(b[0])
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(b[1] - a[1]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


rows = []
for tn, g in stats.groupby("track_num"):
    g = g.sort_values("day_of_track")
    la, lo = g.centroid_lat.to_numpy(), g.centroid_lon.to_numpy()
    steps = [hav((la[k - 1], lo[k - 1]), (la[k], lo[k])) for k in range(1, len(g))]
    rows.append(dict(
        track_num=tn, birth_region=g.region.iloc[0], modal_region=g.region.mode().iloc[0],
        birth_phase=g.phase.iloc[0], birth_phase_ja=g.phase_ja.iloc[0], birth_cmz_z=g.cmz_z.iloc[0],
        mean_size=g["size"].mean(), peak_size=g["size"].max(),
        mean_mm=g.mean_mm.mean(), peak_mean_mm=g.mean_mm.max(), peak_max_mm=g.max_mm.max(),
        total_vol=g.vol_mm.sum(),
        day1_size=g["size"].iloc[0], day1_mean_mm=g.mean_mm.iloc[0],
        day1_max_mm=g.max_mm.iloc[0], day1_vol=g.vol_mm.iloc[0],
        mean_speed_kmpd=(np.mean(steps) if steps else np.nan),
        birth_month=int(g.date.iloc[0][4:6])))
tracks = (pd.DataFrame(rows)
          .merge(summary[["track_num", "track_id", "birth_date", "year", "duration", "start_type",
                          "end_type", "n_merge_events", "n_split_events"]], on="track_num")
          .merge(track_class, on="track_num"))

stats.to_parquet(DER / "cp7_objdays.parquet")
tracks.to_parquet(DER / "cp7_tracks.parquet")
season.to_csv(DER / "cmz_daily_index.csv", index=False)
print("tracks by birth region:", tracks.birth_region.value_counts().to_dict())
print("3+ day tracks by LPS class:", tracks[tracks.duration >= 3].lps_class.value_counts().to_dict())
print("saved cp7_objdays / cp7_tracks / cmz_daily_index ->", DER)
