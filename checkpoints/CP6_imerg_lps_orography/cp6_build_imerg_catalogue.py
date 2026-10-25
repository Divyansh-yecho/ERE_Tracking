# cp6_build_imerg_catalogue.py: local re-run of the CP6b IMERG tracker (no Kaggle needed).
# Same recipe as the SETUP + run_tracking_mean cells of cp6b_imerg_tracking_lps_colocation.ipynb:
#   per-grid wet-day (>1 mm) 99th percentile over India, floored at 50 mm/day; 8-connectivity
#   daily objects; overlap tracking with explicit merges/splits, no fallback; domain exit when
#   >50% of an object lies outside India; season reset on 30 September.
# Two merge/split gates are run (continuations are identical in both):
#   mean  : (fwd+bwd)/2 >= 0.10, the CP5/CP6 notebook rule (71 merges, 51 splits)
#   min05 : min(fwd,bwd) >= 0.05, the stricter rule used from CP7 on (28 merges, 17 splits)
# Every object-day also gets its rainfall statistics and the nearest same-date LPS centre.
# Inputs : data/imerg_india_025_jjas_2000_2024.nc, data/lps_daily_2000_2022.parquet
# Outputs: data/derived/imerg_setup.npz, objdays_{mean,min05}.parquet, tracks_{gate}.parquet,
#          merges_{gate}.parquet, splits_{gate}.parquet
import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import regionmask
from scipy import ndimage

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ERE_DATA", REPO / "data"))
DER = REPO / "data" / "derived"
DER.mkdir(parents=True, exist_ok=True)

RAINY_DAY_MIN, PERCENTILE, FLOOR_MM = 1.0, 99.0, 50.0
MIN_TRACK_CELLS, DOMAIN_EXIT_FRAC = 1, 0.50
GATES = {"mean": ("mean", 0.10), "min05": ("min", 0.05)}
RH850_MIN = 50.0                       # moist-LPS filter (drops 2 of 2,232 system-days)

# ---------- IMERG cube, India mask, thresholds, 8-connectivity objects ----------
ds = xr.open_dataset(DATA / "imerg_india_025_jjas_2000_2024.nc")
data_precip = ds["tp"].values.astype("float32")          # (3050, 141, 161) mm/day
lat = ds["lat"].values.astype("float32")                 # 40 -> 5 (descending)
lon = ds["lon"].values.astype("float32")                 # 60 -> 100
times_pd = pd.to_datetime(ds["time"].values)
ds.close()
n_days, n_lat, n_lon = data_precip.shape
print(f"IMERG {data_precip.shape}  {times_pd[0]:%Y-%m-%d} -> {times_pd[-1]:%Y-%m-%d}")

countries = regionmask.defined_regions.natural_earth_v5_0_0.countries_110
india_mask = (countries.mask(lon, lat).values == 98)
print(f"India cells: {india_mask.sum()} / {india_mask.size}")

data_masked = np.where((data_precip > RAINY_DAY_MIN) & india_mask[None], data_precip,
                       np.nan).astype("float32")
with warnings.catch_warnings():                          # all-NaN columns = cells outside India
    warnings.simplefilter("ignore", RuntimeWarning)
    per_grid_99 = np.nanpercentile(data_masked, PERCENTILE, axis=0).astype("float32")
per_grid_threshold = np.maximum(per_grid_99, FLOOR_MM).astype("float32")
per_grid_threshold[~india_mask] = 999.0                  # non-India cells never exceed
del data_masked

c = data_precip > per_grid_threshold[None]
struct8 = ndimage.generate_binary_structure(2, 2)
Label8 = np.zeros((n_days, n_lat, n_lon), dtype="int32")
NE8_raw = np.zeros(n_days, dtype="int32")
for k in range(n_days):
    Label8[k], NE8_raw[k] = ndimage.label(c[k].astype("int8"), structure=struct8)
thr_in = per_grid_threshold[india_mask]
print(f"threshold over India min/median/max = {thr_in.min():.1f}/{np.median(thr_in):.1f}/"
      f"{thr_in.max():.1f} mm/day · floor binds in {(per_grid_99[india_mask] < FLOOR_MM).mean()*100:.1f}%"
      f" · mean NE = {NE8_raw.mean():.3f}/day")


# ---------- tracker (cell 6 of the notebook; only the merge/split score is parameterised) ----------
def run_tracking(score, gate):
    track_stats_rows, merge_event_rows, split_event_rows = [], [], []
    active_tracks, births_per_date = {}, {}
    counters = {"track": 0, "merge": 0, "split": 0}

    def jjas_day_num(ts): return int((ts - pd.Timestamp(f"{ts.year}-06-01")).days + 1)

    def get_objects(day_idx):
        labeled, n_obj, objs = Label8[day_idx], NE8_raw[day_idx], {}
        for lbl in range(1, n_obj + 1):
            mask = labeled == lbl; sz = int(mask.sum())
            if sz < MIN_TRACK_CELLS: continue
            r_, c_ = np.where(mask)
            objs[lbl] = {"size": sz, "centroid": (float(lat[r_].mean()), float(lon[c_].mean())),
                         "outside": int((mask & ~india_mask).sum())}
        return objs

    def compute_overlaps(dt, dt1, ot, ot1):
        lab_t, lab_t1, pairs = Label8[dt], Label8[dt1], {}
        for li in ot:
            mi = lab_t == li
            for lj in np.unique(lab_t1[mi]):
                if lj == 0 or lj not in ot1: continue
                sh = int((mi & (lab_t1 == lj)).sum())
                if sh == 0: continue
                fwd, bwd = sh / ot[li]["size"], sh / ot1[lj]["size"]
                sc = (fwd + bwd) / 2.0 if score == "mean" else min(fwd, bwd)
                pairs[(li, lj)] = (fwd, bwd, sh, sc)
        return pairs

    def domain_exit(o): return (o["outside"] / o["size"]) > DOMAIN_EXIT_FRAC

    def new_track(di, lbl, o, start_type, split_from=-1):
        counters["track"] += 1; tnum, ts = counters["track"], times_pd[di]
        ds_ = ts.strftime("%Y%m%d")
        births_per_date[ds_] = births_per_date.get(ds_, 0) + 1
        tid = f"{ds_}_{births_per_date[ds_]:03d}"; clat, clon = o["centroid"]
        row = {"track_num": tnum, "track_id": tid, "date": ds_, "year": ts.year,
               "jjas_day": jjas_day_num(ts), "day_of_track": 0, "size": o["size"],
               "centroid_lat": round(clat, 3), "centroid_lon": round(clon, 3),
               "merge_event": False, "split_event": False, "link_type": "birth",
               "link_sh": 0, "link_score": np.nan, "start_type": start_type, "end_type": None,
               "duration": None, "split_from_track": split_from, "merge_into_track": -1,
               "day_lbl": lbl}
        active_tracks[tnum] = {"track_id": tid, "last_day": di, "last_lbl": lbl,
                               "start_type": start_type, "duration": 1,
                               "split_from": split_from, "daily_rows": [row]}
        return tnum

    def extend_track(tnum, di, lbl, o, merge_event=False, split_event=False,
                     link_type="overlap", link_sh=0, link_score=np.nan):
        tr = active_tracks[tnum]; tr["duration"] += 1; ts = times_pd[di]; clat, clon = o["centroid"]
        tr["daily_rows"].append(
            {"track_num": tnum, "track_id": tr["track_id"], "date": ts.strftime("%Y%m%d"),
             "year": ts.year, "jjas_day": jjas_day_num(ts), "day_of_track": tr["duration"] - 1,
             "size": o["size"], "centroid_lat": round(clat, 3), "centroid_lon": round(clon, 3),
             "merge_event": merge_event, "split_event": split_event, "link_type": link_type,
             "link_sh": int(link_sh),
             "link_score": (round(float(link_score), 4) if link_score == link_score else np.nan),
             "start_type": tr["start_type"], "end_type": None, "duration": None,
             "split_from_track": tr["split_from"], "merge_into_track": -1, "day_lbl": lbl})
        tr["last_day"], tr["last_lbl"] = di, lbl

    def terminate_track(tnum, end_type, merge_into=-1):
        tr = active_tracks[tnum]; rows = tr["daily_rows"]
        for r in rows: r["duration"] = tr["duration"]
        rows[-1]["end_type"] = end_type; rows[-1]["merge_into_track"] = merge_into
        track_stats_rows.extend(rows); del active_tracks[tnum]

    for lbl, o in get_objects(0).items():
        tn = new_track(0, lbl, o, "season_start")
        if domain_exit(o): terminate_track(tn, "domain_exit")

    for di in range(n_days - 1):
        ts_today, ts_next = times_pd[di], times_pd[di + 1]
        if ts_today.month == 9 and ts_today.day == 30:           # season reset
            for tn in list(active_tracks): terminate_track(tn, "season_end")
            for lbl, o in get_objects(di + 1).items():
                tn = new_track(di + 1, lbl, o, "season_start")
                if domain_exit(o): terminate_track(tn, "domain_exit")
            continue
        ot, ot1 = get_objects(di), get_objects(di + 1)
        if len(ot1) == 0:
            for tn in list(active_tracks): terminate_track(tn, "natural_death")
            continue
        if len(ot) == 0:
            for tn in list(active_tracks): terminate_track(tn, "natural_death")
            for lbl, o in ot1.items():
                tn = new_track(di + 1, lbl, o, "natural_birth")
                if domain_exit(o): terminate_track(tn, "domain_exit")
            continue

        pairs = compute_overlaps(di, di + 1, ot, ot1)
        fwd_links, bwd_links = {}, {}
        for (li, lj), (f, b, sh, sc) in pairs.items():
            fwd_links.setdefault(li, []).append((lj, f, b, sh, sc))
            bwd_links.setdefault(lj, []).append((li, f, b, sh, sc))
        lbl_to_track = {tr["last_lbl"]: tn for tn, tr in active_tracks.items() if tr["last_day"] == di}
        handled_today, handled_next = set(), set()

        # STEP 1 : MERGES (>= 2 parents clear the gate into one object)
        mc = []
        for lj in ot1:
            par = [p for p in bwd_links.get(lj, []) if p[4] >= gate and p[0] in lbl_to_track]
            if len(par) >= 2: mc.append((sum(p[4] for p in par), lj, par))
        mc.sort(key=lambda x: x[0], reverse=True)
        for combined, lj, par in mc:
            if lj in handled_next or any(p[0] in handled_today for p in par): continue
            par = [p for p in par if lbl_to_track.get(p[0]) in active_tracks]
            if len(par) < 2: continue
            par.sort(key=lambda p: ot[p[0]]["size"], reverse=True)
            obj_j = ot1[lj]; dom_tn = lbl_to_track[par[0][0]]
            absorbed_tns = [lbl_to_track[p[0]] for p in par[1:]]
            clat_j, clon_j = obj_j["centroid"]; counters["merge"] += 1
            merge_event_rows.append(
                {"merge_id": counters["merge"], "date": ts_next.strftime("%Y%m%d"),
                 "year": ts_next.year, "jjas_day": jjas_day_num(ts_next), "dominant_track": dom_tn,
                 "absorbed_tracks": ",".join(str(t) for t in absorbed_tns),
                 "n_absorbed": len(absorbed_tns), "merged_size_after": obj_j["size"],
                 "centroid_lat": round(clat_j, 3), "centroid_lon": round(clon_j, 3)})
            extend_track(dom_tn, di + 1, lj, obj_j, merge_event=True, link_type="merge",
                         link_sh=par[0][3], link_score=par[0][4])
            for atn in absorbed_tns: terminate_track(atn, "merge_death", merge_into=dom_tn)
            for p in par: handled_today.add(p[0])
            handled_next.add(lj)

        # STEP 2 : SPLITS (one object into >= 2 children that clear the gate)
        for li, obj_i in ot.items():
            if li in handled_today or li not in lbl_to_track: continue
            par_tn = lbl_to_track[li]
            if par_tn not in active_tracks: continue
            ch = [cc for cc in fwd_links.get(li, []) if cc[4] >= gate and cc[0] not in handled_next]
            if len(ch) < 2: continue
            ch.sort(key=lambda cc: ot1[cc[0]]["size"], reverse=True)
            dom_lj = ch[0][0]; clat_i, clon_i = obj_i["centroid"]
            extend_track(par_tn, di + 1, dom_lj, ot1[dom_lj], split_event=True, link_type="split",
                         link_sh=ch[0][3], link_score=ch[0][4])
            handled_today.add(li); handled_next.add(dom_lj); child_tns = []
            for co in ch[1:]:
                ctn = new_track(di + 1, co[0], ot1[co[0]], "split_birth", split_from=par_tn)
                handled_next.add(co[0]); child_tns.append(ctn)
                if domain_exit(ot1[co[0]]): terminate_track(ctn, "domain_exit")
            counters["split"] += 1
            split_event_rows.append(
                {"split_id": counters["split"], "date": ts_next.strftime("%Y%m%d"),
                 "year": ts_next.year, "jjas_day": jjas_day_num(ts_next), "parent_track": par_tn,
                 "child_tracks": ",".join(str(t) for t in child_tns), "n_children": len(child_tns),
                 "parent_size": obj_i["size"], "centroid_lat": round(clat_i, 3),
                 "centroid_lon": round(clon_i, 3)})

        # STEP 3 : CONTINUATIONS (any non-zero overlap, strongest shared-cell count wins)
        cc = [(sh, li, lj) for (li, lj), (f, b, sh, sc) in pairs.items()
              if li not in handled_today and lj not in handled_next and li in lbl_to_track]
        cc.sort(key=lambda x: x[0], reverse=True)
        for sh, li, lj in cc:
            if li in handled_today or lj in handled_next: continue
            tn = lbl_to_track.get(li)
            if tn is None or tn not in active_tracks: continue
            extend_track(tn, di + 1, lj, ot1[lj], link_type="overlap", link_sh=sh,
                         link_score=pairs[(li, lj)][3])
            handled_today.add(li); handled_next.add(lj)
            if domain_exit(ot1[lj]): terminate_track(tn, "domain_exit")

        # STEP 4 : NO FALLBACK (any active track not extended today dies)
        for tn in list(active_tracks):
            tr = active_tracks.get(tn)
            if tr is not None and tr["last_day"] == di: terminate_track(tn, "natural_death")

        # STEP 5 : BIRTHS
        for lj, obj_j in ot1.items():
            if lj in handled_next: continue
            tn = new_track(di + 1, lj, obj_j, "natural_birth")
            if domain_exit(obj_j): terminate_track(tn, "domain_exit")

    for tn in list(active_tracks): terminate_track(tn, "season_end")

    stats = pd.DataFrame(track_stats_rows)        # rows in termination order, as in the notebook
    srows = []
    for tnum, grp in stats.groupby("track_num"):
        first, last = grp.iloc[0], grp.iloc[-1]
        srows.append({"track_num": tnum, "track_id": first["track_id"], "birth_date": first["date"],
                      "death_date": last["date"], "year": int(first["year"]),
                      "duration": int(first["duration"]), "peak_size": int(grp["size"].max()),
                      "start_type": first["start_type"], "end_type": last["end_type"],
                      "n_merge_events": int(grp["merge_event"].sum()),
                      "n_split_events": int(grp["split_event"].sum())})
    return stats, pd.DataFrame(srows), pd.DataFrame(merge_event_rows), pd.DataFrame(split_event_rows)


# ---------- per-object rainfall statistics (keyed by date + daily label) ----------
date_str = times_pd.strftime("%Y%m%d")
thr_valid = np.where(per_grid_threshold > 900, np.nan, per_grid_threshold)
obj_rows = []
for t in range(n_days):
    ne = int(NE8_raw[t])
    if ne == 0: continue
    idx = np.arange(1, ne + 1)
    mm = ndimage.mean(data_precip[t], Label8[t], idx)
    mx = ndimage.maximum(data_precip[t], Label8[t], idx)
    sm = ndimage.sum(data_precip[t], Label8[t], idx)
    for k, lbl in enumerate(idx):
        tv = thr_valid[Label8[t] == lbl]
        tv = tv[~np.isnan(tv)]
        obj_rows.append((date_str[t], int(lbl), mm[k], mx[k], sm[k], tv.mean() if len(tv) else np.nan))
objects = pd.DataFrame(obj_rows, columns=["date", "day_lbl", "mean_mm", "max_mm", "vol_mm", "thr_mean"])
objects["rel_int"] = objects.mean_mm / objects.thr_mean

# ---------- nearest LPS centre on the same calendar date (cell 17 of the notebook) ----------
lps = pd.read_parquet(DATA / "lps_daily_2000_2022.parquet")
lps = lps[lps.rh850 >= RH850_MIN]
lps["date_str"] = pd.to_datetime(lps["date"]).dt.strftime("%Y%m%d")
lps_by_date = {d: (g.lat.to_numpy(), g.lon.to_numpy(), g.track_id.to_numpy())
               for d, g in lps.groupby("date_str")}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def add_lps(stats):
    n = len(stats)
    near, offE, offN = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    mid = np.full(n, -1, dtype="int64")
    for i, (d, la, lo) in enumerate(zip(stats.date, stats.centroid_lat, stats.centroid_lon)):
        rec = lps_by_date.get(d)
        if rec is None: continue
        clat, clon, cid = rec
        dist = haversine_km(la, lo, clat, clon); j = int(np.argmin(dist))
        near[i], mid[i] = dist[j], cid[j]
        offE[i] = (lo - clon[j]) * 111.320 * np.cos(np.radians(0.5 * (la + clat[j])))  # + = rain east
        offN[i] = (la - clat[j]) * 110.574                                             # + = rain north
    stats["nearest_lps_km"], stats["matched_lps_id"] = near, mid
    stats["offset_E_km"], stats["offset_N_km"] = offE, offN
    stats["matchable"] = ~np.isnan(near)
    return stats


for name, (score, gate) in GATES.items():
    stats, summary, merges, splits = run_tracking(score, gate)
    stats = add_lps(stats.merge(objects, on=["date", "day_lbl"], how="left"))
    assert stats.mean_mm.notna().all()
    stats.to_parquet(DER / f"objdays_{name}.parquet")
    summary.to_parquet(DER / f"tracks_{name}.parquet")
    merges.to_parquet(DER / f"merges_{name}.parquet")
    splits.to_parquet(DER / f"splits_{name}.parquet")
    md = summary[summary.duration >= 3].track_num
    per = stats[stats.matchable & stats.track_num.isin(md)].groupby("track_num").nearest_lps_km.mean()
    print(f"[{name:5s} {score}>={gate:.2f}] tracks={len(summary):,} obj-days={len(stats):,} "
          f"merges={len(merges)} splits={len(splits)} · 3+day near/far/no-LPS = "
          f"{(per <= 500).sum()}/{(per > 500).sum()}/{len(md) - len(per)}")

np.savez_compressed(DER / "imerg_setup.npz", lat=lat, lon=lon,
                    times=np.asarray(times_pd.strftime("%Y%m%d"), dtype="U8"), india_mask=india_mask,
                    per_grid_99=per_grid_99, per_grid_threshold=per_grid_threshold,
                    Label8=Label8.astype("int16"), NE8_raw=NE8_raw)
print("saved catalogue ->", DER)
