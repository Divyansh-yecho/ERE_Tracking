# cp6_lps_null.py: track-level re-dating null for rain-track / LPS co-location (cells 22-23 of
# cp6b_imerg_tracking_lps_colocation; figure: figures/cp6_lps_null.png).
# Each 3+ day track is kept rigid (positions and calendar days) and re-dated to a random other
# catalogue year (2000-2022): the seasonal cycle and the track's own autocorrelation survive, the
# real-year alignment with the depressions does not. The track, not the object-day, is the unit.
# Pool: only tracks with >= 1 LPS day in their real year (125). The first version of the null
# also let the 31 never-matched tracks (incl. every 2023-24 track) enter the shuffles although
# they cannot enter the observed fraction; that asymmetric pool is reported for comparison.
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ERE_DATA", REPO / "data"))
DER = REPO / "data" / "derived"
FIG = REPO / "figures"
B, SEEDS, RADII = 5000, range(10), (200, 300, 500)

stats = pd.read_parquet(DER / "objdays_mean.parquet")
summary = pd.read_parquet(DER / "tracks_mean.parquet")
lps = pd.read_parquet(DATA / "lps_daily_2000_2022.parquet")
lps["date"] = pd.to_datetime(lps["date"])
lps_ymd = {(d.year, d.month, d.day): (g.lat.to_numpy(), g.lon.to_numpy()) for d, g in lps.groupby("date")}
YEARS = np.arange(lps.date.dt.year.min(), lps.date.dt.year.max() + 1)


def hav(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


md = summary[summary.duration > 2]
matchable_real = set(stats.loc[stats.matchable, "track_num"])
rigid = []
for tn, yr in zip(md.track_num, md.year):
    g = stats[stats.track_num == tn].sort_values("day_of_track")
    days = pd.to_datetime(g.date, format="%Y%m%d")
    rigid.append((tn in matchable_real, int(yr), g.centroid_lat.to_numpy(), g.centroid_lon.to_numpy(),
                  list(zip(days.dt.month, days.dt.day))))

# per (track, candidate year): number of LPS days, hits within each radius, mean distance
n_trk, n_yr = len(rigid), len(YEARS)
ndays = np.zeros((n_trk, n_yr)); hits = np.zeros((len(RADII), n_trk, n_yr)); mdist = np.full((n_trk, n_yr), np.nan)
for i, (_, _, la, lo, md_) in enumerate(rigid):
    for j, y in enumerate(YEARS):
        d = [hav(a, b, *lps_ymd[(y, m, dd)]).min() for a, b, (m, dd) in zip(la, lo, md_) if (y, m, dd) in lps_ymd]
        if d:
            d = np.array(d)
            ndays[i, j], mdist[i, j] = len(d), d.mean()
            hits[:, i, j] = [(d <= R).sum() for R in RADII]


def fractions(yidx, rows):
    """co-location fractions (per object-day, per track) for year choices yidx[..., track]."""
    r = np.asarray(rows)
    nd = ndays[r, yidx]; has = nd > 0
    od = hits[:, r, yidx].sum(axis=-1) / nd.sum(axis=-1)
    tm = np.stack([((mdist[r, yidx] <= R) & has).sum(axis=-1) / has.sum(axis=-1) for R in RADII])
    return od, tm


def run(rows, label):
    own = np.array([np.searchsorted(YEARS, rigid[i][1]) if rigid[i][1] in YEARS else -1 for i in rows])
    null_od, null_tm = [], []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        k = rng.integers(0, n_yr - 1, size=(B, len(rows)))           # random OTHER year per track
        yidx = np.where((own >= 0) & (k >= own), k + 1, k)
        if (own < 0).any():                                             # 2023-24 tracks: any catalogue year
            yidx[:, own < 0] = rng.integers(0, n_yr, size=(B, int((own < 0).sum())))
        a, b = fractions(yidx, rows)
        null_od.append(a); null_tm.append(b)
    null_od, null_tm = np.concatenate(null_od, axis=1), np.concatenate(null_tm, axis=1)
    print(f"\n{label}: {len(rows)} tracks, {B * len(SEEDS):,} shuffles")
    return null_od, null_tm


pool = [i for i, r in enumerate(rigid) if r[0]]
own_pool = np.array([np.searchsorted(YEARS, rigid[i][1]) for i in pool])
obs_od, obs_tm = fractions(own_pool, pool)
print(f"observed: {len(pool)} tracks with >= 1 LPS day ({int(ndays[pool, own_pool].sum())} object-days)")
results = {}
for name, rows in [("symmetric pool (used)", pool), ("asymmetric pool (first version)", list(range(n_trk)))]:
    null_od, null_tm = run(rows, name)
    results[name] = null_tm
    for k, R in enumerate(RADII):
        for unit, o, nl in [("per track", obs_tm[k], null_tm[k]), ("per obj-day", obs_od[k], null_od[k])]:
            p = (np.sum(nl >= o) + 1) / (len(nl) + 1)
            print(f"  <= {R} km {unit:11s} observed {100 * o:4.1f}% · chance {100 * nl.mean():4.1f}% "
                  f"+- {100 * nl.std():.1f}% · enrichment {o / nl.mean():.1f}x · p = {p:.1e}")

null_tm = results["symmetric pool (used)"]
fig, axes = plt.subplots(1, len(RADII), figsize=(5 * len(RADII), 4.2))
for ax, k, R in zip(axes, range(len(RADII)), RADII):
    pooled, o = 100 * null_tm[k], 100 * obs_tm[k]
    ax.hist(pooled, bins=np.arange(-0.4, pooled.max() + 1.2, 0.8), color="0.7")
    ax.axvline(o, color="crimson", lw=2.5, label=f"observed {o:.1f}%")
    ax.axvline(pooled.mean(), color="navy", lw=1.5, ls="--", label=f"null {pooled.mean():.1f}%")
    ax.set_title(f"≤{R} km · per track · {o / pooled.mean():.1f}× chance", fontsize=11)
    ax.set_xlabel("% of tracks within R"); ax.legend(fontsize=8)
    ax.set_xlim(0, max(o * 1.15, pooled.max() * 1.1))
fig.suptitle(f"Observed co-location vs track-level null ({B * len(SEEDS):,} year-shuffles of "
             f"{len(pool)} tracks) — observed sits far outside the chance cloud", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(FIG / "cp6_lps_null.png", dpi=120, bbox_inches="tight")
print("\nsaved figures/cp6_lps_null.png")
