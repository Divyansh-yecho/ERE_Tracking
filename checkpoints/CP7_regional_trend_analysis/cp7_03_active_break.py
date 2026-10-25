# cp7_03_active_break.py: EREs in active vs break monsoon spells (README section 3).
# Spells come from the CMZ rainfall z-index (cp7_00) classified on JULY-AUGUST only: in June the
# monsoon is still arriving and in September it is withdrawing, so a JJAS-wide rule labels onset
# and withdrawal days as "breaks" (audited below). EREs of all four months are counted; only the
# spell labels are restricted to Jul-Aug. Caveat: the index is built from core-zone rain, so the
# central-India numbers are partly circular; the absolute Northeast shift is the clean result.
# Also re-runs the classification with every ingredient varied (threshold, box, months).
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ERE_DATA", REPO / "data"))
DER = REPO / "data" / "derived"
FIG = REPO / "figures"
PHASES = ["active", "neutral", "break"]
REGIONS = ["central_india", "himalayan_foothills", "northeast", "west_ghats", "other"]
COLORS = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#999999"]

stats = pd.read_parquet(DER / "cp7_objdays.parquet")
season = pd.read_csv(DER / "cmz_daily_index.csv", dtype={"date": str})
z = np.load(DER / "imerg_setup.npz")
lat, lon, india_mask = z["lat"], z["lon"], z["india_mask"]
times = pd.to_datetime(z["times"], format="%Y%m%d")
n_lat, n_lon = india_mask.shape

# ---------- audit: what the JJAS-wide classification would call a spell ----------
mo = season.date.str[4:6].astype(int)
brk = season.phase == "break"
print(f"JJAS-wide rule: {int(brk.sum())} break days, {int((brk & mo.isin([6, 9])).sum())} of them "
      f"in June/September ({100 * (brk & mo.isin([6, 9])).sum() / brk.sum():.0f}%)")

# ---------- rates per Jul-Aug phase-day ----------
nd = season.phase_ja.value_counts()
sj = stats[stats.phase_ja.isin(PHASES)]
rows = []
for ph in PHASES:
    od = sj[sj.phase_ja == ph]
    first = od[od.day_of_track == 0]
    rows.append(dict(phase=ph, n_days=int(nd[ph]), objdays_per_day=len(od) / nd[ph],
                     mbirths_per_day=len(first[first.duration >= 2]) / nd[ph],
                     vol_per_day=od.vol_mm.sum() / nd[ph], med_size=od["size"].median()))
rates = pd.DataFrame(rows)
comp = pd.crosstab(sj.phase_ja, sj.region, normalize="index").mul(100).reindex(PHASES)
absr = pd.crosstab(sj.phase_ja, sj.region).reindex(PHASES).div(nd.reindex(PHASES), axis=0)
print("\nERE rates by phase (Jul-Aug classification):")
print(rates.round(2).to_string(index=False))
print("\nregional share of object-days (%):")
print(comp[REGIONS].round(1).to_string())
print("\nabsolute object-days per phase-day:")
print(absr[REGIONS].round(2).to_string())
a, b = sj[sj.phase_ja == "active"]["size"], sj[sj.phase_ja == "break"]["size"]
print(f"\nobject size active vs break: median {a.median():.0f} vs {b.median():.0f} "
      f"(MWU p={mannwhitneyu(a, b).pvalue:.0e})")
print(f"activity ratio active/break = {rates.objdays_per_day[0] / rates.objdays_per_day[2]:.1f} · "
      f"multi-day birth ratio = {rates.mbirths_per_day[0] / rates.mbirths_per_day[2]:.1f}")

# ---------- sensitivity: every ingredient of the definition varied ----------
ds = xr.open_dataset(DATA / "imerg_india_025_jjas_2000_2024.nc")
tp = ds["tp"].values.astype("float32"); ds.close()
key = times.strftime("%m%d")


def zindex(la0, la1, lo0, lo1):
    box = india_mask & ((lat >= la0) & (lat <= la1))[:, None] & ((lon >= lo0) & (lon <= lo1))[None, :]
    df = pd.DataFrame({"r": tp[:, box].mean(axis=1), "k": key})
    cm, cs = df.groupby("k").r.mean(), df.groupby("k").r.std()
    cmm = dict(zip(cm.index, pd.Series(cm.to_numpy()).rolling(15, center=True, min_periods=1).mean()))
    css = dict(zip(cs.index, pd.Series(cs.to_numpy()).rolling(15, center=True, min_periods=1).mean()))
    return np.array([(r - cmm[k]) / css[k] for r, k in zip(df.r, key)])


def classify(zv, months, thr):
    mo_, yr = times.month.to_numpy(), times.year.to_numpy()
    inm = np.isin(mo_, months)
    ph = np.array(["excluded"] * len(zv), dtype=object); ph[inm] = "neutral"
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


z_std = zindex(18, 28, 73, 88)
variants = [("Jul-Aug, 18-28N 73-88E, z +-1.0 (used)", z_std, [7, 8], 1.00),
            ("z +-0.75", z_std, [7, 8], 0.75), ("z +-1.25", z_std, [7, 8], 1.25),
            ("alternative box 16-26N 74-86E", zindex(16, 26, 74, 86), [7, 8], 1.00),
            ("wide box 18-28N 65-88E", zindex(18, 28, 65, 88), [7, 8], 1.00),
            ("full JJAS classification", z_std, [6, 7, 8, 9], 1.00)]
print("\nsensitivity of the two headline numbers:")
date_of = times.strftime("%Y%m%d")
for name, zv, months, thr in variants:
    ph = pd.Series(classify(zv, months, thr), index=date_of)
    s = stats.assign(ph=stats.date.map(ph))
    n = ph.value_counts()
    sa, sb = s[s.ph == "active"], s[s.ph == "break"]
    print(f"  {name:42s} days {n['active']:4d}/{n['break']:4d} · activity ratio "
          f"{(len(sa) / n['active']) / (len(sb) / n['break']):.1f} · NE share "
          f"{100 * (sa.region == 'northeast').mean():.0f}% -> {100 * (sb.region == 'northeast').mean():.0f}%")

# ---------- figure ----------
lat_e = np.r_[lat[0] + 0.125, lat - 0.125]
lon_e = np.r_[lon[0] - 0.125, lon + 0.125]
dens = {}
for ph in ["active", "break"]:
    F = np.zeros((n_lat, n_lon))
    g = sj[sj.phase_ja == ph]
    i = np.clip(np.searchsorted(-lat_e, -g.centroid_lat.to_numpy()) - 1, 0, n_lat - 1)
    j = np.clip(np.searchsorted(lon_e, g.centroid_lon.to_numpy()) - 1, 0, n_lon - 1)
    np.add.at(F, (i, j), 1)
    F /= nd[ph]
    F[~india_mask] = np.nan
    dens[ph] = F

fig = plt.figure(figsize=(16, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1])
LONg, LATg = np.meshgrid(lon, lat)
vmax = np.nanmax([np.nanmax(np.where(d > 0, d, np.nan)) for d in dens.values()])
for k, ph in enumerate(["active", "break"]):
    a_ = fig.add_subplot(gs[0, k])
    F = dens[ph]
    pm = a_.pcolormesh(LONg, LATg, np.where(F > 0, F, np.nan), cmap="hot_r",
                       norm=LogNorm(vmin=1e-3, vmax=vmax), shading="auto")
    a_.contour(LONg, LATg, india_mask.astype(int), levels=[0.5], colors="k", linewidths=0.8)
    a_.plot([73, 88, 88, 73, 73], [18, 18, 28, 28, 18], "b--", lw=1.2, label="CMZ box")
    a_.set_xlim(66, 98); a_.set_ylim(6, 33); a_.set_aspect(1.057)
    a_.set_title(f"{ph.upper()} (Jul-Aug) · ERE density/day · {int(nd[ph])} days")
    plt.colorbar(pm, ax=a_, shrink=0.75, label="objects/day (log)")
    a_.legend(loc="lower left", fontsize=8)

axz = fig.add_subplot(gs[0, 2])
one = season[season.date.str.startswith("2013")].reset_index(drop=True)
t = np.arange(len(one))
axz.plot(t, one.cmz_z, "-", color="0.3", lw=1.2)
axz.axhline(1, color="tab:red", ls=":", lw=1); axz.axhline(-1, color="tab:blue", ls=":", lw=1)
for ph, c in [("active", "tab:red"), ("break", "tab:blue")]:
    axz.fill_between(t, -3, 3, where=one.phase_ja == ph, color=c, alpha=0.18)
axz.set_ylim(-3, 3)
axz.set_xticks(one.index[one.date.str[6:8] == "01"]); axz.set_xticklabels(["Jun", "Jul", "Aug", "Sep"])
axz.set_ylabel("CMZ rainfall z-score")
axz.set_title("Example season (2013): z-index and\nJul-Aug active/break spells")
axz.grid(alpha=0.25)

x = np.arange(3)
axb = fig.add_subplot(gs[1, 0])
axb.bar(x - 0.2, rates.objdays_per_day, 0.38, color="tab:blue", label="ERE object-days/day")
axb.bar(x + 0.2, rates.mbirths_per_day * 10, 0.38, color="tab:orange", label="multi-day births/day (x10)")
axb.set_xticks(x); axb.set_xticklabels(PHASES)
axb.set_title("ERE frequency by phase (Jul-Aug)")
axb.legend(fontsize=8); axb.grid(alpha=0.25, axis="y")

axc = fig.add_subplot(gs[1, 1])
bottom = np.zeros(3)
for rgn, c in zip(REGIONS, COLORS):
    axc.bar(x, comp[rgn].to_numpy(), bottom=bottom, label=rgn, color=c, width=0.62)
    bottom += comp[rgn].to_numpy()
axc.set_xticks(x); axc.set_xticklabels(PHASES)
axc.set_ylabel("% of object-days")
axc.set_title("Where EREs occur, by phase (Jul-Aug)")
axc.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))

axd = fig.add_subplot(gs[1, 2])
for ph, c in [("active", "tab:red"), ("break", "tab:blue"), ("neutral", "0.6")]:
    g = sj[sj.phase_ja == ph]["size"]
    xs = np.sort(g)
    axd.loglog(xs, 1 - np.arange(len(xs)) / len(xs), c=c, label=f"{ph} (med {g.median():.0f})")
axd.set_xlabel("object size (cells)"); axd.set_ylabel("P(size > x)")
axd.set_title("Size distribution by phase (Jul-Aug)")
axd.legend(fontsize=8); axd.grid(alpha=0.25)
plt.suptitle("EREs in active vs break monsoon spells — July-August classification", y=1.0, fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "cp7_03_active_break.png", dpi=150, bbox_inches="tight")
print("\nsaved figures/cp7_03_active_break.png")
