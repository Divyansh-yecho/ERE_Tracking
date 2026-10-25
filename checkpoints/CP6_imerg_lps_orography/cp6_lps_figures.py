# cp6_lps_figures.py: LPS co-location statistics and figures, run locally on the
# catalogue from cp6_build_imerg_catalogue.py (cells 17-30 of cp6b_imerg_tracking_lps_colocation).
#   rain-centroid offset from the depression centre, all vs inland centres -> cp6_rain_offset_from_lps.png
#   20 best-matched 3+ day tracks vs their LPS                            -> cp6_lps_best_matched.png
#   10 longest 3+ day tracks vs their best-matching LPS                    -> cp6_lps_longest_tracks.png
#   strict FAR (> 500 km) 3+ day tracks, no-LPS genesis as context         -> cp6_far_tracks_map.png
#   motion of near / far / no-LPS 3+ day tracks                            -> cp6_far_vs_near_motion.png
# Classes (3+ day tracks, mean nearest-LPS distance over days with an LPS): NEAR <= 500 km,
# FAR > 500 km, NO-LPS = no LPS on any day (includes every 2023-24 track; catalogue ends 2022).
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ERE_DATA", REPO / "data"))
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

stats = pd.read_parquet(DER / "objdays_mean.parquet")
summary = pd.read_parquet(DER / "tracks_mean.parquet")
lps = pd.read_parquet(DATA / "lps_daily_2000_2022.parquet")
lps["date"] = pd.to_datetime(lps["date"]).dt.normalize()
z = np.load(DER / "imerg_setup.npz")
LON, LAT = np.meshgrid(z["lon"], z["lat"])
india_mask = z["india_mask"]
dur_of = summary.set_index("track_num").duration
stats["dur"] = stats.track_num.map(dur_of)
by_trk = {tn: g.sort_values("day_of_track").reset_index(drop=True) for tn, g in stats.groupby("track_num")}


def hav(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lo2 - lo1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def classes(dur_min):
    md = summary.loc[summary.duration >= dur_min, "track_num"]
    per = stats[stats.matchable & stats.track_num.isin(md)].groupby("track_num").nearest_lps_km.mean()
    return set(per[per <= 500].index), set(per[per > 500].index), set(md) - set(per.index)


# ---------- date-keyed join and distance statistics ----------
ov = stats[stats.matchable]
md = ov[ov.dur > 2]
print(f"LPS catalogue: {len(lps):,} system-days, {lps.track_id.nunique()} systems "
      f"({lps.groupby(lps.date.dt.year).track_id.nunique().mean():.1f}/season); "
      f"moist filter rh850>=50% keeps {(lps.rh850 >= 50).sum():,}")
print(f"object-days {len(stats):,} | matchable {len(ov):,} | unmatchable {len(stats) - len(ov):,} "
      f"(of which 2023-24: {int((stats.year >= 2023).sum())})")
print(f"nearest-LPS distance, all matchable: median {ov.nearest_lps_km.median():.0f} km, "
      f"{100 * (ov.nearest_lps_km <= 500).mean():.1f}% within 500 km")
print(f"multi-day (> 2 d) object-days: n={len(md)}, median {md.nearest_lps_km.median():.0f} km, "
      f"{100 * (md.nearest_lps_km <= 500).mean():.1f}% within 500 km")
pt = ov.groupby("track_num").agg(dur=("dur", "first"), mdist=("nearest_lps_km", "mean"))
p_len = mannwhitneyu(pt[pt.dur >= 5].mdist, pt[pt.dur <= 2].mdist, alternative="less").pvalue
r_od = spearmanr(ov.dur, ov.nearest_lps_km)
print(f"per track: 1-2 d median {pt[pt.dur <= 2].mdist.median():.0f} km vs 5+ d "
      f"{pt[pt.dur >= 5].mdist.median():.0f} km (MWU p={p_len:.1e}); "
      f"per object-day Spearman rho={r_od.statistic:+.2f} (p={r_od.pvalue:.0e}, negligible effect)")

# ---------- south-west offset, all vs inland centres ----------
cl = stats[stats.matchable & (stats.dur > 2) & (stats.nearest_lps_km <= 500) & (stats.centroid_lat <= 30)].copy()
clat = cl.centroid_lat - cl.offset_N_km / 110.574
clon = cl.centroid_lon - cl.offset_E_km / (111.320 * np.cos(np.radians(0.5 * (cl.centroid_lat + clat))))
ci = cl[clon.between(76, 85) & clat.between(19, 27)]       # centre inland: whole rain shield on land
for d, name in [(cl, "on-system pairs"), (ci, "inland centres")]:
    print(f"{name:16s} n={len(d):3d}: mean offset E {d.offset_E_km.mean():+.0f} km, N {d.offset_N_km.mean():+.0f} km, "
          f"{100 * ((d.offset_E_km < 0) & (d.offset_N_km < 0)).mean():.0f}% in the SW quadrant")

fig, ax = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
for a, (d, ttl) in zip(ax, [(cl, "All on-system matches\n(3+day, ≤500 km, ≤30°N)"),
                            (ci, "Inland depression centres only\n(lon 76-85, lat 19-27 — full shield on land)")]):
    hb = a.hexbin(d.offset_E_km, d.offset_N_km, gridsize=28, cmap="YlOrRd", mincnt=1, vmin=1, vmax=4)
    a.axhline(0, color="k", lw=0.8); a.axvline(0, color="k", lw=0.8)
    a.scatter(d.offset_E_km.mean(), d.offset_N_km.mean(), s=200, marker="+", color="navy", lw=3)
    a.text(-680, 620, "SW", fontsize=11, fontweight="bold", color="0.3")
    a.set_title(f"{ttl}\nmean E {d.offset_E_km.mean():+.0f} km · N {d.offset_N_km.mean():+.0f} km · n={len(d)}")
    a.set_xlim(-700, 700); a.set_ylim(-700, 700); a.set_aspect("equal")
    a.set_xlabel("E offset (km)  → rain east of centre")
ax[0].set_ylabel("N offset (km)  → rain north of centre")
fig.colorbar(hb, ax=ax, shrink=0.8, label="object-days")
fig.suptitle("Rain-centroid offset from the depression centre — the ~200 km westward displacement "
             "survives inland (physical, not a land-clip artifact)", fontsize=12)
plt.savefig(FIG / "cp6_rain_offset_from_lps.png", dpi=120, bbox_inches="tight"); plt.close()

# ---------- track-by-track matches ----------
LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = 66, 98, 6, 32
lps_clip = lps[lps.lon.between(LON_MIN, LON_MAX) & lps.lat.between(LAT_MIN, LAT_MAX)]
cand = summary[(summary.duration > 2) & summary.year.isin(set(lps_clip.date.dt.year))]
BANDS = [(100, "#00a651", "0-100 km"), (200, "#a6d96a", "100-200 km"), (400, "#fdae61", "200-400 km"),
         (500, "#fee08b", "400-500 km")]


def band_color(d):
    if d != d: return "black"                                   # no LPS that day
    for lim, c, _ in BANDS:
        if d <= lim: return c
    return "black"


def shared(tn, sys_id=None):
    g = by_trk[tn]; gd = pd.to_datetime(g.date, format="%Y%m%d")
    sub = lps_clip[lps_clip.date.isin(gd)]
    return g, gd, (sub if sys_id is None else sub[sub.track_id == sys_id])


def match_panel(ax, tn, sys_id, title):
    g, gd, sysc = shared(tn, sys_id)
    sysc = sysc.sort_values("date"); sbd = dict(zip(sysc.date, zip(sysc.lat, sysc.lon)))
    ax.contour(LON, LAT, india_mask.astype(int), levels=[0.5], colors="0.6", linewidths=0.7, zorder=1)
    for d, la, lo in zip(gd, g.centroid_lat, g.centroid_lon):             # same-day threads
        if d in sbd:
            ax.plot([lo, sbd[d][1]], [la, sbd[d][0]], "-", color="0.6", lw=0.6, alpha=0.6, zorder=2)
    ax.plot(sysc.lon, sysc.lat, "-", color="tab:blue", lw=1.7, alpha=0.9, zorder=3)
    ax.scatter(sysc.lon, sysc.lat, s=42, color="tab:blue", edgecolor="white", lw=0.5, zorder=4)
    for k in range(len(g) - 1):
        c = "red" if g.merge_event.iloc[k + 1] else ("green" if g.split_event.iloc[k + 1] else "black")
        ax.plot(g.centroid_lon.iloc[k:k + 2], g.centroid_lat.iloc[k:k + 2], "-", color=c, lw=1.6, alpha=0.85, zorder=5)
    cols = [band_color(hav(la, lo, *sbd[d]) if d in sbd else np.nan)
            for d, la, lo in zip(gd, g.centroid_lat, g.centroid_lon)]
    ax.scatter(g.centroid_lon, g.centroid_lat, c=cols, s=44, edgecolor="k", lw=0.4, zorder=6)
    ax.scatter(g.centroid_lon.iloc[0], g.centroid_lat.iloc[0], s=190, facecolor="none", edgecolor="k", lw=1.6, zorder=7)
    ax.set_title(title, fontsize=9)
    xs, ys = pd.concat([g.centroid_lon, sysc.lon]), pd.concat([g.centroid_lat, sysc.lat])
    ax.set_xlim(max(LON_MIN, xs.min() - 3), min(LON_MAX, xs.max() + 3))
    ax.set_ylim(max(LAT_MIN, ys.min() - 3), min(LAT_MAX, ys.max() + 3)); ax.set_aspect(1.057)


LEGEND = ([Line2D([0], [0], color="tab:blue", marker="o", lw=1.7, label="LPS centre (shared days)"),
           Line2D([0], [0], color="black", lw=1.6, label="rain track"),
           Line2D([0], [0], color="red", lw=1.6, label="merge day"),
           Line2D([0], [0], color="green", lw=1.6, label="split day"),
           Line2D([0], [0], color="0.6", lw=0.6, label="same-day thread"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="none", markeredgecolor="k",
                  markersize=12, label="birth (ring)")]
          + [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=l) for _, c, l in BANDS]
          + [Line2D([0], [0], marker="o", color="w", markerfacecolor="black", markersize=9, label=">500 km / no LPS")])


def pair_stats(tn, min_shared):
    g, gd, sub = shared(tn)
    pts = dict(zip(gd, zip(g.centroid_lat, g.centroid_lon)))
    for sys_id, sc in sub.groupby("track_id"):
        d = np.array([hav(*pts[day], la, lo) for day, la, lo in zip(sc.date, sc.lat, sc.lon)])
        if len(d) >= min_shared:
            yield sys_id, len(d), d.mean(), d.min()


scored = sorted([(m, tn, s, n, mn) for tn in cand.track_num for s, n, m, mn in pair_stats(tn, 3)])[:20]
fig, axes = plt.subplots(4, 5, figsize=(24, 20))
for ax, (m, tn, s, n, mn) in zip(axes.ravel(), scored):
    match_panel(ax, tn, s, f"track {tn} ({dur_of[tn]}d) ↔ LPS {s}\n{n} shared d · mean {m:.0f} km · closest {mn:.0f} km")
fig.legend(handles=LEGEND, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("20 best-matched 3+day rain tracks vs their LPS system (smallest mean same-day separation)",
             fontsize=14, y=1.0)
plt.tight_layout(); plt.savefig(FIG / "cp6_lps_best_matched.png", dpi=85, bbox_inches="tight"); plt.close()
print("best-matched pairs: mean separation "
      f"{scored[0][0]:.0f}-{scored[-1][0]:.0f} km, closest approach {min(x[4] for x in scored):.0f} km")

panels = []
for tn in cand.sort_values(["duration", "peak_size"], ascending=False).track_num:
    best = max(((n, -m, s, m, mn) for s, n, m, mn in pair_stats(tn, 1)), default=None)  # most shared days
    if best is not None:
        panels.append((tn, best[2], best[0], best[3], best[4]))
    if len(panels) == 10: break
fig, axes = plt.subplots(2, 5, figsize=(24, 11))
for ax, (tn, s, n, m, mn) in zip(axes.ravel(), panels):
    match_panel(ax, tn, s, f"track {tn} ({dur_of[tn]}d) ↔ LPS {s}\n{n} shared d · mean {m:.0f} km · closest {mn:.0f} km")
fig.legend(handles=LEGEND, loc="lower center", ncol=6, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("10 longest 3+day rain tracks vs best-matching LPS (ranked by duration, not closeness)",
             fontsize=14, y=1.0)
plt.tight_layout(); plt.savefig(FIG / "cp6_lps_longest_tracks.png", dpi=85, bbox_inches="tight"); plt.close()

# ---------- genesis geography and the FAR-track map ----------
near3, far3, nolps3 = classes(3)
print(f"\n3+ day tracks: near {len(near3)} · far (strict) {len(far3)} · no-LPS {len(nolps3)}")


def genesis_region(la, lo):
    if 72 <= lo <= 76.5 and 8 <= la <= 21: return "W.Ghats"
    if la > 27: return "FarNorth"
    if lo > 90 and la > 22: return "NE.hills"
    if 75 <= lo <= 88 and 18 <= la <= 27: return "C.India"
    return "Other"


for tns, name in [(near3, "NEAR"), (far3, "FAR strict"), (nolps3, "NO-LPS")]:
    b = pd.Series([genesis_region(by_trk[t].centroid_lat.iloc[0], by_trk[t].centroid_lon.iloc[0]) for t in tns])
    print(f"  genesis {name:10s}: " + ", ".join(f"{k} {v:.0f}%" for k, v in (b.value_counts(normalize=True) * 100).items()))

fig, ax = plt.subplots(figsize=(12, 13))
ax.contour(LON, LAT, india_mask.astype(int), levels=[0.5], colors="0.5", linewidths=0.9, zorder=2)
ax.axvline(76.5, color="saddlebrown", ls=":", lw=1, alpha=0.6); ax.axhline(27, color="slateblue", ls=":", lw=1, alpha=0.6)
ax.text(74.3, 9, "W.Ghats", color="saddlebrown", fontsize=9, rotation=90, alpha=0.8)
ax.text(67, 27.4, "foothill belt (>27°N)", color="slateblue", fontsize=9, alpha=0.8)
for tn in nolps3:
    ax.scatter(by_trk[tn].centroid_lon.iloc[0], by_trk[tn].centroid_lat.iloc[0], s=12, color="0.7", alpha=0.5, zorder=3)
for tn in far3:
    g = by_trk[tn]
    for k in range(len(g) - 1):
        c = "red" if g.merge_event.iloc[k + 1] else ("green" if g.split_event.iloc[k + 1] else "0.25")
        ax.plot(g.centroid_lon.iloc[k:k + 2], g.centroid_lat.iloc[k:k + 2], "-", color=c, lw=1.0, alpha=0.6, zorder=4)
    ax.scatter(g.centroid_lon.iloc[0], g.centroid_lat.iloc[0], s=16, color="tab:blue", alpha=0.8, zorder=5)
ax.legend(handles=[Line2D([0], [0], color="0.25", lw=1.4, label="FAR strict track (>500 km, matchable)"),
                   Line2D([0], [0], color="red", lw=1.4, label="merge segment"),
                   Line2D([0], [0], color="green", lw=1.4, label="split segment"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, label="far genesis"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor="0.7", markersize=7,
                          label="no-LPS genesis (context)")], loc="lower left", fontsize=9)
ax.set_title(f"{len(far3)} strict far multi-day tracks (>500 km from any LPS) · {len(nolps3)} no-LPS as context\n"
             "(where non-depression rain accumulates — Ghats + foothills)", fontsize=11)
ax.set_xlim(66, 98); ax.set_ylim(6, 32); ax.set_aspect(1.057)
ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
plt.tight_layout(); plt.savefig(FIG / "cp6_far_tracks_map.png", dpi=100, bbox_inches="tight"); plt.close()


# ---------- motion (figure uses the 3+ day classes of the last loop pass) ----------
def track_motion(tns):
    rows = []
    for tn in tns:
        g = by_trk[tn]
        la, lo = g.centroid_lat.to_numpy(), g.centroid_lon.to_numpy()
        steps = hav(la[:-1], lo[:-1], la[1:], lo[1:]); path = steps.sum()
        rows.append(dict(mean_step=steps.mean(), straightness=hav(la[0], lo[0], la[-1], lo[-1]) / path
                         if path > 0 else np.nan))
    return pd.DataFrame(rows)


for dmin in [2, 3]:
    nr, fr, nl = classes(dmin)
    N, F, NL = track_motion(nr), track_motion(fr), track_motion(nl)
    p = mannwhitneyu(F.mean_step, N.mean_step, alternative="less").pvalue
    delta = 2 * mannwhitneyu(F.mean_step, N.mean_step).statistic / (len(F) * len(N)) - 1
    print(f"{dmin}+ day motion: near {N.mean_step.median():.0f} (n={len(N)}) · far {F.mean_step.median():.0f} "
          f"(n={len(F)}) · no-LPS {NL.mean_step.median():.0f} km/day (n={len(NL)}) · far<near p={p:.1e}, "
          f"Cliff delta={delta:+.2f} · straightness near {N.straightness.median():.2f} far {F.straightness.median():.2f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
bins_s, bins_d = np.linspace(0, 300, 31), np.linspace(0, 1, 26)
for key, a, bins in [("mean_step", ax[0], bins_s), ("straightness", ax[1], bins_d)]:
    for D, c, lab in [(N, "tab:blue", "NEAR (≤500 km)"), (F, "tab:red", "FAR strict (>500 km)")]:
        a.hist(D[key].dropna(), bins=bins, alpha=0.5, color=c, density=True,
               label=f"{lab}  med {D[key].median():.0f}" if key == "mean_step" else f"{lab}  med {D[key].median():.2f}")
    a.hist(NL[key].dropna(), bins=bins, histtype="step", ls=":", color="k", lw=1.4, density=True,
           label=f"no-LPS  med {NL[key].median():.0f}" if key == "mean_step" else f"no-LPS  med {NL[key].median():.2f}")
    if key == "mean_step":
        for D, c in [(N, "tab:blue"), (F, "tab:red")]:
            a.axvline(D[key].median(), color=c, ls="--", lw=1.2)
    a.legend(fontsize=8); a.set_ylabel("density")
ax[0].set_xlabel("mean daily displacement (km/day)")
ax[0].set_title(f"Daily step — anchored vs propagating\nfar strict {F.mean_step.median():.0f} vs near "
                f"{N.mean_step.median():.0f} km/day · p={p:.1e} · δ={delta:+.2f}", fontsize=10)
ax[1].set_xlabel("straightness (net / path length)")
ax[1].set_title("Directedness (3+ day) — ~1 propagate, ~0 anchored\n(weak discriminator)", fontsize=10)
fig.suptitle("Motion of far vs near multi-day tracks (3+ day, strict classes: near / far / no-LPS)", fontsize=12)
plt.tight_layout(); plt.savefig(FIG / "cp6_far_vs_near_motion.png", dpi=120, bbox_inches="tight"); plt.close()
print("saved figures/cp6_{rain_offset_from_lps,lps_best_matched,lps_longest_tracks,far_tracks_map,far_vs_near_motion}.png")
