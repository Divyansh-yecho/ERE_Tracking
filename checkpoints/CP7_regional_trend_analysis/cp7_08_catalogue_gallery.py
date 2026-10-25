# cp7_08_catalogue_gallery.py: visual outputs of the IMERG track catalogue (README: The catalogue):
# the ten longest tracks, duration vs peak size, where objects live and
# are born, and the life cycles of the longest tracks. Node labels read (mean rain mm/day · size).
# "Longest" = sorted by duration, ties broken by peak size. Merge/split markers use the
# catalogue's own MEAN-gate flags, as in the CP6 notebook maps.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

stats = pd.read_parquet(DER / "cp7_objdays.parquet")
tracks = pd.read_parquet(DER / "cp7_tracks.parquet")
z = np.load(DER / "imerg_setup.npz")
lat, lon, india_mask = z["lat"], z["lon"], z["india_mask"]
LON, LAT = np.meshgrid(lon, lat)
by_trk = {tn: g.sort_values("day_of_track").reset_index(drop=True) for tn, g in stats.groupby("track_num")}
top10 = tracks.sort_values(["duration", "peak_size"], ascending=False).head(10).track_num.tolist()
OFFSETS = [(-45, 22), (25, -28), (-50, -24), (30, 24), (-20, 34), (35, -10), (-55, 6), (10, -38)]


def outline(ax):
    ax.contour(LON, LAT, india_mask.astype(int), levels=[0.5], colors="0.45", linewidths=0.8, zorder=1)


def frame(ax, g, pad=1.0):
    x0, x1 = g.centroid_lon.min() - pad, g.centroid_lon.max() + pad
    y0, y1 = g.centroid_lat.min() - pad, g.centroid_lat.max() + pad
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect(1 / np.cos(np.radians(0.5 * (y0 + y1))))


def label_nodes(ax, g, fs=6.5):
    for k, r in enumerate(g.itertuples()):
        ax.annotate(f"{r.mean_mm:.0f}mm·{r.size}c", (r.centroid_lon, r.centroid_lat),
                    xytext=OFFSETS[k % len(OFFSETS)], textcoords="offset points", fontsize=fs,
                    color="tab:blue", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.8", lw=0.5),
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6), zorder=8)


def path_panel(ax, tn, by_tau=False, labels=True):
    g = by_trk[tn]
    tau = g.day_of_track / max(len(g) - 1, 1)
    outline(ax)
    ax.plot(g.centroid_lon, g.centroid_lat, "-", color="0.25", lw=1.4, zorder=3)
    if by_tau:
        sc = ax.scatter(g.centroid_lon, g.centroid_lat, c=tau, cmap="viridis", vmin=0, vmax=1,
                        s=12 * np.sqrt(g["size"]) + 20, edgecolor="k", lw=0.4, zorder=5)
        ax.scatter(g.centroid_lon.iloc[0], g.centroid_lat.iloc[0], s=110, facecolor="none",
                   edgecolor="k", lw=1.4, zorder=6)
    else:
        sc = None
        ax.scatter(g.centroid_lon, g.centroid_lat, s=10, color="k", zorder=4)
        ax.scatter(g.centroid_lon.iloc[0], g.centroid_lat.iloc[0], s=40, color="tab:blue", zorder=6)
        mg, sp = g[g.merge_event], g[g.split_event]
        ax.scatter(mg.centroid_lon, mg.centroid_lat, s=40, marker="s", color="red", zorder=6)
        ax.scatter(sp.centroid_lon, sp.centroid_lat, s=40, marker="s", color="green", zorder=6)
    ax.scatter(g.centroid_lon.iloc[-1], g.centroid_lat.iloc[-1], s=45, marker="x", color="k",
               lw=1.6, zorder=7)
    if labels: label_nodes(ax, g)
    frame(ax, g)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.2)
    return sc


def title_of(tn):
    t = tracks.set_index("track_num").loc[tn]
    return f"track {tn} · {t.duration} d · peak {t.peak_size} c · {t.birth_date}→{by_trk[tn].date.iloc[-1]}"


# ---------- N1: the ten longest tracks ----------
fig, axes = plt.subplots(5, 2, figsize=(11, 23))
for ax, tn in zip(axes.ravel(), top10):
    path_panel(ax, tn)
    ax.set_title(title_of(tn), fontsize=9)
fig.suptitle("10 longest tracks · birth ● death ✕ merge ■ (red) split ■ (green) · "
             "blue = (mean intensity mm/day · size cells)", fontsize=12, y=0.995)
plt.tight_layout()
plt.savefig(FIG / "cp7_08_longest_tracks.png", dpi=110, bbox_inches="tight"); plt.close()

# ---------- N2: duration and peak size ----------
edges = [0.5, 1.5, 2.5, 4.5, 9.5, 19.5, 49.5, np.inf]
labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
tracks["szbin"] = pd.cut(tracks.peak_size, edges, labels=labels)
comp = pd.crosstab(tracks.duration, tracks.szbin, normalize="index").mul(100)
durs = sorted(tracks.duration.unique())
rho = spearmanr(tracks.duration, tracks.peak_size).statistic
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
bottom = np.zeros(len(comp))
for lab, c in zip(labels, plt.cm.viridis(np.linspace(0, 1, len(labels)))):
    v = comp.get(lab, pd.Series(0, index=comp.index)).to_numpy()
    ax[0].bar(comp.index, v, bottom=bottom, color=c, label=lab, width=0.8)
    bottom += v
ax[0].set_xlabel("duration (days)"); ax[0].set_ylabel("% of tracks (size composition)")
ax[0].set_title("Size composition within each duration")
ax[0].legend(title="peak size (cells)", fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
data = [tracks[tracks.duration == d].peak_size for d in durs]
ax[1].boxplot(data, positions=durs, showfliers=False)
ax[1].plot(durs, [np.median(d) for d in data], "o-", color="tab:red", label="median")
ax[1].set_yscale("log"); ax[1].set_xlabel("duration (days)"); ax[1].set_ylabel("peak size (cells, log)")
ax[1].set_title(f"Peak size vs duration · rho={rho:.2f}"); ax[1].legend(); ax[1].grid(alpha=0.25)
plt.tight_layout()
plt.savefig(FIG / "cp7_08_size_vs_duration.png", dpi=130, bbox_inches="tight"); plt.close()
print(f"Spearman(duration, peak size) = {rho:.2f}; median peak size by duration:",
      {d: float(np.median(x)) for d, x in zip(durs, data)})

# ---------- N3: object-day and genesis density ----------
lat_e = np.r_[lat[0] + 0.125, lat - 0.125]
lon_e = np.r_[lon[0] - 0.125, lon + 0.125]


def density(la, lo):
    F = np.zeros(india_mask.shape)
    i = np.clip(np.searchsorted(-lat_e, -np.asarray(la)) - 1, 0, len(lat) - 1)
    j = np.clip(np.searchsorted(lon_e, np.asarray(lo)) - 1, 0, len(lon) - 1)
    np.add.at(F, (i, j), 1)
    return np.where(F > 0, F, np.nan)


births = stats[stats.day_of_track == 0]
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
for a, F, ttl in [(ax[0], density(stats.centroid_lat, stats.centroid_lon), "Track density (all object-day centroids)"),
                  (ax[1], density(births.centroid_lat, births.centroid_lon), "Genesis density (track birthpoints)")]:
    pm = a.pcolormesh(LON, LAT, F, cmap="YlOrRd", norm=LogNorm(vmin=1, vmax=np.nanmax(F)), shading="auto")
    a.contour(LON, LAT, india_mask.astype(int), levels=[0.5], colors="k", linewidths=0.8)
    a.set_xlim(66, 98); a.set_ylim(6, 34); a.set_aspect(1.057)
    a.set_title(ttl); a.set_xlabel("longitude"); a.set_ylabel("latitude")
    plt.colorbar(pm, ax=a, shrink=0.8, label="count (log)")
fig.suptitle("Where extreme-rain systems live vs where they are born (JJAS 2000-2024)", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp7_08_density.png", dpi=130, bbox_inches="tight"); plt.close()


# ---------- N4 and N9: life cycles ----------
def magnitudes(ax, tn, title="raw magnitudes"):
    g = by_trk[tn]
    ax.plot(g.day_of_track, g["size"], "o-", color="tab:green", label="size (cells)")
    ax.set_xlabel("day of track"); ax.set_ylabel("size (cells)", color="tab:green")
    ax2 = ax.twinx()
    ax2.plot(g.day_of_track, g.mean_mm, "s-", color="tab:red", label="intensity (mm/day)")
    ax2.set_ylabel("intensity (mm/day)", color="tab:red")
    ax.legend(handles=ax.get_lines() + ax2.get_lines(), fontsize=7, loc="upper right")
    ax.grid(alpha=0.25); ax.set_title(title, fontsize=9)


tn = top10[0]
g = by_trk[tn]
tau = g.day_of_track / (len(g) - 1)
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
a = fig.add_subplot(gs[0, 0])
for col, c, lab in [("size", "tab:green", "size"), ("mean_mm", "tab:red", "intensity"), ("vol_mm", "tab:purple", "volume")]:
    a.plot(tau, g[col] / g[col].max(), "o-", color=c, label=lab)
for t in tau[g.merge_event.to_numpy()]:
    a.axvline(t, color="red", ls="--", lw=1)
a.set_xlabel("normalized lifetime tau (0 birth -> 1 death)"); a.set_ylabel("fraction of own peak")
a.set_title(f"track {tn} · normalized life cycle (dashed = merge days)", fontsize=10)
a.legend(fontsize=8); a.grid(alpha=0.25)
magnitudes(fig.add_subplot(gs[0, 1]), tn)
a = fig.add_subplot(gs[1, :])
sc = path_panel(a, tn, by_tau=True)
plt.colorbar(sc, ax=a, shrink=0.8, label="lifetime fraction tau")
a.set_title("geographic path · ring = birth · ✕ = death · colour = tau · marker size ∝ √footprint", fontsize=10)
fig.suptitle(f"Life cycle of the longest track {tn} · {g.date.iloc[0]}→{g.date.iloc[-1]}", fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp7_08_longest_track_lifecycle.png", dpi=120, bbox_inches="tight"); plt.close()

fig, axes = plt.subplots(10, 2, figsize=(13, 36), layout="constrained",
                         gridspec_kw=dict(width_ratios=[1, 1.2]))
for row, tn in zip(axes, top10):                              # one track per row
    magnitudes(row[0], tn, title=title_of(tn))
    sc = path_panel(row[1], tn, by_tau=True)
fig.colorbar(sc, ax=axes[:, 1], shrink=0.25, label="lifetime fraction tau")
fig.suptitle("Ten longest tracks: raw size and intensity (left), path coloured by lifetime fraction\n"
             "(right; ring = birth, ✕ = death, marker size ∝ √footprint)", fontsize=13)
plt.savefig(FIG / "cp7_08_lifecycles_top10.png", dpi=90); plt.close()
print("ten longest tracks:", top10)
print("saved figures/cp7_08_{longest_tracks,size_vs_duration,density,longest_track_lifecycle,lifecycles_top10}.png")
