# cp7_09_merge_split_events.py: maps of individual merge and split events under the MIN-0.05
# gate: the ten largest two-parent merges, the ten largest two-daughter
# splits, and every event with more than two participants. Merge: parents on the day before (red
# dots) join into the product (black square). Split: the parent on the day before (black square)
# separates into daughters (green dots). Labels read (mean rain mm/day · size cells).
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

stats = pd.read_parquet(DER / "objdays_min05.parquet")
merges = pd.read_parquet(DER / "merges_min05.parquet")
splits = pd.read_parquet(DER / "splits_min05.parquet")
z = np.load(DER / "imerg_setup.npz")
LON, LAT = np.meshgrid(z["lon"], z["lat"])
india_mask = z["india_mask"]
pos = stats.set_index(["track_num", "date"])


def prev_day(d):
    return (pd.to_datetime(d, format="%Y%m%d") - pd.Timedelta(days=1)).strftime("%Y%m%d")


def node(tn, d):
    r = pos.loc[(int(tn), d)]
    return r.centroid_lon, r.centroid_lat, f"{r.mean_mm:.0f}mm·{r['size']}c"


def event_panel(ax, centre, others, colour, title, centre_tag):
    """centre = (lon, lat, label) drawn as a black square; others joined to it by lines."""
    ax.contour(LON, LAT, india_mask.astype(int), levels=[0.5], colors="0.45", linewidths=0.8)
    pts = [centre] + others
    for k, (x, y, lab) in enumerate(others):
        ax.plot([x, centre[0]], [y, centre[1]], "-", color=colour, lw=2, zorder=3)
        ax.scatter(x, y, s=36, color=colour, zorder=4)
    ax.scatter(centre[0], centre[1], s=70, marker="s", color="k", zorder=5)
    # each label points away from the centre; the centre's label goes opposite the mean direction
    units = []
    for x, y, _ in others:
        v = np.array([x - centre[0], y - centre[1]])
        units.append(v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([0.0, 1.0]))
    away = -np.mean(units, axis=0)
    away = away / np.linalg.norm(away) if np.linalg.norm(away) > 0.1 else np.array([0.0, -1.0])
    for k, (x, y, lab) in enumerate(pts):
        u = away if k == 0 else units[k - 1]
        ax.annotate((centre_tag + "\n" if k == 0 else "") + lab, (x, y), xytext=tuple(40 * u),
                    textcoords="offset points", fontsize=7, color="tab:blue", fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.8", lw=0.5),
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6), zorder=6)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = max(0.6, 0.25 * max(np.ptp(xs), np.ptp(ys)))
    ax.set_xlim(min(xs) - pad, max(xs) + pad); ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect(1 / np.cos(np.radians(np.mean(ys))))
    ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7); ax.grid(alpha=0.2)


def merge_panel(ax, r):
    parents = [int(r.dominant_track)] + [int(t) for t in str(r.absorbed_tracks).split(",") if t.strip()]
    others = [node(p, prev_day(r.date)) for p in parents]
    event_panel(ax, node(r.dominant_track, r.date), others, "red",
                f"M{r.merge_id} · {len(parents)} parents · {r.date}", f"M{r.merge_id}")


def split_panel(ax, r):
    daughters = [int(r.parent_track)] + [int(t) for t in str(r.child_tracks).split(",") if t.strip()]
    others = [node(d, r.date) for d in daughters]
    event_panel(ax, node(r.parent_track, prev_day(r.date)), others, "green",
                f"S{r.split_id} · {len(daughters)} daughters · {r.date}", f"S{r.split_id}")


def gallery(rows, panel, title, fname, ncol=2):
    n = len(rows); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow), squeeze=False)
    for ax, r in zip(axes.ravel(), rows.itertuples()):
        panel(ax, r)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=110, bbox_inches="tight"); plt.close()


key = "· blue = (mean intensity mm/day · size cells)"
two_m = merges[merges.n_absorbed == 1].sort_values("merged_size_after", ascending=False).head(10)
two_s = splits[splits.n_children == 1].sort_values("parent_size", ascending=False).head(10)
gallery(two_m, merge_panel, f"MIN-0.05 gate · 10 largest two-parent merges {key}", "cp7_09_merges_two_parent.png")
gallery(two_s, split_panel, f"MIN-0.05 gate · 10 largest two-daughter splits {key}", "cp7_09_splits_two_daughter.png")
multi_m, multi_s = merges[merges.n_absorbed >= 2], splits[splits.n_children >= 2]
gallery(multi_m, merge_panel, f"MIN-0.05 gate · all merges with >2 parents {key}", "cp7_09_merges_multi_parent.png")
gallery(multi_s, split_panel, f"MIN-0.05 gate · all splits with >2 daughters {key}", "cp7_09_splits_multi_daughter.png",
        ncol=max(1, min(2, len(multi_s))))
print(f"merges {len(merges)} ({len(multi_m)} with >2 parents) · splits {len(splits)} "
      f"({len(multi_s)} with >2 daughters)")
print("saved figures/cp7_09_*.png")
