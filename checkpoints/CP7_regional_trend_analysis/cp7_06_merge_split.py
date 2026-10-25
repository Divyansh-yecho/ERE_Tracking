# cp7_06_merge_split.py: rainfall budget of merges and splits under the MIN-0.05 gate
# (README section 6). The earlier MEAN rule, (fwd+bwd)/2 >= 0.10, accepted
# one-sided edge contacts (a sliver brushing a big object) as merges; requiring min(fwd,bwd)
# >= 0.05 keeps only two-sided interactions. Continuations are identical under both gates.
# Merge: product volume vs the summed parent volumes the day before.
# Split: summed daughter volumes vs the parent volume the day before.
# Null : every ordinary continuation step (volume ratio day t+1 / day t).
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

print("events per gate: " + " · ".join(
    f"{g}: {len(pd.read_parquet(DER / f'merges_{g}.parquet'))} merges / "
    f"{len(pd.read_parquet(DER / f'splits_{g}.parquet'))} splits" for g in ["mean", "min05"]))
stats = pd.read_parquet(DER / "objdays_min05.parquet")
merge = pd.read_parquet(DER / "merges_min05.parquet")
split = pd.read_parquet(DER / "splits_min05.parquet")
by_trk = {tn: g.sort_values("day_of_track").reset_index(drop=True) for tn, g in stats.groupby("track_num")}


def node(tn, dot):
    r = by_trk[int(tn)].set_index("day_of_track").loc[dot]
    return dict(size=float(r["size"]), vol=float(r.vol_mm), inten=float(r.mean_mm))


mrows = []
for _, r in merge.iterrows():
    absorbed = [int(x) for x in str(r.absorbed_tracks).split(",") if x.strip()]
    g = by_trk[int(r.dominant_track)]
    k = int(g[(g.date == r.date) & g.merge_event].day_of_track.iloc[0])
    parents = [node(r.dominant_track, k - 1)] + [node(a, by_trk[a].day_of_track.max()) for a in absorbed]
    prod = node(r.dominant_track, k)
    svol, ssz = sum(p["vol"] for p in parents), sum(p["size"] for p in parents)
    mrows.append(dict(sum_parent_vol=svol, product_vol=prod["vol"], vol_ratio=prod["vol"] / svol,
                      int_ratio=prod["inten"] / (svol / ssz)))
M = pd.DataFrame(mrows)

srows = []
for _, r in split.iterrows():
    children = [int(x) for x in str(r.child_tracks).split(",") if x.strip()]
    g = by_trk[int(r.parent_track)]
    k = int(g[(g.date == r.date) & g.split_event].day_of_track.iloc[0])
    parent = node(r.parent_track, k - 1)
    svol = node(r.parent_track, k)["vol"] + sum(node(c, 0)["vol"] for c in children)
    srows.append(dict(parent_vol=parent["vol"], sum_daughter_vol=svol, vol_ratio=svol / parent["vol"]))
S = pd.DataFrame(srows)

# ordinary continuation steps
cont = stats[(stats.link_type == "overlap") & ~stats.merge_event & ~stats.split_event]
ratios = []
for _, r in cont.iterrows():
    g = by_trk[int(r.track_num)]
    pr = g[g.day_of_track == r.day_of_track - 1]
    if not pr.empty and float(pr.iloc[0].vol_mm) > 0:
        ratios.append(float(r.vol_mm) / float(pr.iloc[0].vol_mm))
ratios = np.array(ratios)
pm = mannwhitneyu(M.vol_ratio, ratios, alternative="greater").pvalue
ps = mannwhitneyu(S.vol_ratio, ratios, alternative="less").pvalue
print(f"continuation null: n={len(ratios)}, median volume ratio {np.median(ratios):.2f}")
print(f"MERGES n={len(M)}: median volume ratio {M.vol_ratio.median():.2f} (p={pm:.2g}) · "
      f"product more intense than parents in {100 * (M.int_ratio > 1).mean():.0f}%")
print(f"SPLITS n={len(S)}: median volume ratio {S.vol_ratio.median():.2f} (p={ps:.2g}) · "
      f"volume falls in {100 * (S.vol_ratio < 1).mean():.0f}%")

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
bins = np.arange(0, 5.1, 0.3)
ax[0].hist(np.clip(ratios, 0, 5), bins=bins, density=True, alpha=0.45, color="0.6",
           label=f"ordinary steps (n={len(ratios)})")
ax[0].hist(np.clip(M.vol_ratio, 0, 5), bins=bins, density=True, alpha=0.6, color="tab:red",
           label=f"merge steps (n={len(M)})")
ax[0].axvline(1, color="k", lw=1)
ax[0].set_xlabel("volume ratio (after / before)"); ax[0].set_ylabel("density")
ax[0].set_title(f"Merges grow more than ordinary steps\nmedian {M.vol_ratio.median():.2f} vs "
                f"{np.median(ratios):.2f}  (p={pm:.2g})")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

for a, D, xc, yc, c, ttl in [
        (ax[1], M, "sum_parent_vol", "product_vol", "tab:red", f"Merge budget (n={len(M)})\nabove line = growth"),
        (ax[2], S, "parent_vol", "sum_daughter_vol", "tab:green",
         f"Split budget (n={len(S)})\nbelow line = decay  (p={ps:.2g})")]:
    a.scatter(D[xc], D[yc], s=45, c=c, alpha=0.75, edgecolor="white")
    lim = [D[[xc, yc]].min().min() * 0.7, D[[xc, yc]].max().max() * 1.4]
    a.plot(lim, lim, "k--", lw=1, label="conservation")
    a.set_xscale("log"); a.set_yscale("log"); a.set_title(ttl)
    a.legend(fontsize=9); a.grid(alpha=0.25)
ax[1].set_xlabel("sum of parent volumes"); ax[1].set_ylabel("product volume")
ax[2].set_xlabel("parent volume"); ax[2].set_ylabel("sum of daughter volumes")
plt.suptitle("Merge / split rainfall budget under the MIN-0.05 gate", y=1.03, fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp7_06_merge_split.png", dpi=150, bbox_inches="tight")
print("saved figures/cp7_06_merge_split.png")
