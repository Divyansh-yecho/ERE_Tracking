# cp7_04_regions.py: regional structure of ERE tracks (README section 4).
# Regions are assigned by birth centroid: west_ghats, central_india, himalayan_foothills,
# northeast, other. Tests the size-duration link inside each region, track speed by region,
# seasonal cycles, and interannual co-variation and trends (Mann-Kendall + Theil-Sen, n = 25).
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal, spearmanr, theilslopes, norm

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

tracks = pd.read_parquet(DER / "cp7_tracks.parquet")
REGIONS = ["west_ghats", "central_india", "himalayan_foothills", "northeast", "other"]
NICE = {"west_ghats": "W Ghats", "central_india": "Central India",
        "himalayan_foothills": "Him. foothills", "northeast": "Northeast", "other": "Other"}
COLS = ["#e7298a", "#d95f02", "#1b9e77", "#7570b3", "#999999"]

# ---------- per-region track climatology ----------
rows = []
for r in REGIONS:
    g = tracks[tracks.birth_region == r]
    rows.append(dict(region=r, n_tracks=len(g), n_2day_plus=int((g.duration >= 2).sum()),
                     n_3day_plus=int((g.duration >= 3).sum()),
                     pct_multiday=100 * (g.duration >= 2).mean(),
                     pct_3day_plus=100 * (g.duration >= 3).mean(),
                     med_peak_size_2d=g[g.duration >= 2].peak_size.median(),
                     med_speed_3d=g[g.duration >= 3].mean_speed_kmpd.median(),
                     pct_near_lps=100 * (g.lps_class == "near").mean(),
                     pct_near_lps_3d=100 * (g[g.duration >= 3].lps_class == "near").mean()))
reg = pd.DataFrame(rows)
print("Per-region track climatology:")
print(reg.round(1).to_string(index=False))

speeds = [tracks[(tracks.birth_region == r) & (tracks.duration >= 3)].mean_speed_kmpd.dropna().to_numpy()
          for r in REGIONS]
H, p_speed = kruskal(*speeds)
print(f"\nKruskal-Wallis 3+day track speed across regions: H={H:.1f} p={p_speed:.1e}")

print("\nSpearman(duration, peak_size) within region:")
for r in REGIONS:
    g = tracks[tracks.birth_region == r]
    print(f"  {NICE[r]:15s} n={len(g):5d} rho={spearmanr(g.duration, g.peak_size).statistic:+.3f}")

# ---------- seasonal and interannual ----------
mcyc = pd.crosstab(tracks.birth_month, tracks.birth_region)
mcyc_pct = mcyc.div(mcyc.sum(axis=0), axis=1).mul(100)
print("\npeak birth month by region:", {NICE[r]: int(mcyc_pct[r].idxmax()) for r in REGIONS})
yr = pd.crosstab(tracks.year, tracks.birth_region)[REGIONS]
yr["all_india"] = yr.sum(axis=1)

rho = yr[REGIONS].corr(method="spearman")
print("\ninterannual Spearman correlation of track counts:")
print(rho.round(2).rename(index=NICE, columns=NICE).to_string())
for r in REGIONS:
    if r == "northeast": continue
    print(f"  Northeast vs {NICE[r]:14s} p={spearmanr(yr.northeast, yr[r]).pvalue:.2f}")
others = [r for r in REGIONS if r != "northeast"]
pairs = [rho.loc[a, b] for i, a in enumerate(others) for b in others[i + 1:]]
print(f"  mean pairwise rho among the non-Northeast regions = {np.mean(pairs):+.2f}")

def mann_kendall(x):
    """Two-sided Mann-Kendall test (tie-corrected variance, continuity correction)."""
    x = np.asarray(x, float); n = len(x)
    s = sum(np.sign(x[j] - x[i]) for i in range(n) for j in range(i + 1, n))
    _, t = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(t * (t - 1) * (2 * t + 5))) / 18
    zs = (s - np.sign(s)) / np.sqrt(var)
    return 2 * (1 - norm.cdf(abs(zs)))


print("\ninterannual trend (Mann-Kendall p; Theil-Sen slope in tracks/season):")
for col in REGIONS + ["all_india"]:
    print(f"  {NICE.get(col, 'All India'):15s} p={mann_kendall(yr[col]):.2f} "
          f"slope={theilslopes(yr[col], yr.index)[0]:+.2f}")

# ---------- figure ----------
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
x = np.arange(len(REGIONS))
ax[0, 0].bar(x, reg.pct_multiday, color="tab:blue", width=0.6, label="2+ days")
ax[0, 0].bar(x, reg.pct_3day_plus, color="tab:red", width=0.6, label="3+ days")
ax[0, 0].set_xticks(x); ax[0, 0].set_xticklabels([NICE[r] for r in REGIONS], rotation=15)
ax[0, 0].set_ylabel("% of tracks"); ax[0, 0].set_title("Share of long-lived tracks by region")
ax[0, 0].legend(); ax[0, 0].grid(alpha=0.25, axis="y")

ax[0, 1].boxplot(speeds, showfliers=False, tick_labels=[NICE[r] for r in REGIONS])
ax[0, 1].set_ylabel("mean track speed (km/day)")
ax[0, 1].set_title(f"3+day track speed by region (KW p={p_speed:.1e})")
ax[0, 1].tick_params(axis="x", rotation=15); ax[0, 1].grid(alpha=0.25, axis="y")

for r, c in zip(REGIONS, COLS):
    ax[1, 0].plot([6, 7, 8, 9], mcyc_pct[r].reindex([6, 7, 8, 9]), "o-", color=c, label=NICE[r])
ax[1, 0].set_xticks([6, 7, 8, 9]); ax[1, 0].set_xticklabels(["Jun", "Jul", "Aug", "Sep"])
ax[1, 0].set_ylabel("% of region's births"); ax[1, 0].set_title("Seasonal cycle of ERE births by region")
ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.25)

for r, c in zip(REGIONS, COLS):
    ax[1, 1].plot(yr.index, yr[r], "-", color=c, alpha=0.8, label=NICE[r])
ax[1, 1].set_ylabel("tracks born / season"); ax[1, 1].set_title("Interannual counts by region")
ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.25)
plt.suptitle("Regional structure of ERE tracks (JJAS 2000-2024)", y=1.0, fontsize=14)
plt.tight_layout()
plt.savefig(FIG / "cp7_04_regions.png", dpi=150, bbox_inches="tight")
print("\nsaved figures/cp7_04_regions.png")
