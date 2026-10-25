# cp7_02_scaling.py: are bigger EREs also rainier per cell? (README section 2)
# Raw mean intensity vs size is confounded by the spatially varying p99 threshold, so the test
# uses RELATIVE intensity = footprint-mean rain / footprint-mean threshold (rel_int, built in CP6).
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

ok = pd.read_parquet(DER / "cp7_objdays.parquet")
print(f"rel_int: median {ok.rel_int.median():.2f} · valid {ok.rel_int.notna().sum()}")

# ---------- volume-area scaling ----------
lx = np.log10(ok["size"].to_numpy(float))
ly = np.log10(ok.vol_mm.to_numpy(float))
b, a = np.polyfit(lx, ly, 1)
rng = np.random.default_rng(7)
bs = []
for _ in range(500):
    i = rng.integers(0, len(lx), len(lx))
    bs.append(np.polyfit(lx[i], ly[i], 1)[0])
lo, hi = np.percentile(bs, [2.5, 97.5])
m2 = ok["size"].to_numpy() >= 2
print(f"\nlog10(volume) = {b:.3f} * log10(area) + {a:.2f}   slope CI [{lo:.3f}, {hi:.3f}]"
      f" · excluding 1-cell objects: {np.polyfit(lx[m2], ly[m2], 1)[0]:.3f}")

# ---------- intensity vs size: raw and threshold-relative ----------
edges = [0.5, 1.5, 2.5, 4.5, 9.5, 19.5, 49.5, np.inf]
labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
ok["szbin"] = pd.cut(ok["size"], edges, labels=labels)
rho_raw = spearmanr(ok["size"], ok.mean_mm).statistic
rho_rel = spearmanr(ok["size"], ok.rel_int, nan_policy="omit").statistic
rho_max = spearmanr(ok["size"], ok.max_mm).statistic
print(f"\nSpearman(size, mean_mm) = {rho_raw:+.3f} · (size, rel_int) = {rho_rel:+.3f} "
      f"· (size, max_mm) = {rho_max:+.3f}")

binned = ok.groupby("szbin", observed=True).agg(
    n=("mean_mm", "size"), mean_mm_med=("mean_mm", "median"),
    rel_int_med=("rel_int", "median"), max_mm_med=("max_mm", "median")).reset_index()
print("\nby size bin:")
print(binned.round(2).to_string(index=False))

print("\nSpearman(size, rel_int) within region:")
for r, g in ok.groupby("region"):
    print(f"  {r:20s} n={len(g):5d}  rho={spearmanr(g['size'], g.rel_int, nan_policy='omit').statistic:+.3f}")
yrs = ok.date.str[:4]
rhos = [spearmanr(g["size"], g.rel_int, nan_policy="omit").statistic for _, g in ok.groupby(yrs)]
print(f"per season: rho {min(rhos):+.2f} to {max(rhos):+.2f}, positive in "
      f"{sum(r > 0 for r in rhos)}/{len(rhos)} seasons")

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))
h = ax[0].hexbin(lx, ly, gridsize=40, cmap="viridis", mincnt=1)
xs = np.linspace(lx.min(), lx.max(), 50)
ax[0].plot(xs, b * xs + a, "r-", lw=2, label=f"slope={b:.2f} [{lo:.2f},{hi:.2f}]")
ax[0].plot(xs, xs + (ly.mean() - lx.mean()), "w--", lw=1.5, label="slope=1 (no intensification)")
ax[0].set_xlabel("log10 area (cells)"); ax[0].set_ylabel("log10 rain volume (mm x cells)")
ax[0].set_title(f"Volume-area scaling, all {len(ok):,} object-days")
ax[0].legend(fontsize=9); plt.colorbar(h, ax=ax[0], label="count")

x = np.arange(len(binned))
ax[1].plot(x, binned.mean_mm_med, "o-", color="tab:red", label="raw mean intensity")
ax1b = ax[1].twinx()
ax1b.plot(x, binned.rel_int_med, "s-", color="tab:blue", label="relative intensity\n(/ local threshold)")
ax[1].set_xticks(x); ax[1].set_xticklabels(binned.szbin)
ax[1].set_xlabel("object size (cells)")
ax[1].set_ylabel("median mean rain (mm/day)", color="tab:red")
ax1b.set_ylabel("median relative intensity", color="tab:blue")
ax[1].set_title(f"Intensity vs size: raw rho={rho_raw:+.2f} · relative rho={rho_rel:+.2f}")
ax[1].grid(alpha=0.25)

ax[2].plot(x, binned.max_mm_med, "o-", color="tab:purple")
ax[2].set_xticks(x); ax[2].set_xticklabels(binned.szbin)
ax[2].set_xlabel("object size (cells)"); ax[2].set_ylabel("median peak cell (mm/day)")
ax[2].set_title(f"Peak-cell intensity vs size (rho={rho_max:+.2f})")
ax[2].grid(alpha=0.25)
plt.suptitle("Is bigger also more intense?", y=1.03, fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp7_02_scaling.png", dpi=150, bbox_inches="tight")
print("\nsaved figures/cp7_02_scaling.png")
