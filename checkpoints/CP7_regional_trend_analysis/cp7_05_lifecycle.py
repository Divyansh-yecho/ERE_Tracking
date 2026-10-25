# cp7_05_lifecycle.py: life cycles of the 156 tracks lasting 3+ days, by LPS class (README section 5).
# Absolute magnitudes (figure): near-LPS tracks are ~2x larger but not more intense than far tracks.
# Timing (printed): position of the size peak in normalised lifetime tau (0 = birth, 1 = death),
# growth/decay rates, and whether intensity peaks before size. tau is quantised for short tracks
# ({0, 0.5, 1} for 3-day tracks), so the timing contrast is also broken down by track length.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, wilcoxon

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

stats = pd.read_parquet(DER / "cp7_objdays.parquet")
tracks = pd.read_parquet(DER / "cp7_tracks.parquet")
md = tracks[tracks.duration >= 3]
cls_of = dict(zip(tracks.track_num, tracks.lps_class))
print(f"3+ day tracks: {len(md)} · {md.lps_class.value_counts().to_dict()}")

TAU = np.linspace(0, 1, 101)
curves = {c: {"sz": [], "mx": []} for c in ["near", "far", "no_lps"]}
rows = []
for tn, g in stats[stats.track_num.isin(md.track_num)].groupby("track_num"):
    g = g.sort_values("day_of_track")
    sz, mx = g["size"].to_numpy(float), g.max_mm.to_numpy(float)
    n = len(sz); tau = np.arange(n) / (n - 1)
    curves[cls_of[tn]]["sz"].append(np.interp(TAU, tau, sz))
    curves[cls_of[tn]]["mx"].append(np.interp(TAU, tau, mx))
    i_s, i_m = int(sz.argmax()), int(mx.argmax())
    rows.append(dict(track_num=tn, n_days=n, lps_class=cls_of[tn], tau_peak_size=tau[i_s],
                     tau_peak_size_last=tau[n - 1 - int(sz[::-1].argmax())],
                     lead=tau[i_s] - tau[i_m],
                     growth_cpd=(sz[i_s] - sz[0]) / max(i_s, 1),
                     decay_cpd=(sz[i_s] - sz[-1]) / max(n - 1 - i_s, 1)))
lc = pd.DataFrame(rows)

# ---------- absolute peaks ----------
near, far = md[md.lps_class == "near"], md[md.lps_class == "far"]
tests = {}
print("\nabsolute per-track medians (near vs far, one-sided MWU near > far):")
for col, nice in [("peak_size", "peak size (cells)"), ("mean_size", "mean size over life (cells)"),
                  ("peak_max_mm", "peak single-cell rain (mm/day)"),
                  ("peak_mean_mm", "peak footprint-mean rain (mm/day)"),
                  ("mean_speed_kmpd", "track speed (km/day)")]:
    p = mannwhitneyu(near[col], far[col], alternative="greater").pvalue
    tests[col] = p
    print(f"  {nice:34s} near {near[col].median():6.1f}  far {far[col].median():6.1f}  p={p:.2g}")

# ---------- timing ----------
def contrast(df, col="tau_peak_size", label=""):
    a, b = df[df.lps_class == "near"][col], df[df.lps_class == "far"][col]
    p1 = mannwhitneyu(a, b, alternative="greater").pvalue
    p2 = mannwhitneyu(a, b).pvalue
    print(f"  {label:22s} near {a.mean():.2f} (n={len(a)}) vs far {b.mean():.2f} (n={len(b)})  "
          f"p one-sided {p1:.2g} / two-sided {p2:.2g}")
    return a, b

print("\nposition of the size peak, mean tau (MWU, near later than far):")
a, b = contrast(lc, label="all 3+ day tracks")
rng = np.random.default_rng(3)
diffs = [rng.choice(a.to_numpy(), len(a)).mean() - rng.choice(b.to_numpy(), len(b)).mean()
         for _ in range(5000)]
print(f"  bootstrap 95% CI of the difference: [{np.percentile(diffs, 2.5):+.2f}, {np.percentile(diffs, 97.5):+.2f}]")
contrast(lc, "tau_peak_size_last", "ties -> last argmax")
contrast(lc[lc.n_days == 3], label="3-day tracks")
contrast(lc[lc.n_days == 4], label="4-day tracks")
contrast(lc[lc.n_days >= 4], label="4+ day tracks")
contrast(lc[lc.n_days >= 5], label="5+ day tracks")

gn, gf = lc[lc.lps_class == "near"].growth_cpd, lc[lc.lps_class == "far"].growth_cpd
print(f"\ngrowth rate near {gn.median():.1f} vs far {gf.median():.1f} cells/day "
      f"(MWU two-sided p={mannwhitneyu(gn, gf).pvalue:.2g})")
print(f"pooled growth {lc.growth_cpd.median():.1f} vs decay {lc.decay_cpd.median():.1f} cells/day "
      f"(Wilcoxon p={wilcoxon(lc.growth_cpd, lc.decay_cpd).pvalue:.2f})")
print(f"intensity (max cell) and size peak on the same day in {100 * (lc.lead == 0).mean():.0f}% of tracks; "
      f"intensity first {100 * (lc.lead > 0).mean():.0f}%, size first {100 * (lc.lead < 0).mean():.0f}% "
      f"(Wilcoxon p={wilcoxon(lc.lead[lc.lead != 0]).pvalue:.2f})")

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
cols = {"near": "tab:red", "far": "tab:blue"}
for k, (key, ylabel, title) in enumerate([
        ("sz", "object size (cells) — ABSOLUTE", "Median size through life, real units:\nnear-LPS tracks are ~2x larger"),
        ("mx", "strongest cell in footprint (mm/day) — ABSOLUTE", "Median peak-cell rainfall through life")]):
    for cl in ["near", "far"]:
        S = np.vstack(curves[cl][key])
        ax[k].plot(TAU, np.nanmedian(S, 0), color=cols[cl], lw=2.4, label=f"{cl} (n={len(S)})")
        ax[k].fill_between(TAU, np.nanpercentile(S, 25, 0), np.nanpercentile(S, 75, 0),
                           color=cols[cl], alpha=0.15)
    ax[k].set_xlabel("normalized lifetime tau (0 birth, 1 death)")
    ax[k].set_ylabel(ylabel); ax[k].set_title(title)
    ax[k].legend(fontsize=10); ax[k].grid(alpha=0.25)

b1 = ax[2].boxplot([near.peak_size, far.peak_size], positions=[0, 1], widths=0.55, showfliers=False,
                   patch_artist=True)
ax2b = ax[2].twinx()
b2 = ax2b.boxplot([near.peak_max_mm, far.peak_max_mm], positions=[2.6, 3.6], widths=0.55,
                  showfliers=False, patch_artist=True)
for bx in (b1, b2):
    for patch, c in zip(bx["boxes"], ["tab:red", "tab:blue"]):
        patch.set_facecolor(c); patch.set_alpha(0.45)
ax[2].set_ylabel("peak size (cells)"); ax2b.set_ylabel("peak cell rain (mm/day)")
ax[2].set_xticks([0, 1, 2.6, 3.6]); ax[2].set_xticklabels(["near", "far", "near", "far"])
ax[2].set_title(f"Per-track peaks: size (left, p={tests['peak_size']:.1g})\n"
                f"and strongest cell (right, p={tests['peak_max_mm']:.2g})")
ax[2].grid(alpha=0.25, axis="y")
plt.suptitle("Near-LPS vs far tracks in ABSOLUTE units (all 3+ day tracks)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp7_05_lifecycle.png", dpi=150, bbox_inches="tight")
print("\nsaved figures/cp7_05_lifecycle.png")
