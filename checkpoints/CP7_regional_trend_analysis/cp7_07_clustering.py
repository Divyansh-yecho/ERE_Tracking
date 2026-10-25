# cp7_07_clustering.py: do EREs arrive in multi-day episodes? (README section 7)
# Dispersion (variance/mean) of daily birth counts within each JJAS season, against a null that
# keeps each season's total and the 25-season mean seasonal cycle (multinomial redistribution,
# 2,000 simulated seasons; the all-births test is repeated with a second seed). Plus the
# autocorrelation of daily ERE-count anomalies and their correlation with the CMZ rainfall index.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

tracks = pd.read_parquet(DER / "cp7_tracks.parquet")
stats = pd.read_parquet(DER / "cp7_objdays.parquet")
season = pd.read_csv(DER / "cmz_daily_index.csv", dtype={"date": str})
season["dt"] = pd.to_datetime(season.date, format="%Y%m%d")
season["year"] = season.dt.dt.year
season["jjas_day"] = (season.dt - pd.to_datetime(season.year.astype(str) + "-06-01")).dt.days

YEARS, NDAY = np.arange(2000, 2025), 122


def dispersion_test(bdates, label, rng):
    b = pd.to_datetime(pd.Series(bdates).reset_index(drop=True), format="%Y%m%d")
    yr = b.dt.year.to_numpy()
    jd = (b - pd.to_datetime(pd.Series(yr).astype(str) + "-06-01")).dt.days.to_numpy()
    counts = np.zeros((len(YEARS), NDAY))
    np.add.at(counts, (yr - 2000, jd), 1)
    w = counts.mean(0) / counts.mean(0).sum()            # mean seasonal cycle
    obs = np.mean([c.var() / c.mean() for c in counts if c.mean() > 0])
    null = np.array([np.mean([s.var() / s.mean() for s in
                              (rng.multinomial(int(c.sum()), w) for c in counts if c.sum() > 0)])
                     for _ in range(2000)])
    p = max((null >= obs).mean(), 1 / 2000)
    print(f"  {label:26s} variance/mean obs {obs:.2f} vs null {null.mean():.2f}+-{null.std():.2f} · p={p:.4f}")
    return obs, null


print("Within-season clustering of daily births (null keeps the seasonal cycle + season totals):")
rng = np.random.default_rng(11)
obs_all, null_all = dispersion_test(tracks.birth_date, "all births", rng)
obs_md, null_md = dispersion_test(tracks[tracks.duration >= 2].birth_date, "multi-day births", rng)
dispersion_test(tracks[tracks.peak_size >= 20].birth_date, "large (20+ cell) births", rng)
dispersion_test(tracks.birth_date, "all births, second seed", np.random.default_rng(99))


def mean_acf(col, maxlag=10):
    out = []
    for lag in range(maxlag + 1):
        v = []
        for _, g in season.groupby("year"):
            a = g[col].to_numpy()
            v.append(1.0 if lag == 0 else np.corrcoef(a[:-lag], a[lag:])[0, 1])
        out.append(np.mean(v))
    return np.array(out)


# deseasonalised daily counts: object-days (partly mechanical, as a multi-day track fills
# consecutive days) and births (one event per track, the better clustering measure)
for name, s in [("objdays", stats.groupby("date").size()), ("births", tracks.groupby("birth_date").size())]:
    season[name] = s.reindex(season.date, fill_value=0).to_numpy(float)
    season[f"{name}_anom"] = season[name] - season.jjas_day.map(season.groupby("jjas_day")[name].mean())
ac = mean_acf("objdays_anom")
r_cmz = np.corrcoef(season.objdays_anom, season.cmz_z)[0, 1]
print(f"\nlag-1 autocorrelation of daily anomalies: object-days {ac[1]:.2f} · "
      f"births {mean_acf('births_anom', 1)[1]:.2f}")
print(f"corr(daily ERE-count anomaly, CMZ z-index) = {r_cmz:.2f}")

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for a, null, obs, ttl, xl in [
        (ax[0], null_all, obs_all, "All track births: clustered?", "mean variance/mean of daily births"),
        (ax[1], null_md, obs_md, "Multi-day track births: clustered?",
         "mean variance/mean of daily multi-day births")]:
    a.hist(null, bins=40, color="0.7", label="null (cycle-preserving)")
    a.axvline(obs, color="tab:red", lw=2, label=f"observed {obs:.2f}")
    a.set_xlabel(xl); a.set_title(ttl); a.legend(fontsize=9); a.grid(alpha=0.25)
ax[2].bar(range(11), ac, color="slateblue", width=0.7)
ax[2].axhline(0, color="k", lw=0.8)
ax[2].set_xlabel("lag (days)"); ax[2].set_ylabel("autocorrelation")
ax[2].set_title(f"Persistence of ERE activity anomalies\n(corr with CMZ index r={r_cmz:.2f})")
ax[2].grid(alpha=0.25)
plt.suptitle("Sub-seasonal clustering of extreme-rain events", y=1.03, fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp7_07_clustering.png", dpi=150, bbox_inches="tight")
print("saved figures/cp7_07_clustering.png")
