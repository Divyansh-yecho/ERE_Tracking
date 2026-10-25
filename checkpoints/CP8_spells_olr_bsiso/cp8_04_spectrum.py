# cp8_04_spectrum.py: TEST C: is the CMZ rainfall z-index periodic on the 30-60 day BSISO
# timescale, or just red noise? Per-season (122-day) Hann-tapered periodogram averaged over the
# 25 seasons, against an AR(1) null with the pooled within-season lag-1 autocorrelation
# (10,000 Monte-Carlo sets of 25 seasons) -> 30-61 d band-power-fraction test.
# Positive control: the same test on BSISO1 PC1, a known 30-60 day oscillation. If PC1 passes
# and the rain index fails, the test has power and the rain-index result is a real negative.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

d = pd.read_csv(DER / "spell_drivers_daily.csv", dtype={"date": str})
d["year"] = d.date.str[:4].astype(int)
d["bsiso1_pc1"] = d.bsiso1_pc1.ffill()                   # single missing day (2015-06-21)
years = sorted(d.year.unique()); NS = len(years)


def season_matrix(col):
    M = np.stack([d.loc[d.year == y, col].to_numpy()[:122] for y in years])
    return M - M.mean(axis=1, keepdims=True)


def mean_periodogram(M):
    w = np.hanning(M.shape[1])
    return (np.abs(np.fft.rfft(M * w, axis=1)) ** 2 / (w ** 2).sum()).mean(axis=0)


def lag1_pooled(M):
    return float(np.mean([np.corrcoef(m[:-1], m[1:])[0, 1] for m in M]))


def ar1_seasons(r1, sd):
    x = np.zeros((NS, 152))                               # 30-day spin-up discarded
    x[:, 0] = rng.normal(0, sd, NS)
    eps = rng.normal(0, sd * np.sqrt(1 - r1 ** 2), (NS, 152))
    for t in range(1, 152):
        x[:, t] = r1 * x[:, t - 1] + eps[:, t]
    M = x[:, 30:]
    return M - M.mean(axis=1, keepdims=True)


freqs = np.fft.rfftfreq(122, d=1.0)
periods = np.where(freqs > 0, 1 / np.where(freqs > 0, freqs, np.nan), np.inf)
band = (periods >= 30) & (periods <= 61.5)                # bins at 61.0, 40.7, 30.5 d
rng = np.random.default_rng(0)
results = {}
for col, label in [("cmz_z", "CMZ rain z (IMERG)"), ("olr_z", "CMZ OLR z (CBO)"),
                   ("bsiso1_pc1", "BSISO1 PC1 — positive control")]:
    M = season_matrix(col)
    P, r1 = mean_periodogram(M), lag1_pooled(M)
    Pn = np.array([mean_periodogram(ar1_seasons(r1, M.std())) for _ in range(10000)])
    Pn *= P.sum() / Pn.sum(axis=1, keepdims=True)         # match total power
    band_obs = P[band].sum() / P[1:].sum()
    pval = float(np.mean(Pn[:, band].sum(axis=1) / Pn[:, 1:].sum(axis=1) >= band_obs))
    results[col] = dict(P=P, Pn=Pn, r1=r1, pval=pval, label=label)
    print(f"{label}: r1={r1:.2f}; spectral peak {periods[1:][np.argmax(P[1:])]:.0f} d; "
          f"30-61 d band fraction {band_obs:.3f} -> p = {pval:.3f}")

# within-season autocorrelation of the rain index vs AR(1)
M = season_matrix("cmz_z"); r1 = results["cmz_z"]["r1"]


def mean_acf(M, maxlag=60):
    return np.array([np.mean([np.corrcoef(m[:-lag], m[lag:])[0, 1] if lag else 1.0 for m in M])
                     for lag in range(maxlag + 1)])


acf_obs = mean_acf(M)
acfs_null = np.array([mean_acf(ar1_seasons(r1, M.std())) for _ in range(2000)])

# ---------- figure ----------
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
for col, a in [("cmz_z", ax[0, 0]), ("olr_z", ax[0, 1]), ("bsiso1_pc1", ax[1, 0])]:
    r = results[col]; pm = periods[1:]
    a.semilogx(pm, r["P"][1:], "k-", lw=1.6, label="observed mean spectrum")
    a.semilogx(pm, np.median(r["Pn"], axis=0)[1:], "r--", lw=1.2, label="AR(1) median")
    a.semilogx(pm, np.percentile(r["Pn"], 95, axis=0)[1:], "r:", lw=1, label="AR(1) 95%")
    a.axvspan(30, 61.5, color="tab:blue", alpha=0.12, label="30-61 d band")
    a.set_xlabel("period (days)"); a.set_ylabel("mean power")
    a.set_title(f"{r['label']}\nband test p = {r['pval']:.3f} ({'PASSES' if r['pval'] < 0.05 else 'fails'}) · "
                f"r1 = {r['r1']:.2f}", fontsize=10)
    a.set_xticks([3, 5, 10, 20, 30, 40, 60, 120]); a.set_xticklabels([3, 5, 10, 20, 30, 40, 60, 120])
    a.legend(fontsize=7); a.grid(alpha=0.2, which="both")
a = ax[1, 1]
lags = np.arange(61)
a.plot(lags, acf_obs, "k-", lw=1.6, label="observed mean ACF (rain z)")
a.fill_between(lags, np.percentile(acfs_null, 2.5, axis=0), np.percentile(acfs_null, 97.5, axis=0),
               color="r", alpha=0.15, label="AR(1) 95% envelope")
a.axhline(0, color="0.6", lw=0.7)
a.set_xlabel("lag (days)"); a.set_ylabel("autocorrelation")
a.set_title("CMZ rain z: within-season autocorrelation", fontsize=10)
a.legend(fontsize=8); a.grid(alpha=0.25)
plt.suptitle("TEST C — 30-60 d periodicity vs AR(1) red noise (25 JJAS seasons): "
             "control passes, rain index does NOT", y=1.0, fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp8_04_spectrum.png", dpi=150, bbox_inches="tight")
print("saved figures/cp8_04_spectrum.png")
