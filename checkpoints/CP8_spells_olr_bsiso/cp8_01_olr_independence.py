# cp8_01_olr_independence.py: TEST A: are the rainfall-defined active/break spells visible in
# independent satellite OLR?
#   1. corr(cmz_z, olr_z): expect strongly negative (wet = low OLR)
#   2. OLR z on rainfall-defined active / neutral / break days (Mann-Whitney)
#   3. spells rebuilt from OLR alone (active = olr_z <= -1 for >= 3 d, break = olr_z >= +1 for
#      >= 3 d, Jul-Aug only, mirroring the rainfall rule): agreement and Cohen's kappa
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, mannwhitneyu

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"
CATS = ["active", "neutral", "break"]

d = pd.read_csv(DER / "spell_drivers_daily.csv", dtype={"date": str})
d["dt"] = pd.to_datetime(d.date, format="%Y%m%d")
d["year"], d["month"] = d.dt.dt.year, d.dt.dt.month

# ---------- 1. correlation ----------
ja = d[d.phase_ja != "excluded"].copy()
r_all, r_ja = pearsonr(d.cmz_z, d.olr_z)[0], pearsonr(ja.cmz_z, ja.olr_z)[0]
per_season = d.groupby("year").apply(lambda g: pearsonr(g.cmz_z, g.olr_z)[0], include_groups=False)
print(f"corr(cmz_z, olr_z): JJAS r={r_all:.3f} | Jul-Aug r={r_ja:.3f} | per season "
      f"{per_season.min():.2f} to {per_season.max():.2f} (negative in {(per_season < 0).sum()}/25)")

# ---------- 2. OLR on rainfall-defined spell days ----------
groups = {ph: ja[ja.phase_ja == ph].olr_z for ph in CATS}
for ph, g in groups.items():
    print(f"  {ph:8s} n={len(g):4d}  median OLR z = {g.median():+.2f}")
U, p_ab = mannwhitneyu(groups["active"], groups["break"])
print(f"  active vs break: MWU p={p_ab:.1e}, Cliff delta="
      f"{2 * U / (len(groups['active']) * len(groups['break'])) - 1:+.2f}")

# ---------- 3. spells rebuilt from OLR alone ----------
z, yr, in_ja = d.olr_z.to_numpy(), d.year.to_numpy(), d.month.isin([7, 8]).to_numpy()
olr_phase = np.where(in_ja, "neutral", "excluded").astype(object)
for cond, name in [(z <= -1.0, "active"), (z >= 1.0, "break")]:       # low OLR = wet
    c = cond & in_ja
    i = 0
    while i < len(c):
        if c[i]:
            j = i
            while j + 1 < len(c) and c[j + 1] and yr[j + 1] == yr[i]:
                j += 1
            if j - i + 1 >= 3:
                olr_phase[i:j + 1] = name
            i = j + 1
        else:
            i += 1
d["olr_phase"] = olr_phase
ja = d[d.phase_ja != "excluded"]
ct = pd.crosstab(ja.phase_ja, ja.olr_phase).reindex(index=CATS, columns=CATS, fill_value=0)
po = (ja.phase_ja == ja.olr_phase).mean()
pe = sum((ja.phase_ja == c).mean() * (ja.olr_phase == c).mean() for c in CATS)
kappa = (po - pe) / (1 - pe)
print("\ncontingency (rows = rainfall spell, cols = OLR spell), Jul-Aug days:")
print(ct.to_string())
print(f"agreement {po * 100:.1f}% (chance {pe * 100:.1f}%) · Cohen kappa = {kappa:.2f}")
for ph, opp in [("active", "break"), ("break", "active")]:
    g = ja[ja.phase_ja == ph]
    print(f"  rainfall {ph:6s} days: {100 * (g.olr_phase == ph).mean():.0f}% {ph} in OLR, "
          f"{100 * (g.olr_phase == opp).mean():.1f}% opposite")

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
cols = {"active": "tab:red", "break": "tab:blue", "neutral": "0.7"}
for ph in ["neutral", "active", "break"]:
    g = ja[ja.phase_ja == ph]
    ax[0].scatter(g.cmz_z, g.olr_z, s=8, c=cols[ph], alpha=0.55, label=f"{ph} (rain)")
ax[0].axhline(0, color="0.8", lw=0.6); ax[0].axvline(0, color="0.8", lw=0.6)
ax[0].set_xlabel("CMZ rainfall z (IMERG)"); ax[0].set_ylabel("CMZ OLR z (CPC blended OLR)")
ax[0].set_title(f"Independent convection check\nJul-Aug r = {r_ja:.2f} (JJAS r = {r_all:.2f})")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.2)

vp = ax[1].violinplot([groups[c] for c in CATS], [0, 1, 2], showmedians=True)
for b, c in zip(vp["bodies"], ["tab:red", "0.6", "tab:blue"]):
    b.set_facecolor(c); b.set_alpha(0.6)
ax[1].axhline(0, color="0.8", lw=0.6)
ax[1].set_xticks([0, 1, 2]); ax[1].set_xticklabels(CATS)
ax[1].set_ylabel("OLR z on that day")
ax[1].set_title(f"OLR on rainfall-defined spell days\nactive vs break: MWU p = {p_ab:.1e}")
ax[1].grid(alpha=0.2, axis="y"); ax[1].invert_yaxis()             # low OLR (more convection) up

ax[2].imshow(ct.to_numpy(), cmap="Blues")
for i in range(3):
    for j in range(3):
        v = ct.iloc[i, j]
        ax[2].text(j, i, str(v), ha="center", va="center", color="white" if v > ct.to_numpy().max() * 0.6 else "black")
ax[2].set_xticks(range(3), [f"OLR {c}" for c in CATS]); ax[2].set_yticks(range(3), [f"rain {c}" for c in CATS])
n_opp = ct.loc["active", "break"] + ct.loc["break", "active"]
ax[2].set_title(f"Spells rebuilt from OLR alone\nagreement {po * 100:.0f}% · kappa = {kappa:.2f} · "
                f"{n_opp} day(s) opposite")
plt.suptitle("TEST A — rainfall-defined active/break spells are visible in independent satellite OLR",
             y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp8_01_olr_independence.png", dpi=150, bbox_inches="tight")
print("saved figures/cp8_01_olr_independence.png")
