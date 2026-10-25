# cp8_05_amplitude.py: does the BSISO modulation of CMZ rain, spells and ERE activity scale with
# BSISO1 amplitude? Wet phases = 4-6, dry phases = 7, 8, 1 (from the TEST B composites; 2-3 are
# the transition over the equatorial Indian Ocean and are left out).
#   day level   : wet-minus-dry contrast of CMZ rain z and ERE object-days/day per amplitude bin
#                 (bootstrap 95% CIs)
#   season level: JJAS-mean amplitude vs number of Jul-Aug spell days (only n = 25)
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

d = pd.read_csv(DER / "spell_drivers_daily.csv", dtype={"date": str})
d["year"] = d.date.str[:4].astype(int)
obj = pd.read_parquet(DER / "cp7_objdays.parquet")
d = d.merge(obj.groupby("date").size().rename("n_obj"), left_on="date", right_index=True,
            how="left").fillna({"n_obj": 0})

ja = d[d.phase_ja != "excluded"].dropna(subset=["bsiso1_phase"]).copy()
ja["sector"] = np.where(ja.bsiso1_phase.isin([4, 5, 6]), "wet",
                        np.where(ja.bsiso1_phase.isin([7, 8, 1]), "dry", "trans"))
labels = ["weak (<0.8)", "moderate (0.8-1.5)", "strong (>=1.5)"]
ja["ampbin"] = pd.cut(ja.bsiso1_amp, [0, 0.8, 1.5, 10], labels=labels)

rng = np.random.default_rng(0)


def boot_contrast(gw, gd, col, nboot=5000):
    bs = [rng.choice(gw[col], len(gw)).mean() - rng.choice(gd[col], len(gd)).mean() for _ in range(nboot)]
    return gw[col].mean() - gd[col].mean(), *np.percentile(bs, [2.5, 97.5])


rows = []
for ab in labels:
    g = ja[ja.ampbin == ab]
    gw, gd = g[g.sector == "wet"], g[g.sector == "dry"]
    dz, dz_lo, dz_hi = boot_contrast(gw, gd, "cmz_z")
    do, do_lo, do_hi = boot_contrast(gw, gd, "n_obj")
    rows.append(dict(ampbin=ab, n_wet=len(gw), n_dry=len(gd), dz=dz, dz_lo=dz_lo, dz_hi=dz_hi,
                     dobj=do, dobj_lo=do_lo, dobj_hi=do_hi))
amp = pd.DataFrame(rows)
print("wet (4-6) minus dry (7, 8, 1) contrast by BSISO1 amplitude (Jul-Aug):")
print(amp.round(2).to_string(index=False))

seas = d.groupby("year").apply(lambda g: pd.Series(dict(
    amp_jjas=g.bsiso1_amp.mean(),
    spell_days=g.phase_ja.isin(["active", "break"]).sum(),
    sd_z=g.cmz_z.std())), include_groups=False)
r1, p1 = pearsonr(seas.amp_jjas, seas.spell_days)
rs, ps = spearmanr(seas.amp_jjas, seas.spell_days)
r2, p2 = pearsonr(seas.amp_jjas, seas.sd_z)
print(f"\nseason level (n=25): amplitude vs Jul-Aug spell days r={r1:.2f} (p={p1:.2f}), "
      f"Spearman {rs:.2f} (p={ps:.2f}); amplitude vs sd(cmz_z) r={r2:.2f} (p={p2:.2f})")

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
x = np.arange(3)
for a, col, c, ylab, ttl in [(ax[0], "dz", "tab:green", "CMZ rain z: wet-phase minus dry-phase",
                              "Rain contrast grows with BSISO amplitude"),
                             (ax[1], "dobj", "tab:blue", "ERE object-days/day: wet minus dry",
                              "ERE-activity contrast vs amplitude")]:
    a.bar(x, amp[col], 0.5, color=c, yerr=[amp[col] - amp[col + "_lo"], amp[col + "_hi"] - amp[col]], capsize=4)
    a.set_xticks(x); a.set_xticklabels(amp.ampbin, fontsize=8)
    a.axhline(0, color="0.5", lw=0.8); a.set_ylabel(ylab); a.set_title(ttl); a.grid(alpha=0.25, axis="y")
a = ax[2]
a.scatter(seas.amp_jjas, seas.spell_days, s=38, c="tab:purple")
for y_, r_ in seas.iterrows():
    a.annotate(str(y_)[2:], (r_.amp_jjas, r_.spell_days), fontsize=6, xytext=(3, 3), textcoords="offset points")
m, b = np.polyfit(seas.amp_jjas, seas.spell_days, 1)
xs = np.linspace(seas.amp_jjas.min(), seas.amp_jjas.max(), 10)
a.plot(xs, m * xs + b, "k--", lw=1)
a.set_xlabel("JJAS-mean BSISO1 amplitude"); a.set_ylabel("Jul-Aug spell days (active+break)")
a.set_title(f"Season level: r = {r1:.2f} (p = {p1:.3f}), n = 25"); a.grid(alpha=0.25)
plt.suptitle("BSISO amplitude dependence of the active/break modulation (Jul-Aug 2000-2024)", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp8_05_amplitude.png", dpi=150, bbox_inches="tight")
print("saved figures/cp8_05_amplitude.png")
