# cp8_02_bsiso_composites.py: TEST B: are the active/break spells (and the ERE geography)
# phase-locked to the BSISO? Composites by BSISO1 phase on organised days (amplitude >= 1),
# July-August: CMZ rain z and OLR z, the fraction of days inside rainfall spells, ERE
# object-days/day and the Central-India vs Northeast share.
# Significance is judged at the SPELL level (each spell -> its modal BSISO1 phase, compared with
# phases drawn from the pool of organised days). The day-level chi-square is printed for reference
# only: spells last >= 3 days and the BSISO is autocorrelated, so it is far too optimistic.
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

REPO = Path(__file__).resolve().parents[2]
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

d = pd.read_csv(DER / "spell_drivers_daily.csv", dtype={"date": str})
d["dt"] = pd.to_datetime(d.date, format="%Y%m%d")
obj = pd.read_parquet(DER / "cp7_objdays.parquet")
cnt = pd.DataFrame({"n_obj": obj.groupby("date").size(),
                    "n_ci": obj[obj.region == "central_india"].groupby("date").size(),
                    "n_ne": obj[obj.region == "northeast"].groupby("date").size()})
d = d.merge(cnt, left_on="date", right_index=True, how="left").fillna({"n_obj": 0, "n_ci": 0, "n_ne": 0})

ja = d[d.phase_ja != "excluded"].copy()
org = ja[ja.bsiso1_amp >= 1].copy()
print(f"Jul-Aug days: {len(ja)} | BSISO1 amp >= 1: {len(org)} ({len(org) / len(ja) * 100:.0f}%)")

# ---------- composites by phase ----------
rows = []
for ph in range(1, 9):
    g = org[org.bsiso1_phase == ph]
    rows.append(dict(phase=ph, n_days=len(g), cmz_z=g.cmz_z.mean(), cmz_z_sem=g.cmz_z.sem(),
                     olr_z=g.olr_z.mean(), olr_z_sem=g.olr_z.sem(),
                     frac_active=(g.phase_ja == "active").mean(), frac_break=(g.phase_ja == "break").mean(),
                     objdays_per_day=g.n_obj.mean(),
                     ci_share=g.n_ci.sum() / max(g.n_obj.sum(), 1), ne_share=g.n_ne.sum() / max(g.n_obj.sum(), 1)))
comp = pd.DataFrame(rows)
print("\ncomposites by BSISO1 phase (Jul-Aug, amp >= 1):")
print(comp.round(3).to_string(index=False))
chi2, p_chi, dof, _ = chi2_contingency(pd.crosstab(org.bsiso1_phase, org.phase_ja))
print(f"\nday-level chi2 (phase x spell) = {chi2:.1f}, p = {p_chi:.0e}  <- NOT valid: days are not independent")

# ---------- spell-level phase assignment ----------
ja = ja.sort_values("dt").reset_index(drop=True)
ja["spell_id"] = ((ja.phase_ja != ja.phase_ja.shift()) | (ja.dt.diff() > pd.Timedelta(days=1))).cumsum()
spells = []
for _, g in ja[ja.phase_ja.isin(["active", "break"])].groupby("spell_id"):
    go = g[g.bsiso1_amp >= 1]
    spells.append(dict(kind=g.phase_ja.iloc[0], phase=(go.bsiso1_phase.mode().iloc[0] if len(go) else np.nan)))
sp = pd.DataFrame(spells)
print(f"\nspells: {(sp.kind == 'active').sum()} active, {(sp.kind == 'break').sum()} break "
      f"({sp.phase.isna().sum()} dropped: all days amp < 1)")
print(pd.crosstab(sp.dropna().phase.astype(int), sp.dropna().kind).reindex(range(1, 9), fill_value=0).T.to_string())


def resultant(phases):
    v = np.exp(1j * np.radians((np.asarray(phases, float) - 1) * 45 + 22.5)).mean()
    return np.abs(v), (np.degrees(np.angle(v)) % 360) / 45 + 0.5


pool = org.bsiso1_phase.to_numpy()
pool_frac = np.array([(pool == ph).mean() for ph in range(1, 9)])
spell_p = {}
for kind in ["active", "break"]:
    phs = sp[sp.kind == kind].dropna().phase.to_numpy(); n = len(phs)
    R, mph = resultant(phs)
    exp = pool_frac * n
    chi_obs = (((np.array([(phs == ph).sum() for ph in range(1, 9)]) - exp) ** 2) / exp).sum()
    rng = np.random.default_rng(0)
    Rn, chin = [], []
    for _ in range(10000):
        s = rng.choice(pool, n)
        Rn.append(resultant(s)[0])
        chin.append((((np.array([(s == ph).sum() for ph in range(1, 9)]) - exp) ** 2) / exp).sum())
    spell_p[kind] = (np.mean(np.array(Rn) >= R), np.mean(np.array(chin) >= chi_obs))
    print(f"  {kind:6s} spells: n={n}, mean phase {mph:.1f}, R={R:.2f} (perm p={spell_p[kind][0]:.3f}) · "
          f"chi2 vs pool perm p={spell_p[kind][1]:.3f}")

# ---------- figure ----------
x = comp.phase.to_numpy()
fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))
a = ax[0, 0]
a.bar(x - 0.2, comp.cmz_z, 0.4, yerr=comp.cmz_z_sem, color="tab:green", label="CMZ rain z (IMERG)", capsize=2)
a.bar(x + 0.2, -comp.olr_z, 0.4, yerr=comp.olr_z_sem, color="tab:purple", label="-1 x OLR z (convection up)", capsize=2)
a.axhline(0, color="0.5", lw=0.8)
a.set_xlabel("BSISO1 phase"); a.set_ylabel("composite anomaly (z)")
a.set_title("Rain and independent convection through the BSISO cycle")
a.legend(fontsize=8); a.grid(alpha=0.25, axis="y")

a = ax[0, 1]
a.bar(x - 0.2, comp.frac_active * 100, 0.4, color="tab:red", label="% days in ACTIVE spell")
a.bar(x + 0.2, comp.frac_break * 100, 0.4, color="tab:blue", label="% days in BREAK spell")
a.axhline((ja.phase_ja == "active").mean() * 100, color="tab:red", ls=":", lw=1, label="base rates")
a.axhline((ja.phase_ja == "break").mean() * 100, color="tab:blue", ls=":", lw=1)
a.set_xlabel("BSISO1 phase"); a.set_ylabel("% of phase days")
a.set_title(f"Spell days by BSISO1 phase\nspell-level permutation p: active {spell_p['active'][1]:.3f}, "
            f"break {spell_p['break'][1]:.2f}")
a.legend(fontsize=8); a.grid(alpha=0.25, axis="y")

a = ax[1, 0]
a.bar(x, comp.objdays_per_day, 0.55, color="tab:blue")
a.axhline(org.n_obj.mean(), color="0.4", ls=":", lw=1, label="all-day mean")
a.set_xlabel("BSISO1 phase"); a.set_ylabel("ERE object-days / day")
a.set_title("ERE activity through the BSISO cycle")
a.legend(fontsize=8); a.grid(alpha=0.25, axis="y")

a = ax[1, 1]
a.plot(x, comp.ci_share * 100, "o-", color="#d95f02", label="Central-India share")
a.plot(x, comp.ne_share * 100, "s-", color="#7570b3", label="Northeast share")
a.set_xlabel("BSISO1 phase"); a.set_ylabel("% of ERE object-days")
a.set_title("ERE geography tracks BSISO phase")
a.legend(fontsize=8); a.grid(alpha=0.25)
plt.suptitle("TEST B — active/break spells and ERE geography vs BSISO1 phase (Jul-Aug, amp >= 1)",
             y=1.0, fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp8_02_bsiso_composites.png", dpi=150, bbox_inches="tight")
print("saved figures/cp8_02_bsiso_composites.png")
