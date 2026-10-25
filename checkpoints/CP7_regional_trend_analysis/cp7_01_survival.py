# cp7_01_survival.py: does size predict track survival, beyond pure geometry? (README section 1)
# (a) P(track continues tomorrow | today's size), censored endings removed
# (b) geometric null: same-shape footprint slid to random India positions: how often would it
#     touch ANY real next-day object (the tracker's continuation condition)?
# (c) day-1 size as a predictor of becoming multi-day (AUC, overall and per season)
# (d) logistic model (IRLS) for adjusted effects of size, mean and peak-cell intensity
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

stats = pd.read_parquet(DER / "cp7_objdays.parquet")
tracks = pd.read_parquet(DER / "cp7_tracks.parquet")
z = np.load(DER / "imerg_setup.npz")
Label8, india_mask = z["Label8"], z["india_mask"]
times = pd.to_datetime(z["times"], format="%Y%m%d")
date_to_idx = {d: i for i, d in enumerate(times.strftime("%Y%m%d"))}
n_lat, n_lon = india_mask.shape
rng = np.random.default_rng(42)

# ---------- (a) conditional continuation probability ----------
s = stats.copy()
s["is_last"] = s.day_of_track == (s.duration - 1)
# censored final days: the track did not die naturally (season end, left domain, merged away)
CENSOR = {"season_end", "domain_exit", "merge_death"}
s["censored_last"] = s.is_last & s.end_type.isin(CENSOR)
u = s[~s.censored_last].copy()
u["continues"] = (~u.is_last).astype(int)

edges = [0.5, 1.5, 2.5, 4.5, 9.5, 19.5, 49.5, np.inf]
labels = ["1", "2", "3-4", "5-9", "10-19", "20-49", "50+"]
u["szbin"] = pd.cut(u["size"], edges, labels=labels)


def wilson(k, n, zz=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n
    den = 1 + zz**2 / n
    c = (p + zz**2 / (2 * n)) / den
    h = zz * np.sqrt(p * (1 - p) / n + zz**2 / (4 * n**2)) / den
    return c - h, c + h


rows = []
for b in labels:
    g = u[u.szbin == b]
    k, n = int(g.continues.sum()), len(g)
    lo, hi = wilson(k, n)
    rows.append(dict(size_bin=b, n_days=n, n_continue=k,
                     p_continue=k / n if n else np.nan, ci_lo=lo, ci_hi=hi))
cont = pd.DataFrame(rows)
print(f"object-days {len(s)} · censored last days removed {int(s.censored_last.sum())} · usable {len(u)}")
print("P(continue tomorrow | size today):")
print(cont.round(3).to_string(index=False))

# ---------- (b) geometric null: random relocation ----------
sept30 = set(np.where((times.month == 9) & (times.day == 30))[0])
next_day_mask = {t: Label8[t + 1] > 0 for t in range(Label8.shape[0] - 1) if t not in sept30}

N_SHIFTS = 60
sub = u[u["size"] >= 1].copy()
sub["t"] = sub.date.map(date_to_idx)
sub = sub[sub.t.isin(next_day_mask.keys())]
p_null = np.full(len(sub), np.nan)
obs_overlap = np.zeros(len(sub), dtype=bool)
for i, (t, lbl) in enumerate(zip(sub.t.to_numpy(), sub.day_lbl.to_numpy())):
    ii, jj = np.where(Label8[t] == int(lbl))
    nm = next_day_mask[t]
    obs_overlap[i] = nm[ii, jj].any()
    di_lo, di_hi = -ii.min(), n_lat - 1 - ii.max()
    dj_lo, dj_hi = -jj.min(), n_lon - 1 - jj.max()
    hits = tries = attempts = 0
    while tries < N_SHIFTS and attempts < N_SHIFTS * 6:
        attempts += 1
        di = rng.integers(di_lo, di_hi + 1); dj = rng.integers(dj_lo, dj_hi + 1)
        i2, j2 = ii + di, jj + dj
        if india_mask[i2, j2].mean() < 0.5: continue     # keep >= 50% of the footprint on India
        tries += 1
        if nm[i2, j2].any(): hits += 1
    p_null[i] = hits / tries if tries else np.nan
sub["p_null"] = p_null
sub["obs_overlap"] = obs_overlap
sub["szbin"] = pd.cut(sub["size"], edges, labels=labels)
geo = sub.groupby("szbin", observed=True).agg(
    n=("p_null", "size"), observed_cont=("continues", "mean"),
    observed_overlap=("obs_overlap", "mean"), null_overlap=("p_null", "mean")).reset_index()
geo["ratio"] = geo.observed_cont / geo.null_overlap
print("\nGeometric null (random relocation, >=1 shared cell with any next-day object):")
print(geo.round(3).to_string(index=False))

# ---------- (c) day-1 size predicting multi-day ----------
tr = tracks[~tracks.end_type.isin(CENSOR)].copy()      # uncensored tracks only
tr["multiday"] = (tr.duration >= 2).astype(int)
aucs = {}
for col in ["day1_size", "day1_max_mm", "day1_vol"]:
    a = tr[tr.multiday == 1][col].to_numpy(); b = tr[tr.multiday == 0][col].to_numpy()
    U, pval = mannwhitneyu(a, b, alternative="greater")
    aucs[col] = U / (len(a) * len(b))
    print(f"AUC {col:12s} -> multi-day = {aucs[col]:.3f} (p={pval:.1e})")
auc_size = aucs["day1_size"]
per_year = []
for yr, g in tr.groupby("year"):
    if g.multiday.sum() >= 5:
        a = g[g.multiday == 1].day1_size; b = g[g.multiday == 0].day1_size
        per_year.append(mannwhitneyu(a, b).statistic / (len(a) * len(b)))
print(f"per-season AUC (day-1 size): median {np.median(per_year):.2f}, "
      f"range {min(per_year):.2f}-{max(per_year):.2f} over {len(per_year)} seasons")

tr["d1bin"] = pd.cut(tr.day1_size, edges, labels=labels)
md = tr.groupby("d1bin", observed=True).agg(n=("multiday", "size"),
                                            p_multiday=("multiday", "mean")).reset_index()
print("\nP(multi-day | day-1 size):")
print(md.round(3).to_string(index=False))


# ---------- (d) multivariate logistic via IRLS ----------
def logit_irls(X, y, iters=60):
    Xd = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xd.shape[1])
    for _ in range(iters):
        eta = Xd @ w
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = p * (1 - p) + 1e-9
        w_new = np.linalg.solve(Xd.T @ (Xd * W[:, None]), Xd.T @ (W * (eta + (y - p) / W)))
        if np.max(np.abs(w_new - w)) < 1e-8:
            w = w_new; break
        w = w_new
    p = 1 / (1 + np.exp(-(Xd @ w)))
    W = p * (1 - p) + 1e-9
    se = np.sqrt(np.diag(np.linalg.inv(Xd.T @ (Xd * W[:, None]))))
    return w, se


X = np.log10(u[["size", "mean_mm", "max_mm"]].to_numpy(float))
w, se = logit_irls((X - X.mean(0)) / X.std(0), u["continues"].to_numpy(float))
print("\nLogistic P(continue) ~ standardized predictors (odds ratio per 1 sd):")
for nm, wi, si in zip(["intercept", "log_size", "log_mean_mm", "log_max_mm"], w, se):
    print(f"  {nm:12s} beta={wi:+.3f} (se {si:.3f})  OR={np.exp(wi):.2f}")

# ---------- figure ----------
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
x = np.arange(len(labels))
ax[0].errorbar(x, cont.p_continue, yerr=[cont.p_continue - cont.ci_lo, cont.ci_hi - cont.p_continue],
               fmt="o-", color="tab:blue", capsize=3, label="observed P(continue)")
ax[0].plot(x, geo.set_index("szbin").reindex(labels).null_overlap, "s--", color="0.5",
           label="geometric null\n(random relocation)")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
ax[0].set_xlabel("object size today (cells)"); ax[0].set_ylabel("probability")
ax[0].set_title("Survival vs size, against pure geometry")
ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

ax[1].bar(np.arange(len(md)), md.p_multiday, color="tab:green", width=0.7)
ax[1].set_xticks(np.arange(len(md))); ax[1].set_xticklabels(md.d1bin)
for k, (n, p) in enumerate(zip(md.n, md.p_multiday)):
    ax[1].text(k, p + 0.01, f"n={n}", ha="center", fontsize=8)
ax[1].set_xlabel("size on day 1 (cells)"); ax[1].set_ylabel("P(track lasts 2+ days)")
ax[1].set_title(f"Birth size predicts longevity (AUC={auc_size:.2f})")
ax[1].grid(alpha=0.25, axis="y")

o = np.exp(w[1:]); o_lo = np.exp(w[1:] - 1.96 * se[1:]); o_hi = np.exp(w[1:] + 1.96 * se[1:])
ax[2].errorbar(o, np.arange(3), xerr=[o - o_lo, o_hi - o], fmt="o", color="tab:red", capsize=4)
ax[2].axvline(1, color="0.5", ls=":")
ax[2].set_yticks(np.arange(3)); ax[2].set_yticklabels(["log size", "log mean int.", "log max int."])
ax[2].set_xlabel("odds ratio for continuing (per 1 sd)")
ax[2].set_title("Adjusted effects (logistic)")
ax[2].grid(alpha=0.25, axis="x")
plt.suptitle("ERE track survival: size dominates, and it is not just geometry", y=1.02, fontsize=13)
plt.tight_layout()
plt.savefig(FIG / "cp7_01_survival.png", dpi=150, bbox_inches="tight")
print("\nsaved figures/cp7_01_survival.png")
