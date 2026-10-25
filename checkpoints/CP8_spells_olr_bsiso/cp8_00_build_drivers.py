# cp8_00_build_drivers.py: daily driver table for the active/break spell validation:
# the CMZ rainfall index from CP7 + an independent CMZ OLR z-index (CPC blended OLR 2.5 deg)
# + the APCC Lee et al. (2013) BSISO index (PCs, amplitude, phase).
# OLR z uses exactly the rainfall-index recipe: per-calendar-day mean/std over the 25 JJAS
# seasons, both smoothed with a centred 15-day window. The CMZ OLR box (18-28N, 73-88E) is a
# plain area-weighted 2.5 deg box mean (no land mask).
# BSISO1 phase = octant of atan2(-PC1, PC2), numbered 1-8 anticlockwise as in Lee et al. (2013);
# with this convention the convection band is over India in phases 4-6 (checked in cp8_03).
# Output: data/derived/spell_drivers_daily.csv (3,050 JJAS days, 2000-2024)
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ERE_DATA", REPO / "data"))
DER = REPO / "data" / "derived"

season = pd.read_csv(DER / "cmz_daily_index.csv", dtype={"date": str})
season["dt"] = pd.to_datetime(season.date, format="%Y%m%d")
assert len(season) == 3050, "run CP7 cp7_00_prepare_tables.py first"

# ---------- CMZ OLR box mean ----------
ds = xr.open_dataset(DATA / "olr.cbo-2.5deg.day.mean.nc")
latsl = slice(18, 28) if ds.lat.values[0] < ds.lat.values[-1] else slice(28, 18)
box = ds["olr"].sel(lat=latsl, lon=slice(73, 88), time=slice("2000-01-01", "2024-12-31"))
ts = box.weighted(np.cos(np.deg2rad(box.lat))).mean(("lat", "lon")).to_series()
ds.close()
# CBO has a few whole-field missing days in 2024 (isolated 1-2 day holes): interpolate
print("missing OLR days filled:", ts.index[ts.isna()].strftime("%Y-%m-%d").tolist())
ts = ts.interpolate(limit=3)
season = season.merge(pd.DataFrame({"dt": ts.index.normalize(), "olr_wm2": ts.values}), on="dt", how="left")
assert season.olr_wm2.notna().all()

key = season.dt.dt.strftime("%m%d")
cal = pd.DataFrame({"olr": season.olr_wm2, "key": key})
cm, cs = cal.groupby("key").olr.mean(), cal.groupby("key").olr.std()
cm_map = dict(zip(cm.index, pd.Series(cm.to_numpy()).rolling(15, center=True, min_periods=1).mean()))
cs_map = dict(zip(cs.index, pd.Series(cs.to_numpy()).rolling(15, center=True, min_periods=1).mean()))
season["olr_z"] = [(o - cm_map[k]) / cs_map[k] for o, k in zip(season.olr_wm2, key)]

# ---------- APCC BSISO index ----------
bs = pd.read_csv(DATA / "BSISO.INDEX.NORM.LY.data", sep=r"\s+")
bs.columns = ["year", "doy", "pc1", "pc2", "pc3", "pc4", "amp1", "amp2"]
cols = ["pc1", "pc2", "pc3", "pc4", "amp1", "amp2"]
bs[cols] = bs[cols].where(bs.amp1 > -99)                 # -999.9 = missing (JJAS: 2015-06-21 only)
bs["dt"] = pd.to_datetime(bs.year.astype(str), format="%Y") + pd.to_timedelta(bs.doy - 1, unit="D")
assert (bs.dt.diff().dropna() == pd.Timedelta(days=1)).all(), "BSISO series has date gaps"


def octant(px, py):
    ph = (np.floor((np.degrees(np.arctan2(py, px)) + 180) / 45) % 8) + 1
    return ph.where(px.notna())


bs["bsiso1_phase"] = octant(bs.pc2, -bs.pc1)
bs["bsiso2_phase"] = octant(bs.pc4, -bs.pc3)
season = season.merge(bs[["dt", "pc1", "pc2", "amp1", "bsiso1_phase", "pc3", "pc4", "amp2", "bsiso2_phase"]]
                      .rename(columns={"pc1": "bsiso1_pc1", "pc2": "bsiso1_pc2", "amp1": "bsiso1_amp",
                                       "pc3": "bsiso2_pc1", "pc4": "bsiso2_pc2", "amp2": "bsiso2_amp"}),
                      on="dt", how="left")
assert season.bsiso1_amp.isna().sum() <= 1 and len(season) == 3050

season.drop(columns=["dt"]).to_csv(DER / "spell_drivers_daily.csv", index=False)
print(f"saved data/derived/spell_drivers_daily.csv: {len(season)} rows")
print(f"corr(cmz_z, olr_z) = {np.corrcoef(season.cmz_z, season.olr_z)[0, 1]:.3f} (expect negative)")
print("BSISO1 phase counts (JJAS):",
      {int(k): int(v) for k, v in season.bsiso1_phase.value_counts().sort_index().items()})
