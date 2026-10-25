# cp8_03_bsiso_phase_maps.py: check of the BSISO1 phase convention: composite CPC blended OLR
# anomaly maps by phase (JJAS 2000-2024, amplitude >= 1). The Lee et al. (2013) BSISO1 life cycle
# starts with convection over the equatorial Indian Ocean (phases 1-2) and carries it north over
# India (phases 4-6) and then to the Bay of Bengal / western North Pacific (phases 6-8).
import os
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("ERE_DATA", REPO / "data"))
DER = REPO / "data" / "derived"
FIG = REPO / "figures"

d = pd.read_csv(DER / "spell_drivers_daily.csv", dtype={"date": str})
d["dt"] = pd.to_datetime(d.date, format="%Y%m%d")

ds = xr.open_dataset(DATA / "olr.cbo-2.5deg.day.mean.nc")
latsl = slice(-10, 40) if ds.lat.values[0] < ds.lat.values[-1] else slice(40, -10)
fld = ds["olr"].sel(lat=latsl, lon=slice(40, 160)).sel(time=d.dt.to_numpy()).load()
ds.close()

# per-calendar-day climatology, 15-day smoothed (same recipe as the indices)
fld = fld.assign_coords(key=("time", pd.to_datetime(fld.time.values).strftime("%m%d")))
clim = fld.groupby("key").mean("time").rolling(key=15, center=True, min_periods=1).mean()
anom = fld.groupby("key") - clim

sel, phases = (d.bsiso1_amp >= 1).to_numpy(), d.bsiso1_phase.to_numpy()
fig, ax = plt.subplots(2, 4, figsize=(19, 6.5), sharex=True, sharey=True)
for k, ph in enumerate(range(1, 9)):
    m = sel & (phases == ph)
    comp = anom.isel(time=np.where(m)[0]).mean("time")
    a = ax[k // 4, k % 4]
    pm = a.pcolormesh(comp.lon, comp.lat, comp, cmap="RdBu_r", vmin=-20, vmax=20, shading="auto")
    a.plot([73, 88, 88, 73, 73], [18, 18, 28, 28, 18], "k--", lw=1)
    a.set_title(f"phase {ph} (n={m.sum()})", fontsize=10)
    a.axhline(0, color="0.5", lw=0.5)
fig.colorbar(pm, ax=ax, shrink=0.8, label="OLR anomaly (W/m2), blue = wet")
plt.suptitle("BSISO1 life cycle in CBO OLR anomalies (JJAS 2000-2024, amp>=1) — phase convention check",
             fontsize=13)
plt.savefig(FIG / "cp8_03_bsiso_phase_maps.png", dpi=130, bbox_inches="tight")
print("saved figures/cp8_03_bsiso_phase_maps.png")
