"""
CP1 - Datasets and extreme-rainfall thresholds

Project : Lagrangian tracking of Indian monsoon extreme rainfall events (ERE_Tracking)
Source  : cp1_datasets_thresholding.ipynb (converted cell by cell; figures and printed results are in the notebook)

Downloads ERA5 hourly total precipitation for JJAS 1998-2020 over India (1 deg) through the
Copernicus CDS API, aggregates it to daily mm/day and computes per-grid-cell 99th-percentile
wet-day thresholds used to define extreme rainfall events (EREs).
Set your own CDS key in ~/.cdsapirc before running (the placeholder YOUR-API-KEY-HERE is used here).

Originally run on Kaggle: input paths point to /kaggle/input/... (see data/README.md at the
repository root). Notebook shell commands such as `!pip install` are kept as comments.
"""


# %% Cell 1
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# %% Cell 2
import os

# Set up CDS API credentials
cds_config = """url: https://cds.climate.copernicus.eu/api
key: YOUR-API-KEY-HERE
"""

with open(os.path.expanduser('~/.cdsapirc'), 'w') as f:
    f.write(cds_config)

print("CDS credentials configured")


# %% Cell 3
# !pip install cdsapi -q

import cdsapi
import xarray as xr
import numpy as np
import gc

c = cdsapi.Client()
print("All imports successful")
print("CDS client ready")


# %% Cell 4
def download_and_aggregate(year, client):
    """
    Downloads all 24 hourly ERA5 precipitation values
    for one monsoon season (June-September),
    sums to true daily totals in mm/day,
    saves as a small daily NetCDF file,
    deletes the large hourly file immediately.
    """
    hours    = [f'{h:02d}:00' for h in range(24)]
    raw_file = f'/kaggle/working/era5_hourly_{year}.nc'
    day_file = f'/kaggle/working/era5_daily_{year}.nc'

    # Download hourly data
    print(f"\nDownloading {year} hourly data...")
    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type' : 'reanalysis',
            'variable'     : 'total_precipitation',
            'year'         : str(year),
            'month'        : ['06', '07', '08', '09'],
            'day'          : [str(d).zfill(2) for d in range(1, 32)],
            'time'         : hours,
            'format'       : 'netcdf',
            'area'         : [40, 60, 5, 100],  # N, W, S, E — India
            'grid'         : [1.0, 1.0],
        },
        raw_file
    )

    # Aggregate 24 hourly values → 1 daily total
    print(f"  Aggregating hourly → daily...")
    ds_hr  = xr.open_dataset(raw_file)
    ds_day = ds_hr['tp'].resample(valid_time='1D').sum() * 1000  # m → mm

    print(f"  Hourly shape : {ds_hr['tp'].shape}")
    print(f"  Daily shape  : {ds_day.shape}")
    print(f"  Daily max    : {float(ds_day.max()):.2f} mm/day")
    print(f"  Daily mean   : {float(ds_day.mean()):.2f} mm/day")

    # Save daily file
    ds_day.to_netcdf(day_file)
    print(f"  Saved → {day_file}")

    # Delete hourly file immediately to save space
    ds_hr.close()
    os.remove(raw_file)
    del ds_hr
    del ds_day
    gc.collect()

    print(f"  Hourly file deleted")
    print(f"  Year {year} complete ✅")

    return day_file


# %% Cell 5
years_all  = list(range(1998, 2021))

already_done = []
to_download  = []

for year in years_all:
    f = f'/kaggle/working/era5_daily_{year}.nc'
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024*1024)
        already_done.append(year)
        print(f"  ✅ {year} already done ({size_mb:.1f} MB)")
    else:
        to_download.append(year)
        print(f"  ⏳ {year} needed")

print(f"\nAlready done : {len(already_done)} years")
print(f"Still needed : {len(to_download)} years")


# %% Cell 6
import os
import gc
import xarray as xr

# set up credentials first
cds_config = """url: https://cds.climate.copernicus.eu/api
key: YOUR-API-KEY-HERE
"""
with open(os.path.expanduser('~/.cdsapirc'), 'w') as f:
    f.write(cds_config)

# verify the file was written correctly
with open(os.path.expanduser('~/.cdsapirc'), 'r') as f:
    print("credentials file:")
    print(f.read())

# now import and start client
# !pip install cdsapi -q
import cdsapi
c = cdsapi.Client()
print("client ready")

# download loop
years_all = list(range(1998, 2021))
hours     = [f'{h:02d}:00' for h in range(24)]
failed    = []

for year in years_all:
    
    day_file = f'/kaggle/working/era5_daily_{year}.nc'
    
    if os.path.exists(day_file):
        size_mb = os.path.getsize(day_file) / (1024*1024)
        print(f"{year} already done ({size_mb:.1f} MB) — skipping")
        continue
    
    raw_file = f'/kaggle/working/era5_hourly_{year}.nc'
    
    try:
        print(f"\n{year} downloading...")
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type' : 'reanalysis',
                'variable'     : 'total_precipitation',
                'year'         : str(year),
                'month'        : ['06', '07', '08', '09'],
                'day'          : [str(d).zfill(2) for d in range(1, 32)],
                'time'         : hours,
                'format'       : 'netcdf',
                'area'         : [40, 60, 5, 100],
                'grid'         : [1.0, 1.0],
            },
            raw_file
        )
        
        ds_hr  = xr.open_dataset(raw_file)
        ds_day = ds_hr['tp'].resample(valid_time='1D').sum() * 1000
        
        print(f"{year} shape : {ds_day.shape}")
        print(f"{year} max   : {float(ds_day.max()):.2f} mm/day")
        print(f"{year} mean  : {float(ds_day.mean()):.2f} mm/day")
        
        ds_day.to_netcdf(day_file)
        size_mb = os.path.getsize(day_file) / (1024*1024)
        print(f"{year} saved ({size_mb:.1f} MB) checkpoint ✓")
        
        ds_hr.close()
        os.remove(raw_file)
        del ds_hr, ds_day
        gc.collect()
        
    except Exception as e:
        print(f"{year} failed: {e}")
        failed.append(year)
        if os.path.exists(raw_file):
            os.remove(raw_file)
        continue

print(f"\nfinished")
print(f"failed: {failed if failed else 'none'}")


# %% Cell 7
import xarray as xr
import numpy as np
import os

# check all yearly files exist
print("=== FILE CHECK ===")
total_size = 0
missing = []

for year in range(1998, 2021):
    f = f'/kaggle/working/era5_daily_{year}.nc'
    if os.path.exists(f):
        size_mb = os.path.getsize(f) / (1024*1024)
        total_size += size_mb
        print(f"{year} — {size_mb:.1f} MB")
    else:
        missing.append(year)
        print(f"{year} — MISSING")

print(f"\ntotal size : {total_size:.1f} MB")
print(f"missing    : {missing if missing else 'none'}")

# load and check each year
print("\n=== PER YEAR STATS ===")
print(f"{'year':<6} {'days':<6} {'max':>10} {'mean':>10}")
print("-" * 35)

for year in range(1998, 2021):
    f = f'/kaggle/working/era5_daily_{year}.nc'
    if os.path.exists(f):
        ds = xr.open_dataarray(f)
        n_days = len(ds.valid_time)
        mx     = float(ds.max())
        mn     = float(ds.mean())
        print(f"{year:<6} {n_days:<6} {mx:>10.2f} {mn:>10.2f}")
        ds.close()

# load one year and do a physical check
print("\n=== PHYSICAL CHECK — 1998 ===")
ds = xr.open_dataarray('/kaggle/working/era5_daily_1998.nc')

# check seasonal cycle — july/august should be wetter than june/september
import pandas as pd
times = pd.to_datetime(ds.valid_time.values)
data  = ds.values  # (days, lat, lon)

june = data[times.month == 6].mean()
july = data[times.month == 7].mean()
aug  = data[times.month == 8].mean()
sept = data[times.month == 9].mean()

print(f"June mean  : {june:.2f} mm/day")
print(f"July mean  : {july:.2f} mm/day")
print(f"August mean: {aug:.2f} mm/day")
print(f"Sept mean  : {sept:.2f} mm/day")
print(f"July/August should be higher than June/September")

ds.close()

# check a known active monsoon year — 2007 should have high values
print("\n=== ACTIVE YEAR CHECK ===")
print("2007 and 2010 should have higher max/mean than 1999 and 2002")

for year in [1999, 2002, 2007, 2010]:
    f = f'/kaggle/working/era5_daily_{year}.nc'
    if os.path.exists(f):
        ds  = xr.open_dataarray(f)
        mx  = float(ds.max())
        mn  = float(ds.mean())
        print(f"{year} — max {mx:.1f}  mean {mn:.2f}")
        ds.close()


# %% Cell 8
import xarray as xr
import os

datasets = []

for year in range(1998, 2021):
    f = f'/kaggle/working/era5_daily_{year}.nc'
    ds = xr.open_dataarray(f)
    datasets.append(ds)
    print(f"loaded {year}")

ds_all   = xr.concat(datasets, dim='valid_time')
ds_all   = ds_all.rename({'valid_time': 'time', 
                           'latitude' : 'lat', 
                           'longitude': 'lon'})
ds_final = xr.Dataset({'precipitation': ds_all})

print(f"\nshape      : {ds_final['precipitation'].shape}")
print(f"time range : {str(ds_final.time.values[0])[:10]} to {str(ds_final.time.values[-1])[:10]}")
print(f"max        : {float(ds_final.precipitation.max()):.2f} mm/day")
print(f"mean       : {float(ds_final.precipitation.mean()):.2f} mm/day")

ds_final.to_netcdf('/kaggle/working/era5_india_daily_correct.nc')
print("\nsaved → era5_india_daily_correct.nc")


# %% Cell 9
import xarray as xr
import numpy as np
import pandas as pd

ds = xr.open_dataset('/kaggle/working/era5_india_daily_correct.nc')

print("=== BASIC STRUCTURE ===")
print(ds)
print()

print("=== KEY NUMBERS ===")
print(f"shape      : {ds['precipitation'].shape}")
print(f"time range : {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")
print(f"total days : {len(ds.time)}")
print(f"lat range  : {float(ds.lat.min())} to {float(ds.lat.max())}")
print(f"lon range  : {float(ds.lon.min())} to {float(ds.lon.max())}")
print()

print("=== PRECIPITATION VALUES ===")
print(f"max  : {float(ds.precipitation.max()):.2f} mm/day")
print(f"mean : {float(ds.precipitation.mean()):.2f} mm/day")
print(f"min  : {float(ds.precipitation.min()):.2f} mm/day")
print()

# check all 23 years are present
times = pd.to_datetime(ds.time.values)
years_present = sorted(times.year.unique())
print(f"=== YEARS PRESENT ===")
print(years_present)
print(f"count : {len(years_present)}")
print()

# check day counts per year
print("=== DAYS PER YEAR ===")
for yr in years_present:
    n = (times.year == yr).sum()
    print(f"{yr} : {n} days")

ds.close()


# %% Cell 10
import os

# search for the file
for root, dirs, files in os.walk('/kaggle/input'):
    for f in files:
        print(os.path.join(root, f))


# %% Cell 11
import xarray as xr

ds = xr.open_dataset('/kaggle/input/datasets/divyanshyecho/era5-india-daily-precip/era5_india_daily_correct.nc')

print(f"shape      : {ds['precipitation'].shape}")
print(f"time range : {str(ds.time.values[0])[:10]} to {str(ds.time.values[-1])[:10]}")
print(f"max        : {float(ds.precipitation.max()):.2f} mm/day")
print(f"mean       : {float(ds.precipitation.mean()):.2f} mm/day")

ds.close()
print("\ndataset loaded successfully")


# %% [markdown]
# The dataset has been loaded, 1998-2020 , june 1st to sep 30th , daily precipitation data in mm/day. Now making the thresholding model for ERE.


# %% Cell 12
import xarray as xr
import numpy as np
import pandas as pd
from scipy import ndimage

ds = xr.open_dataset('/kaggle/input/datasets/divyanshyecho/era5-india-daily-precip/era5_india_daily_correct.nc')

data_precip = np.array(ds['precipitation'][:])
lat         = ds['lat'].values
lon         = ds['lon'].values
time        = ds['time'].values

print(f"shape : {data_precip.shape}")
print(f"time  : {str(time[0])[:10]} to {str(time[-1])[:10]}")
print(f"max   : {np.nanmax(data_precip):.2f} mm/day")
print(f"mean  : {np.nanmean(data_precip):.2f} mm/day")


# %% Cell 13
data_precip1 = np.copy(data_precip)
data_precip1[data_precip1 <= 0.0] = np.nan

rainy = np.count_nonzero(~np.isnan(data_precip1))
dry   = np.count_nonzero(np.isnan(data_precip1))

print(f"rainy grid-days : {rainy}")
print(f"dry grid-days   : {dry}")
print(f"rainy fraction  : {rainy/(rainy+dry)*100:.1f}%")


# %% Cell 14
per_grid = np.full((len(lat), len(lon)), np.nan, dtype='float32')

for x in range(len(lat)):
    for y in range(len(lon)):
        col = data_precip1[:, x, y]
        col = col[~np.isnan(col)]
        if len(col) > 0:
            per_grid[x, y] = np.percentile(col, 99)

print(f"threshold shape : {per_grid.shape}")
print(f"min threshold   : {np.nanmin(per_grid):.2f} mm/day")
print(f"max threshold   : {np.nanmax(per_grid):.2f} mm/day")
print(f"mean threshold  : {np.nanmean(per_grid):.2f} mm/day")


# %% Cell 15
prep_thrsh = np.empty_like(data_precip)
for t in range(len(time)):
    prep_thrsh[t, :, :] = per_grid[:, :]

data_precip_safe = np.copy(data_precip)
data_precip_safe[np.isnan(data_precip_safe)] = 0.0
prep_thrsh[np.isnan(prep_thrsh)]             = 999.0

c = (data_precip_safe > prep_thrsh)

print(f"exceedance mask shape   : {c.shape}")
print(f"total extreme grid-days : {c.sum()}")
print(f"fraction exceeding      : {c.mean()*100:.2f}%")


# %% Cell 16
Loc_ERE       = np.zeros((len(time), len(lat), len(lon)), dtype='float32')
nt_per_day    = []
n_ere_per_day = []

for k in range(len(time)):
    img = c[k, :, :].astype(float)

    labeled, nr_objects = ndimage.label(img)

    nt_per_day.append(int(c[k, :, :].sum()))
    n_ere_per_day.append(nr_objects)

    if nr_objects >= 1:
        sfc_areas = np.bincount(labeled.flat)[1:]
        for i in range(nr_objects):
            sz  = sfc_areas[i]
            ind = np.where(labeled == i + 1)
            Loc_ERE[k, ind[0], ind[1]] = sz

    day_slice = Loc_ERE[k, :, :]
    day_slice[day_slice == 0.0] = np.nan
    Loc_ERE[k, :, :] = day_slice

    if k % 500 == 0:
        print(f"day {k}/{len(time)}")

print(f"\ndetection complete")
print(f"mean EREs per day : {np.mean(n_ere_per_day):.2f}")
print(f"max EREs one day  : {np.max(n_ere_per_day)}")
print(f"mean NT per day   : {np.mean(nt_per_day):.2f}")


# %% Cell 17
small  = (Loc_ERE == 1)
medium = (Loc_ERE >= 2) & (Loc_ERE <= 5)
large  = (Loc_ERE >= 6)

print("=== SIZE CLASSIFICATION ===")
print(f"small  (size=1)   : {small.sum()}")
print(f"medium (size 2-5) : {medium.sum()}")
print(f"large  (size>=6)  : {large.sum()}")

df = pd.DataFrame({
    'date' : pd.to_datetime(time),
    'NT'   : nt_per_day,
    'NE'   : n_ere_per_day,
})
df['year']  = df['date'].dt.year
df['S_bar'] = df['NT'] / df['NE'].replace(0, np.nan)

annual = df.groupby('year')[['NT', 'NE', 'S_bar']].mean().reset_index()
print("\n=== ANNUAL STATS ===")
print(annual.to_string(index=False))


# %% Cell 18
# These are well documented extreme rainfall events over India
# ERA5 should flag them as extreme even if values are smoothed

known_events = {
    '2005-07-26': (19, 73, 'Mumbai floods'),
    '2013-06-16': (30, 79, 'Uttarakhand floods'),
    '2007-07-15': (25, 85, 'Bihar floods'),
    '2010-08-06': (34, 74, 'Leh cloudbursts'),
}

print("=== KNOWN EXTREME EVENT CHECK ===\n")

times_pd = pd.to_datetime(time)

for date_str, (target_lat, target_lon, name) in known_events.items():
    
    date = pd.Timestamp(date_str)
    
    # check if date is in our dataset
    idx = np.where(times_pd == date)[0]
    if len(idx) == 0:
        print(f"{name} ({date_str}) — not in dataset (outside monsoon season or year range)")
        continue
    
    idx = idx[0]
    
    # find nearest grid cell
    lat_idx = np.argmin(np.abs(lat - target_lat))
    lon_idx = np.argmin(np.abs(lon - target_lon))
    
    actual_rain = data_precip[idx, lat_idx, lon_idx]
    threshold   = per_grid[lat_idx, lon_idx]
    exceeded    = actual_rain > threshold
    ere_size    = Loc_ERE[idx, lat_idx, lon_idx]
    
    print(f"{name} ({date_str})")
    print(f"  nearest grid    : {lat[lat_idx]}N {lon[lon_idx]}E")
    print(f"  rainfall        : {actual_rain:.2f} mm/day")
    print(f"  threshold       : {threshold:.2f} mm/day")
    print(f"  exceeded        : {exceeded}")
    print(f"  ERE size        : {ere_size if not np.isnan(ere_size) else 'not part of ERE'}")
    print(f"  NT on this day  : {nt_per_day[idx]}")
    print(f"  NE on this day  : {n_ere_per_day[idx]}")
    print()


# %% Cell 19
# july and august should have more extreme events than june and september
# this is a basic physical consistency check

print("=== SEASONAL CYCLE CHECK ===\n")

times_pd  = pd.to_datetime(time)
month_map = {6: 'June', 7: 'July', 8: 'August', 9: 'September'}

df_daily = pd.DataFrame({
    'date'  : times_pd,
    'month' : times_pd.month,
    'NT'    : nt_per_day,
    'NE'    : n_ere_per_day,
})
df_daily['S_bar'] = df_daily['NT'] / df_daily['NE'].replace(0, np.nan)

monthly = df_daily.groupby('month')[['NT', 'NE', 'S_bar']].mean()

print(f"{'month':<12} {'NT':>8} {'NE':>8} {'S_bar':>8}")
print("-" * 40)
for m in [6, 7, 8, 9]:
    print(f"{month_map[m]:<12} {monthly.loc[m,'NT']:>8.2f} {monthly.loc[m,'NE']:>8.2f} {monthly.loc[m,'S_bar']:>8.2f}")

print()
print("expected: July and August should have highest NT and NE")

# check if july/august peak
nt_values  = [monthly.loc[m, 'NT'] for m in [6,7,8,9]]
peak_month = [6,7,8,9][np.argmax(nt_values)]
print(f"actual peak month: {month_map[peak_month]}")
if peak_month in [7, 8]:
    print("seasonal cycle: PASS")
else:
    print("seasonal cycle: FAIL — check threshold calculation")


# %% Cell 20
# well documented active and weak monsoon years
# active years should have higher NT, NE and S_bar

print("=== ACTIVE VS WEAK MONSOON YEAR CHECK ===\n")

# documented weak monsoon years (drought/El Nino)
weak_years   = [2002, 2004, 2009, 2014, 2015]

# documented active monsoon years
active_years = [2007, 2010, 2011, 2019, 2020]

annual_stats = df.groupby('year')[['NT', 'NE', 'S_bar']].mean()

print("weak monsoon years:")
print(f"{'year':<6} {'NT':>8} {'NE':>8} {'S_bar':>8}")
print("-" * 35)
for yr in weak_years:
    if yr in annual_stats.index:
        row = annual_stats.loc[yr]
        print(f"{yr:<6} {row['NT']:>8.2f} {row['NE']:>8.2f} {row['S_bar']:>8.2f}")

print()
print("active monsoon years:")
print(f"{'year':<6} {'NT':>8} {'NE':>8} {'S_bar':>8}")
print("-" * 35)
for yr in active_years:
    if yr in annual_stats.index:
        row = annual_stats.loc[yr]
        print(f"{yr:<6} {row['NT']:>8.2f} {row['NE']:>8.2f} {row['S_bar']:>8.2f}")

print()
weak_mean_NT   = annual_stats.loc[annual_stats.index.isin(weak_years), 'NT'].mean()
active_mean_NT = annual_stats.loc[annual_stats.index.isin(active_years), 'NT'].mean()
print(f"mean NT weak years   : {weak_mean_NT:.2f}")
print(f"mean NT active years : {active_mean_NT:.2f}")

if active_mean_NT > weak_mean_NT:
    print("active vs weak check : PASS")
else:
    print("active vs weak check : FAIL")


# %% Cell 21
print("=== SPATIAL PATTERN CHECK ===\n")

ere_freq = (~np.isnan(Loc_ERE)).sum(axis=0)

regions = {
    'Western Ghats'   : {'lat_min': 8,  'lat_max': 20, 'lon_min': 72, 'lon_max': 76},
    'Central India'   : {'lat_min': 15, 'lat_max': 25, 'lon_min': 75, 'lon_max': 85},
    'Northeast India' : {'lat_min': 22, 'lat_max': 30, 'lon_min': 88, 'lon_max': 97},
    'Northwest India' : {'lat_min': 25, 'lat_max': 35, 'lon_min': 68, 'lon_max': 76},
    'Pakistan/Desert' : {'lat_min': 25, 'lat_max': 35, 'lon_min': 60, 'lon_max': 68},
}

for region_name, bounds in regions.items():
    lat_idx = np.where((lat >= bounds['lat_min']) & (lat <= bounds['lat_max']))[0]
    lon_idx = np.where((lon >= bounds['lon_min']) & (lon <= bounds['lon_max']))[0]
    
    region_freq = ere_freq[np.ix_(lat_idx, lon_idx)].mean()
    print(f"{region_name:<20} : {region_freq:.1f} days with ERE")

print()
print("expected: Western Ghats and Central India highest")
print("expected: Pakistan/Desert lowest")


# %% Cell 22
# the 99th percentile threshold should be higher in wet regions
# and lower in dry regions
# also check fraction exceeding is close to 1%

print("=== THRESHOLD SANITY CHECK ===\n")

# fraction of rainy days exceeding threshold per grid
# should be close to 1% everywhere by definition
exceedance_rate = c.mean(axis=0) * 100  # percentage

print(f"mean exceedance rate across all grids : {exceedance_rate.mean():.2f}%")
print(f"max exceedance rate at any grid       : {exceedance_rate.max():.2f}%")
print(f"min exceedance rate at any grid       : {exceedance_rate.min():.2f}%")
print(f"expected: all values close to 1%")
print()

# threshold should correlate with mean rainfall
mean_rain = np.nanmean(data_precip, axis=0)
corr = np.corrcoef(mean_rain.flatten(), per_grid.flatten())[0,1]
print(f"correlation between mean rain and threshold : {corr:.3f}")
print(f"expected: high positive correlation (>0.7)")
if corr > 0.7:
    print("threshold spatial pattern : PASS")
else:
    print("threshold spatial pattern : CHECK — lower than expected")
print()

# wet vs dry region thresholds
wet_threshold = per_grid[
    np.ix_(np.where((lat >= 10) & (lat <= 20))[0],
           np.where((lon >= 72) & (lon <= 76))[0])
].mean()

dry_threshold = per_grid[
    np.ix_(np.where((lat >= 25) & (lat <= 35))[0],
           np.where((lon >= 60) & (lon <= 68))[0])
].mean()

print(f"mean threshold Western Ghats : {wet_threshold:.2f} mm/day")
print(f"mean threshold Pakistan/dry  : {dry_threshold:.2f} mm/day")
print(f"wet region should have higher threshold than dry region")
if wet_threshold > dry_threshold:
    print("wet vs dry threshold : PASS")
else:
    print("wet vs dry threshold : FAIL")


# %% [markdown]
# The 99 percentile must be month based and not all the months should be used together. More Nt for the month of june.
