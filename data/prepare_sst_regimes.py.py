import xarray as xr
import numpy as np
import os

NETCDF_PATH = r"C:\Users\DELL\Desktop\Capstone\2016-dec\20160101120000-ESACCI-L4_GHRSST-SST-GMPE-GLOB_CDR2.0-v02.0-fv01.0.nc"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

ds = xr.open_dataset(NETCDF_PATH)

sst = ds["analysed_sst"]
if "time" in sst.dims:
    sst = sst.isel(time=0)

sst_c = sst.values - 273.15
lat = ds["lat"].values
lon = ds["lon"].values

valid = sst_c[~np.isnan(sst_c)]
q1, q2, q3 = np.percentile(valid, [25, 50, 75])

regimes = {
    "winter": sst_c <= q1,
    "spring": (sst_c > q1) & (sst_c <= q2),
    "rainy": (sst_c > q2) & (sst_c <= q3),
    "summer": sst_c > q3
}

for name, mask in regimes.items():
    data = np.where(mask, sst_c, np.nan)

    ds_out = xr.Dataset(
        {"analysed_sst": (("lat", "lon"), data + 273.15)},
        coords={"lat": lat, "lon": lon},
        attrs={
            "season": name,
            "sst_mean_c": float(np.nanmean(data)),
            "source": "ESACCI GHRSST (real SST regime)"
        }
    )

    path = f"{OUT_DIR}/{name}_location.nc"
    ds_out.to_netcdf(path)
    print(f"Saved {path} | Mean SST = {np.nanmean(data):.2f} °C")

ds.close()
