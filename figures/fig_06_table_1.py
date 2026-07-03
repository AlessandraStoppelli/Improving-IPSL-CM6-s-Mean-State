#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.



import xarray as xr
import numpy as np
import xesmf as xe
from tqdm import tqdm

# from scipy.stats import ttest_1samp
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from matplotlib.ticker import FormatStrFormatter




#### Grids

file_nemo = xr.open_dataset('/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-r4_merged_grid_T.nc').sel(time_counter=slice('1950-01-01', '1950-02-01'))
area_nemo = file_nemo['cell_area']

file_regular = xr.open_dataset('/data/astoppel/CMIP6/ensamble_historical/ipsl_CMIP6_TOS_1921_2014.nc')
grid_regular = {"lon": file_regular["lon"], "lat": file_regular["lat"]}




####################################################
#### Part one : for the time series of the ZG
####################################################




#### Functions

def annual_mean_time_counter(ds_list):
    
    return [ds.groupby('time_counter.year').mean(dim='time_counter') for ds in ds_list]


def compute_zgi(sst_list, area):
    
    zgi_list = []
    west_list = []
    east_list = []

    for sst in sst_list:
        
        def box_mean(ds, lat_min, lat_max, lon_min, lon_max):
            lat_mask = (ds.nav_lat >= lat_min) & (ds.nav_lat <= lat_max)
            lon_mask = (ds.nav_lon >= lon_min) & (ds.nav_lon <= lon_max)
            geo_mask = lat_mask & lon_mask
            
            ds_box = ds.where(geo_mask, drop=True)
            area_box = area.where(geo_mask, drop=True)
            
            return ds_box.weighted(area_box).mean(dim=['y', 'x'], skipna=True)
    
        west = box_mean(sst, -5, 5, 110,180)  
        east = box_mean(sst, -5, 5, -180, -80)  
        Trop = box_mean(sst, -30, 30, -180, 180)  

        zgi_list.append(west - east)
        west_list.append(west - Trop)
        east_list.append(east - Trop)
        
    return zgi_list, west_list, east_list


def rolling_mean(yearly_da, window=31):
    
    """
    A 31-year centered moving average was applied along the temporal dimension. 
    For year i the mean is computed using data from i-15, i+ 15. 
    When these limits exceed the bounds of the time series the window is truncated to the available years
    """
    
    half_window = window // 2
    data = yearly_da.values
    smoothed = np.zeros_like(data)

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        smoothed[i] = np.mean(data[start:end])

    return xr.DataArray(
        smoothed,
        dims=["year"],                 # keep solo la dimensione year
        coords={"year": yearly_da.year}  
    )

def get_CI_90(values, n):
     
    std = np.std(values, axis=0, ddof=1)

    alpha = 0.10  # 90% confidence
    
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    return t_critical * (std / np.sqrt(n))



def load_ensemble(base_path, scenario, pattern, members):
    ds_list = []
    for member in tqdm(members, desc=f"Loading {scenario}"): # crea una progress bar
        file = pattern.format(member=member)
        try:
            ds = xr.open_dataset(f"{base_path}{file}")
            ds_list.append(ds)
        except FileNotFoundError:
            print(f"File not found: {base_path}{file}")
            continue
    return ds_list
    

def merge_historical_future(hist_list, fut_list, var_name, time_dim='time_counter'):
    """Unisce periodi storici e futuri"""
    merged = []
    for h_ds, f_ds in zip(hist_list, fut_list):

        h_ds = h_ds.rename({time_dim: 'time_counter'}) if time_dim != 'time_counter' else h_ds
        f_ds = f_ds.rename({time_dim: 'time_counter'}) if time_dim != 'time_counter' else f_ds
        
        # Seleziona periodi
        hist = h_ds[var_name].sel(time_counter=slice('1950-01-01', '2014-12-31'))
        fut = f_ds[var_name].sel(time_counter=slice('2015-01-01', '2099-12-31'))
        
        merged.append(xr.concat([hist, fut], dim='time_counter'))
    return merged




#### Upload models and observations

members = ['r1', 'r2', 'r3', 'r4', 'r14', 'r33', 'r12', 'r13', 'r15', 'r16', 'r18', 'r22', 'r23', 'r29', 'r30'] 
base_fa = "/scratchu/astoppel/IPSL_FA/"
base_ctl ="/scratchu/astoppel/IPSL_CTL/"

# ERA5 
era5 = xr.open_dataset("/data/astoppel/ERA5/ERA5_tos_1949_2019_gn.nc").sel(time_counter=slice("1950-01-01", "2019-12-31"))
era5_last = xr.open_dataset("/data/astoppel/ERA5/ERA5_tos_2020_2024_gn_Kelvin.nc") 
era5_last['tos'] = era5_last['tos'] - 273.15 
era5_all = xr.concat([era5, era5_last], dim="time_counter")


# HadISST 
outfile = "/data/astoppel/HadISST/HadISST_gn_regridded_2025.nc"
hadisst = xr.open_dataset(outfile).sel(time_counter=slice("1950-01-01", "2024-12-31"))

#ERSST 
ersstv5 = xr.open_dataset("/scratchu/astoppel/ERSSTv5/Monthly/sst.mnmean_regridded.nc").sel(time_counter=slice("1950-01-01", "2024-12-31"))

#COBE2 
cobe2 = xr.open_dataset("/scratchu/astoppel/COBE2/Monthly/sst.mon.mean_regridded.nc").sel(time_counter=slice("1950-01-01", "2024-12-31"))

#IPSLFA
hist_fa = load_ensemble(base_fa, "historical", 
                       "historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-{member}_merged_grid_T.nc", members)
fut_fa = load_ensemble(base_fa, "ssp585", 
                      "ssp585/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-ssp585-{member}_merged_grid_T.nc", members)

tos_fa = merge_historical_future(hist_fa, fut_fa, 'tos')

#IPSL
hist_ctl = load_ensemble(base_ctl, "historical",
                        "historical/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_historical_{member}_gn_1950-2014.nc",
                        members)


fut_ctl = load_ensemble(base_ctl, "ssp585",
                        "ssp585/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_ssp585_{member}_gn_2015_2099.nc",
                        members)

tos_ctl = merge_historical_future(hist_ctl, fut_ctl, 'tos', 'time')




#### Compute annual mean 

era5_ann = annual_mean_time_counter([era5_all['tos']]) 
hadisst_ann = annual_mean_time_counter([hadisst['tos']]) 
ersstv5_ann = annual_mean_time_counter([ersstv5['sst']]) 
cobe2_ann = annual_mean_time_counter([cobe2['sst']])  
tos_fa_ann = annual_mean_time_counter(tos_fa)
tos_ctl_ann = annual_mean_time_counter(tos_ctl) 




#### Compute ZGI
zgi_era5, west_era5, east_era5 = compute_zgi(era5_ann, area_nemo)
zgi_hadisst, west_hadisst, east_hadisst  = compute_zgi(hadisst_ann, area_nemo)
zgi_ersstv5, west_ersstv5, east_ersstv5  = compute_zgi(ersstv5_ann, area_nemo)
zgi_cobe2, west_cobe2, east_cobe2  = compute_zgi(cobe2_ann, area_nemo)

zgi_fa, west_fa, east_fa = compute_zgi(tos_fa_ann, area_nemo)
zgi_ctl, west_ctl, east_ctl = compute_zgi(tos_ctl_ann, area_nemo)




#### multimember for ipsl. ipsl-fa

zgi_ctl = xr.concat(zgi_ctl, dim="member")
west_ctl = xr.concat(west_ctl, dim="member")
east_ctl = xr.concat(east_ctl, dim="member")

zgi_fa  = xr.concat(zgi_fa, dim="member")
west_fa = xr.concat(west_fa, dim="member")
east_fa = xr.concat(east_fa, dim="member")

zgi_ctl["member"], west_ctl["member"], east_ctl["member"] = [i for i in members], [i for i in members], [i for i in members]
zgi_fa["member"], west_fa["member"], east_fa["member"]  = [i for i in members], [i for i in members], [i for i in members]





#### smoothing timeseries 

#obs

smoothed_zgi_era5 = rolling_mean(zgi_era5[0], window=31)
smoothed_zgi_hadisst = rolling_mean(zgi_hadisst[0], window=31)
smoothed_zgi_ersstv5 = rolling_mean(zgi_ersstv5[0], window=31)
smoothed_zgi_cobe2 = rolling_mean(zgi_cobe2[0], window=31)

smoothed_west_era5 = rolling_mean(west_era5[0], window=31)
smoothed_west_hadisst = rolling_mean(west_hadisst[0], window=31)
smoothed_west_ersstv5 = rolling_mean(west_ersstv5[0], window=31)
smoothed_west_cobe2 = rolling_mean(west_cobe2[0], window=31)

smoothed_east_era5 = rolling_mean(east_era5[0], window=31)
smoothed_east_hadisst = rolling_mean(east_hadisst[0], window=31)
smoothed_east_ersstv5 = rolling_mean(east_ersstv5[0], window=31)
smoothed_east_cobe2 = rolling_mean(east_cobe2[0], window=31)


smoothed_zgi_members_ctl, smoothed_west_members_ctl, smoothed_east_members_ctl = [], [], []
smoothed_zgi_members_fa, smoothed_west_members_fa, smoothed_east_members_fa  = [], [], []

# ipsl 
for member in zgi_ctl.member.values:
    
    da_member_zgi = zgi_ctl.sel(member=member)
    da_member_west = west_ctl.sel(member=member)
    da_member_east = east_ctl.sel(member=member)
    
    smoothed_zgi = rolling_mean(da_member_zgi, window=31)
    smoothed_west = rolling_mean(da_member_west, window=31)
    smoothed_east = rolling_mean(da_member_east, window=31)

    smoothed_zgi = smoothed_zgi.assign_coords(member=member)  
    smoothed_west = smoothed_west.assign_coords(member=member)  
    smoothed_east = smoothed_east.assign_coords(member=member)  

    smoothed_zgi_members_ctl.append(smoothed_zgi)
    smoothed_west_members_ctl.append(smoothed_west)
    smoothed_east_members_ctl.append(smoothed_east)


# Concateno lungo la dimensione member
zgi_ctl_smooth = xr.concat(smoothed_zgi_members_ctl, dim="member")
west_ctl_smooth = xr.concat(smoothed_west_members_ctl, dim="member")
east_ctl_smooth = xr.concat(smoothed_east_members_ctl, dim="member")

# --- Ciclo su ogni membro FA ---
for member in zgi_fa.member.values:
    da_member_zgi = zgi_fa.sel(member=member)
    da_member_west = west_fa.sel(member=member)
    da_member_east = east_fa.sel(member=member)
    
    smoothed_zgi = rolling_mean(da_member_zgi, window=31)
    smoothed_west = rolling_mean(da_member_west, window=31)
    smoothed_east = rolling_mean(da_member_east, window=31)

    smoothed_zgi = smoothed_zgi.assign_coords(member=member)  
    smoothed_west = smoothed_west.assign_coords(member=member)  
    smoothed_east = smoothed_east.assign_coords(member=member)  

    smoothed_zgi_members_fa.append(smoothed_zgi)
    smoothed_west_members_fa.append(smoothed_west)
    smoothed_east_members_fa.append(smoothed_east)


zgi_fa_smooth = xr.concat(smoothed_zgi_members_fa, dim="member")
west_fa_smooth = xr.concat(smoothed_west_members_fa, dim="member")
east_fa_smooth = xr.concat(smoothed_east_members_fa, dim="member")






#### Compute anomalies with respect to the reference period (obs)

baseline_period = slice("1950", "1979")

#ZG obs
baseline_zgi_era5  = zgi_era5[0].sel(year=baseline_period).mean("year")
zgi_era5_anom  = smoothed_zgi_era5 - baseline_zgi_era5

baseline_zgi_hadisst  = zgi_hadisst[0].sel(year=baseline_period).mean("year")
zgi_hadisst_anom  = smoothed_zgi_hadisst - baseline_zgi_hadisst

baseline_zgi_ersstv5  = zgi_ersstv5[0].sel(year=baseline_period).mean("year")
zgi_ersstv5_anom  = smoothed_zgi_ersstv5 - baseline_zgi_ersstv5

baseline_zgi_cobe2  = zgi_cobe2[0].sel(year=baseline_period).mean("year")
zgi_cobe2_anom  = smoothed_zgi_cobe2 - baseline_zgi_cobe2

#WEST obs
baseline_west_era5 = west_era5[0].sel(year=baseline_period).mean("year")
west_era5_anom = smoothed_west_era5 - baseline_west_era5

baseline_west_hadisst  = west_hadisst[0].sel(year=baseline_period).mean("year")
west_hadisst_anom  = smoothed_west_hadisst - baseline_west_hadisst

baseline_west_ersstv5  = west_ersstv5[0].sel(year=baseline_period).mean("year")
west_ersstv5_anom  = smoothed_west_ersstv5 - baseline_west_ersstv5

baseline_west_cobe2  = west_cobe2[0].sel(year=baseline_period).mean("year")
west_cobe2_anom  = smoothed_west_cobe2 - baseline_west_cobe2

#EAST obs
baseline_east_era5 = east_era5[0].sel(year=baseline_period).mean("year")
east_era5_anom = smoothed_east_era5 - baseline_east_era5

baseline_east_hadisst  = east_hadisst[0].sel(year=baseline_period).mean("year")
east_hadisst_anom  = smoothed_east_hadisst - baseline_east_hadisst

baseline_east_ersstv5  = east_ersstv5[0].sel(year=baseline_period).mean("year")
east_ersstv5_anom  = smoothed_east_ersstv5 - baseline_east_ersstv5

baseline_east_cobe2  = east_cobe2[0].sel(year=baseline_period).mean("year")


#ZG fa/ipsl
baseline_zgi_ctl = zgi_ctl.sel(year=baseline_period).mean("year")
zgi_ctl_anom_smooth = zgi_ctl_smooth - baseline_zgi_ctl

baseline_zgi_fa  = zgi_fa.sel(year=baseline_period).mean("year")
zgi_fa_anom_smooth  = zgi_fa_smooth - baseline_zgi_fa

#WEST fa/ipsl
baseline_west_ctl = west_ctl.sel(year=baseline_period).mean("year")
west_ctl_anom_smooth = west_ctl_smooth - baseline_west_ctl 

baseline_west_fa  = west_fa.sel(year=baseline_period).mean("year")
west_fa_anom_smooth  = west_fa_smooth - baseline_west_fa 

#EAST fa/ipsl
baseline_east_ctl = east_ctl.sel(year=baseline_period).mean("year")
east_ctl_anom_smooth = east_ctl_smooth - baseline_east_ctl 

baseline_east_fa  = east_fa.sel(year=baseline_period).mean("year")
east_fa_anom_smooth  = east_fa_smooth - baseline_east_fa 

#differences west east: ipsl/ipsl-fa

diff_west_mem_smooth = west_fa_anom_smooth - west_ctl_anom_smooth

diff_east_mem_smooth = -1 * (east_fa_anom_smooth - east_ctl_anom_smooth)





#### IPSL/IPSL-FA CI 90%

# IPSL
ctl_mean_smooth = zgi_ctl_anom_smooth.mean("member")
ctl_ci          = get_CI_90(zgi_ctl_anom_smooth, n=15)  
ctl_mean_sig    = ctl_mean_smooth.where(np.abs(ctl_mean_smooth) > ctl_ci)   

# IPSL-FA
fa_mean_smooth = zgi_fa_anom_smooth.mean("member")
fa_ci          = get_CI_90(zgi_fa_anom_smooth, n=15)  
fa_mean_sig    = fa_mean_smooth.where(np.abs(fa_mean_smooth) > fa_ci)


# WEP difference
diff_west_mean_smooth = diff_west_mem_smooth.mean("member")
west_ci        = get_CI_90(diff_west_mem_smooth, n=15) 
west_mean_sig = diff_west_mean_smooth.where(np.abs(diff_west_mean_smooth) > west_ci)

# EEP difference
diff_east_mean_smooth  = diff_east_mem_smooth.mean("member")
east_ci       = get_CI_90(diff_east_mem_smooth, n=15)  
east_mean_sig = diff_east_mean_smooth.where(np.abs(diff_east_mean_smooth) > east_ci)


#### IPSL/IPSL-FA 5-95 PERCENTILE 

fa_perc_5 = np.percentile(zgi_fa_anom_smooth, 5, axis=0)
fa_perc_95 = np.percentile(zgi_fa_anom_smooth, 95, axis=0)

ctl_perc_5 = np.percentile(zgi_ctl_anom_smooth, 5, axis=0)
ctl_perc_95 = np.percentile(zgi_ctl_anom_smooth, 95, axis=0)

# WEP difference
perc_5_west = np.percentile(diff_west_mem_smooth, 5, axis=0)     
perc_95_west = np.percentile(diff_west_mem_smooth, 95, axis=0)   

# EEP difference
perc_5_east = np.percentile(diff_east_mem_smooth, 5, axis=0)     
perc_95_east = np.percentile(diff_east_mem_smooth, 95, axis=0)   




####################################################
#### Part two : compute percentage of change, CI for table 1 
#### (you can add the part of cmip6 from figure 1 python code to get cmip6 values)
####################################################




#### Functions

def get_period_means(data, period):

    period_list = []

    for m in data.member:  
        zgi = data.sel(member=m)

        data_period_mean = zgi.sel(year=period).mean(dim="year")
        period_list.append(data_period_mean)

    return period_list




present_period = slice("1950", "1979") #ref period 
#future_period  = slice("1980", "2024") #fut period 1
future_period  = slice("2070", "2099") #fut period 2
#future_period  = slice("2005", "2035") #fut period 3


ctl_present = get_period_means(zgi_ctl, present_period)
ctl_future = get_period_means(zgi_ctl, future_period)

fa_present = get_period_means(zgi_fa, present_period)  
fa_future = get_period_means(zgi_fa, future_period)

# cmip6_present = get_period_means(zgi_cmip6, present_period)  
# cmip6_future = get_period_means(zgi_cmip6, future_period)

# Percentage change for each member
ctl_changes = [(f-p)/p*100 for p,f in zip(ctl_present, ctl_future)]
ctl_changes_members = xr.concat(ctl_changes, dim="member")

fa_changes = [(f-p)/p*100 for p,f in zip(fa_present, fa_future)]
fa_changes_members = xr.concat(fa_changes, dim="member")

# cmip6_changes = [(f-p)/p*100 for p,f in zip(cmip6_present, cmip6_future)]
# cmip6_changes_members = xr.concat(cmip6_changes, dim="member")

fa_changes_range = get_CI_90(fa_changes_members, n=15)  
ctl_changes_range = get_CI_90(ctl_changes_members, n=15)  
# cmip6_changes_range = ZGI_helpers.get_CI(cmip6_changes_members, conf_level = 0.90)


print(
    f"FA Change%: {fa_changes_members.mean(dim='member').item():.2f} "
    f"[{(fa_changes_members.mean(dim='member')-fa_changes_range).item():.2f}, "
    f"{(fa_changes_members.mean(dim='member')+fa_changes_range).item():.2f}]"
)

print(
    f"CTL Change%: {ctl_changes_members.mean(dim='member').item():.2f} "
    f"[{(ctl_changes_members.mean(dim='member')-ctl_changes_range).item():.2f}, "
    f"{(ctl_changes_members.mean(dim='member')+ctl_changes_range).item():.2f}]"
)

# print(
#     f"CMIP6 Change%: {cmip6_changes_members.mean(dim='member').item():.2f} "
#     f"[{(cmip6_changes_members.mean(dim='member')-cmip6_changes_range).item():.2f}, "
#     f"{(cmip6_changes_members.mean(dim='member')+cmip6_changes_range).item():.2f}]"
# )




####################################################
#### Part three : Plot
####################################################




colors = {
    'fa': 'blue',         # blue  
    'fa_range': '#5b9bd5',  # Light blue  
    'ctl': 'red',        # red 
    'ctl_range': '#e06c6c', # Light red
    'era5':'black',
    'cmip6': '#5c3324',      # marrone
    'hadisst': '#FBC02D',    # giallo
    'ersstv5': '#FF9800',    # arancione
    'cobe2': '#43A047'       # Orange  
}





fig = plt.figure(figsize=(15, 12))

gs = GridSpec(
    nrows=2,
    ncols=1,
    height_ratios=[1, 1],
    hspace=0.25
)

ax_zgi    = fig.add_subplot(gs[0])
ax_diff   = fig.add_subplot(gs[1])

##########################################
# PLOT ZGI
##########################################

ax_zgi.fill_between(ctl_mean_smooth.year, ctl_perc_5, ctl_perc_95, color=colors["ctl_range"],alpha=0.3, label= "IPSL (5th-95th perc.)")

ax_zgi.plot(ctl_mean_smooth.year, ctl_mean_smooth, color=colors["ctl"], linewidth=1, label="IPSL")
ax_zgi.plot(ctl_mean_smooth.year, ctl_mean_sig, color=colors["ctl"], linewidth=3)  

ax_zgi.fill_between(fa_mean_smooth.year, fa_perc_5, fa_perc_95, color=colors["fa_range"],alpha=0.3, label= "IPSL-FA (5th-95th perc.)")

ax_zgi.plot(fa_mean_smooth.year, fa_mean_smooth, color=colors["fa"], linewidth=1, label="IPSL-FA")
ax_zgi.plot(fa_mean_smooth.year, fa_mean_sig, color=colors["fa"], linewidth=3)  


ax_zgi.plot(zgi_era5_anom.year, zgi_era5_anom, color=colors["era5"], linewidth=2, label="ERA5")
ax_zgi.plot(zgi_hadisst_anom.year, zgi_hadisst_anom, color=colors["hadisst"], linewidth=2, label="HadISST")
ax_zgi.plot(zgi_ersstv5_anom.year, zgi_ersstv5_anom, color=colors["ersstv5"], linewidth=2, label="ERSSTv5")
ax_zgi.plot(zgi_cobe2_anom.year, zgi_cobe2_anom, color=colors["cobe2"], linewidth=2, label="COBE2")

handles, labels = ax_zgi.get_legend_handles_labels()
legend_dict = dict(zip(labels, handles))

ax_zgi.axhline(0, color="gray", linestyle="--", linewidth=1)
ax_zgi.set_title("a)", fontsize=16, loc='left')
ax_zgi.set_ylabel(r"$\Delta$ ZG (°C)", fontsize=14)
ax_zgi.grid(True, linestyle=":")
ax_zgi.tick_params(labelsize=14) 

order = ["HadISST", "ERSSTv5", "COBE2", "ERA5", "IPSL", "IPSL-FA", "IPSL (5th-95th perc.)", "IPSL-FA (5th-95th perc.)"]
ordered_handles = [legend_dict[l] for l in order if l in legend_dict]
ordered_labels = [l for l in order if l in legend_dict]

ax_zgi.legend(ordered_handles, ordered_labels, fontsize=14, loc="lower left", ncol=2, frameon=True)
ax_zgi.set_xlim(1950, 2100)
ax_zgi.set_ylim(-0.72, 0.42)

################################## Differenze IPSL-FA − IPSL 

# WP
ax_diff.fill_between(
    diff_west_mean_smooth.year,
    perc_5_west,
    perc_95_west,
    color='#FB9F89',
    alpha=0.4,
    label = "IPSL-FA − IPSL : WEP (5th-95th perc.)"
)


ax_diff.plot(diff_west_mean_smooth.year, diff_west_mean_smooth, color='#754F5B', lw=1, label="IPSL-FA − IPSL : WEP")
ax_diff.plot(diff_west_mean_smooth.year, west_mean_sig, color='#754F5B', lw=3)  # spessa dove significativo


# EP 
ax_diff.fill_between(
    diff_east_mean_smooth.year,
    perc_5_east,
    perc_95_east,
    color='#A3AF9C',
    alpha=0.4,
    label = "−(IPSL-FA − IPSL): EEP (5th-95th perc.)"
)

ax_diff.plot(diff_east_mean_smooth.year, diff_east_mean_smooth, color='#21A179', lw=1, label="−(IPSL-FA − IPSL) : EEP")
ax_diff.plot(diff_east_mean_smooth.year, east_mean_sig, color='#21A179', lw=3)  # spessa dove significativo


ax_diff.axhline(0, color="gray", linestyle="--", linewidth=1)


handles, labels = ax_diff.get_legend_handles_labels()
legend_dict = dict(zip(labels, handles))

ax_diff.set_title("b)", fontsize=16, loc='left')
ax_diff.set_xlabel("Year", fontsize=14)
ax_diff.set_ylabel(r"$\Delta$ RSST (°C)", fontsize=14)
ax_diff.tick_params(labelsize=14)

order = ["IPSL-FA − IPSL : WEP", "IPSL-FA − IPSL : WEP (5th-95th perc.)",
         "−(IPSL-FA − IPSL) : EEP", "−(IPSL-FA − IPSL): EEP (5th-95th perc.)"]

ordered_handles = [legend_dict[l] for l in order if l in legend_dict]
ordered_labels  = [l for l in order if l in legend_dict]

ax_diff.legend( ordered_handles, ordered_labels, fontsize = 14, loc="lower left", ncol=2, frameon=True )

for ax in [ax_zgi, ax_diff]:
    ax.axvspan(2005, 2034, color='silver', alpha=0.3, lw=0, zorder = 0)
    ax.axvspan(2070, 2099, color='silver', alpha=0.3, lw=0, zorder = 0)

ax_diff.set_xlim(1950, 2100)  
# ax_diff.set_ylim(-0.5, 0.35)
ax_diff.set_ylim(-0.72, 0.42)

ax_diff.grid(True, linestyle=":")

for ax in [ax_zgi, ax_diff]:
    ymin = ax.get_ylim()[0]
    ax.plot([1980, 2024], [ymin, ymin], color='black', linewidth=4,
            solid_capstyle='butt', clip_on=False, zorder=5)
    
# plt.savefig("/home/astoppel/figure/ZGI/ZGI_diff_EP_WP.pdf", bbox_inches='tight')

plt.show()


