#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.


import xarray as xr
import numpy as np
import xesmf as xe
from scipy.stats import ttest_1samp
import scipy.stats as stats
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FormatStrFormatter




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

def compute_zgi_cmip(sst_list):

    zgi_list = []

    for sst in sst_list:

        def box_mean(ds, lat_min, lat_max, lon_min, lon_max):
            
            ds_box = ds.sel(lat=slice(lat_min, lat_max),
                            lon=slice(lon_min, lon_max))

            return ds_box.mean(dim=['lon', 'lat'])

        # Wills 21: box tropicale
        west = box_mean(sst, -5, 5, 110, 180)
        east = box_mean(sst, -5, 5, 180, 280) 


        zgi_list.append(west - east)

    return zgi_list


def rolling_mean(yearly_da, window=31):

    
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




#### Upload models and observations

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

#CMIP6
base_cmip6 = xr.open_dataset("/data/astoppel/CMIP6/ensamble_historical/CMIP6_tos_1950_2099_models.nc")
cmip6_list = [base_cmip6['tos'].sel(model=m)for m in base_cmip6.model.values]




#### Calcola media annuale  

era5_ann = annual_mean_time_counter([era5_all['tos']]) 
hadisst_ann = annual_mean_time_counter([hadisst['tos']]) 
ersstv5_ann = annual_mean_time_counter([ersstv5['sst']]) 
cobe2_ann = annual_mean_time_counter([cobe2['sst']])  




#### Calcola ZGI
zgi_era5, west_era5, east_era5 = compute_zgi(era5_ann, area_nemo)
zgi_hadisst, west_hadisst, east_hadisst  = compute_zgi(hadisst_ann, area_nemo)
zgi_ersstv5, west_ersstv5, east_ersstv5  = compute_zgi(ersstv5_ann, area_nemo)
zgi_cobe2, west_cobe2, east_cobe2  = compute_zgi(cobe2_ann, area_nemo)

zgi_cmip6 = compute_zgi_cmip(cmip6_list) #regular grid for cmip6 models 
zgi_cmip6 = [zgi_cmip6[m].rename({'time': 'year'}) for m in range(len(base_cmip6.model))]
zgi_cmip6  = xr.concat(zgi_cmip6, dim="member")
zgi_cmip6["member"]= [i for i in base_cmip6.model.values]




#### smoothing timeseries 

smoothed_zgi_era5 = rolling_mean(zgi_era5[0], window=31)
smoothed_zgi_hadisst = rolling_mean(zgi_hadisst[0], window=31)
smoothed_zgi_ersstv5 = rolling_mean(zgi_ersstv5[0], window=31)
smoothed_zgi_cobe2 = rolling_mean(zgi_cobe2[0], window=31)


smoothed_members_cmip6  = []
for member in zgi_cmip6.member.values:
    da_member = zgi_cmip6.sel(member=member)
    smoothed = rolling_mean(da_member, window=31)
    smoothed = smoothed.assign_coords(member=member)
    smoothed_members_cmip6.append(smoothed)
zgi_cmip6_smooth = xr.concat(smoothed_members_cmip6, dim="member")




#### change with respect to reference period 

baseline_period = slice("1950", "1979")

#obs
baseline_zgi_era5  = zgi_era5[0].sel(year=baseline_period).mean("year")
zgi_era5_anom  = smoothed_zgi_era5 - baseline_zgi_era5

baseline_zgi_hadisst  = zgi_hadisst[0].sel(year=baseline_period).mean("year")
zgi_hadisst_anom  = smoothed_zgi_hadisst - baseline_zgi_hadisst

baseline_zgi_ersstv5  = zgi_ersstv5[0].sel(year=baseline_period).mean("year")
zgi_ersstv5_anom  = smoothed_zgi_ersstv5 - baseline_zgi_ersstv5

baseline_zgi_cobe2  = zgi_cobe2[0].sel(year=baseline_period).mean("year")
zgi_cobe2_anom  = smoothed_zgi_cobe2 - baseline_zgi_cobe2

#cmip6
baseline_cmip6  = zgi_cmip6.sel(year=baseline_period).mean("year")
zgi_cmip6_anom_smooth  = zgi_cmip6_smooth - baseline_cmip6




#CMIP6  CI 90% 
cmip6_mean = zgi_cmip6_anom_smooth.mean("member")
cmip6_ci = get_CI_90(zgi_cmip6_anom_smooth, n=41)
cmip6_mean_sig = cmip6_mean.where(np.abs(cmip6_mean) > cmip6_ci)

#CMIP6 5-95 PERCENTILE 
zgi_cmip6_min = np.percentile(zgi_cmip6_anom_smooth, 5, 0)
zgi_cmip6_max = np.percentile(zgi_cmip6_anom_smooth, 95, 0)




####################################################
#### Part two: for the maps
####################################################




#### Functions

def compute_trend(data):

    ny = data.sizes['lat']
    nx = data.sizes['lon']
    
    trends_member = np.full((ny, nx), np.nan)

    # for each grid point i compute the trend
    
    for i, lat in enumerate(data['lat']):
        for j, lon in enumerate(data['lon']):
            yvals = data.sel({'lat': lat, 'lon': lon}).values
            if np.isfinite(yvals).sum() > 1:
                coeffs = np.polyfit(data['year'].values[np.isfinite(yvals)],
                                    yvals[np.isfinite(yvals)], 1)
                trends_member[i, j] = coeffs[0]
                
    trends_da = xr.DataArray(
        np.array(trends_member),
        dims=['lat', 'lon'],
        coords={'lat': data.lat,
                'lon': data.lon}
    )

    return trends_da



def compute_trend_per_member(data_list):

    trends_all = []

    for m, member_da in enumerate(data_list):
        
        time_dim, lat_dim, lon_dim = 'year', 'nav_lat', 'nav_lon'
        
        ny, nx =  member_da.sizes['y'],  member_da.sizes['x']   

        trends_member = np.full((ny, nx), np.nan)

        # for each grid point i compute the trend
        
        for i, y_val in enumerate(member_da['y']):
            for j, x_val in enumerate(member_da['x']):
                yvals = member_da.sel({'y': y_val, 'x': x_val}).values
                if np.isfinite(yvals).sum() > 1:
                    coeffs = np.polyfit(member_da[time_dim].values[np.isfinite(yvals)],
                                        yvals[np.isfinite(yvals)], 1)
                    trends_member[i, j] = coeffs[0]
                    
        trends_all.append(trends_member)

    trends_da = xr.DataArray(
        np.array(trends_all),
        dims=['member', 'y', 'x'],
        coords={
            'member': range(len(data_list)),
            'y': data_list[0]['y'],
            'x': data_list[0]['x'],
            'nav_lon': data_list[0].nav_lon,
            'nav_lat': data_list[0].nav_lat
        })

    return trends_da


def filter_outliers(data, lower_percentile=2, upper_percentile=98):

    lower_bound = np.nanpercentile(data, lower_percentile)
    upper_bound = np.nanpercentile(data, upper_percentile)
    
    return data.where( (data <= upper_bound) & (data >= lower_bound) )
    

def ensemble_ttest_members(trend_da, alpha=0.1,varname=None):
    """
    One-sample t-test on ensemble members.
    To test if the trend is diff from zero.
    """
            
    trend = trend_da[varname]


    t_stat, p_val = ttest_1samp(
        trend.values,
        popmean=0.0,
        axis=0,
        nan_policy="omit")

    spatial_dims = [d for d in trend_da.dims if d != "member"]

    p_values = xr.DataArray(
        p_val,
        coords={d: trend_da.coords[d] for d in spatial_dims},
        dims=spatial_dims,
        name="p_value"
    )

    significant_mask = p_values < alpha

    return p_values, significant_mask
    




#### ERA5 RSST trend 

era5_sel = era5_all.sel(time_counter=slice('1980-01-01','2024-12-31'))

regridder = xe.Regridder(era5_sel, grid_regular, method="bilinear", reuse_weights=False, ignore_degenerate=True)

era5_sel_reg = regridder(era5_sel)

era5_ann_reg = era5_sel_reg.groupby('time_counter.year').mean(dim='time_counter')


era5_trop = (era5_ann_reg['tos'].lat >= -30) & (era5_ann_reg['tos'].lat <= 30)

era_5_trop_mean = (era5_ann_reg['tos']*0 + era5_ann_reg['tos'].where(era5_trop == 1)).mean(dim=['lat', 'lon'], skipna=True)

trend_era5 = compute_trend(era5_ann_reg['tos'] - era_5_trop_mean)

trend_era5_dec = trend_era5 * 10 #dec trend




#### ERA5 wind stress trend 

era5_wind_x = xr.open_dataset("/data/astoppel/ERA5/ERA5_1m_ewss_194001_202412.nc")
era5_wind_y = xr.open_dataset("/data/astoppel/ERA5/ERA5_1m_nsss_194001_202412.nc")

era5_x = era5_wind_x['ewss'].sel(time=slice('1980-01-01','2024-12-31'))
era5_y = era5_wind_y['nsss'].sel(time=slice('1980-01-01','2024-12-31'))

era5_x_ann = era5_x.groupby('time.year').mean(dim='time') 
era5_y_ann = era5_y.groupby('time.year').mean(dim='time') 

regridder = xe.Regridder(era5_x_ann, grid_regular, method="bilinear", reuse_weights=False, ignore_degenerate=True)

era5_wind_reg_x = regridder(era5_x_ann)
era5_wind_reg_y = regridder(era5_y_ann)

trend_era5_taux = compute_trend(era5_wind_reg_x)
trend_era5_tauy = compute_trend(era5_wind_reg_y)

trend_era5_taux = trend_era5_taux * 10 #dec trend
trend_era5_tauy = trend_era5_tauy * 10 #dec trend

trend_era5_taux_filt = filter_outliers(trend_era5_taux)
trend_era5_tauy_filt = filter_outliers(trend_era5_tauy)




#### CMIP6 RSST trend 

# cmip6_list_sel = [cmip6_list[m].sel(time=slice('1980-01-01','2024-12-31')).rename({'time': 'year'}) for m in range(len(base_cmip6.model))]
# trend_cmip6 = compute_trends_per_member(cmip6_list_sel) 
# trend_cmip6.to_dataset(name='trend_1980_2024_cmip6')

cmip6_trop = (trend_cmip6.lat >= -30) & (trend_cmip6.lat <= 30)

cmip6_rsst_trend_pac = (trend_cmip6['trend_1980_2024_cmip6'] * 0 + 
                        trend_cmip6['trend_1980_2024_cmip6'].where(cmip6_trop == 1)).mean(dim=['lat', 'lon'], skipna=True)

cmip6_rsst_trend = trend_cmip6 - cmip6_rsst_trend_pac

_, sig_mask_cmip6 =  ensemble_ttest_members(cmip6_rsst_trend, alpha=0.05, varname='trend_1980_2024_cmip6')
trend_cmip6_mmm = cmip6_rsst_trend['trend_1980_2024_cmip6'].mean("member")
trend_cmip6_mmm_dec = trend_cmip6_mmm * 10

step = 7  
sig_sparse = np.zeros_like(sig_mask_cmip6, dtype=bool)
sig_sparse[::step, ::step] = sig_mask_cmip6[::step, ::step]




####################################################
#### Part three : Plot
####################################################




colors = {
    'era5':'black',
    'cmip6': '#5c3324',      # marrone
    'hadisst': '#FBC02D',    # giallo
    'ersstv5': '#FF9800',    # arancione
    'cobe2': '#43A047'       # Orange  
}




diff_levels = np.linspace(-0.4, 0.4, 17)
cmap = "RdBu_r"

def plot_trend_map(
    ax,
    sst_trend,
    sig_mask,
    title,
    y_title = None,
    tauu=None, tauv=None,
    wind_scale=None,
    lat_range=(-30, 30),
    lon_range=None,
    rectangles=None,
    sig_color="black",
    sig_alpha=0.4,
    sig_linewidth=0.8,
    sampling_step=6):
    
    # map extension
    
    lon_min = lon_range[0]  
    lon_max = lon_range[1]  
    lat_min, lat_max = lat_range
    
    ax.set_extent([lon_min, lon_max, lat_min, lat_max],crs=ccrs.PlateCarree())

    lon = sst_trend.lon
    lat = sst_trend.lat

    # Contourf

    cf = ax.contourf(
        lon, lat, sst_trend,
        levels=diff_levels,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        rasterized=True)

    cs = ax.contour(
        lon, lat, sst_trend,
        levels=diff_levels,
        colors="black",
        linewidths=0.2,
        transform=ccrs.PlateCarree()
    )
    for c in cs.collections:
        c.set_rasterized(True)

    # Significativity (hatching)

    if sig_mask is not None and sig_mask.any():
        sig_data = np.zeros_like(sst_trend)
        sig_data[sig_mask] = 1

        contourf = ax.contourf(
            lon, lat, sig_data,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["."], #////
            transform=ccrs.PlateCarree()
        )

        for coll in contourf.collections:
            coll.set_edgecolor(sig_color)
            coll.set_linewidth(sig_linewidth)
            coll.set_alpha(sig_alpha)
            coll.set_facecolor("none")

    # wind stress trend 
    q = None
    if tauu is not None and tauv is not None:
        step = sampling_step
        step_x = 8
        q = ax.quiver(
            lon[::step_x], lat[::step],
            tauu.values[::step, ::step_x], tauv.values[::step, ::step_x],
            transform=ccrs.PlateCarree(),
            color='black',
            scale=wind_scale,
            width=0.002,         
            pivot='tail',
            zorder=1
        )        

    # Boxes
    
    if rectangles is not None:
        for rect in rectangles:
            lon0, lon1, lat0, lat1 = rect
            ax.plot(
                [lon0, lon1, lon1, lon0, lon0],
                [lat0, lat0, lat1, lat1, lat0],
                color="black",
                linewidth=3,
                transform=ccrs.PlateCarree()
            )


    ax.add_feature(cfeature.LAND, facecolor="white", zorder=2)
    ax.add_feature(cfeature.COASTLINE, zorder=3)

    ax.set_title(title, y = y_title, loc='left', fontsize=16)

    return cf, q




fig = plt.figure(figsize=(8, 14))

gs = GridSpec(nrows=4, ncols=1, height_ratios=[1, 1, 1, 0.1], wspace=0.05, hspace=0.1)

ax_zgi   = fig.add_subplot(gs[0])  
ax_era5  = fig.add_subplot(gs[1], projection=ccrs.PlateCarree(central_longitude=180))
ax_cmip6 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree(central_longitude=180))
ax_cbar  = fig.add_subplot(gs[3])

# PLOT ZGI

ax_zgi.fill_between(
    zgi_cmip6_anom_smooth.year,
    zgi_cmip6_min,
    zgi_cmip6_max,
    color=colors["cmip6"],
    alpha=0.2,
    label="CMIP6 (5th-95th perc.)")

ax_zgi.plot(zgi_cmip6_anom_smooth.year, cmip6_mean , color=colors["cmip6"], linewidth=1, label="CMIP6")
ax_zgi.plot(zgi_cmip6_anom_smooth.year, cmip6_mean_sig, color=colors["cmip6"], linewidth=3)  


ax_zgi.plot(zgi_era5_anom.year, zgi_era5_anom,
            color=colors["era5"], linewidth=2, label="ERA5")

ax_zgi.plot(zgi_hadisst_anom.year, zgi_hadisst_anom,
            color=colors["hadisst"], linewidth=2, label="HadISST")

ax_zgi.plot(zgi_ersstv5_anom.year, zgi_ersstv5_anom,
            color=colors["ersstv5"], linewidth=2, label="ERSSTv5")

ax_zgi.plot(zgi_cobe2_anom.year, zgi_cobe2_anom,
            color=colors["cobe2"], linewidth=2, label="COBE2")

handles, labels = ax_zgi.get_legend_handles_labels()

legend_dict = dict(zip(labels, handles))

ax_zgi.axhline(0, color="gray", linestyle="--", linewidth=1)
ax_zgi.set_title("a) Zonal SST gradient changes / 1950–1979", y=1.03, loc='left', fontsize=16)
ax_zgi.set_xlabel("Year", fontsize=14)
ax_zgi.set_ylabel("Δ ZGI (°C)", fontsize=14)
ax_zgi.grid(True, linestyle=":")
ax_zgi.tick_params(labelsize=14)

order = ["HadISST", "ERSSTv5", "COBE2", "ERA5", "CMIP6", "CMIP6 (5th-95th perc.)"]  

ordered_handles = [legend_dict[l] for l in order if l in legend_dict]
ordered_labels  = [l for l in order if l in legend_dict]

ax_zgi.legend( ordered_handles, ordered_labels, fontsize=14, loc="lower left", ncol=2, frameon=True)

ax_zgi.axvspan(1980, 2024, color='silver', alpha=0.3, lw=0, zorder=0)

ax_zgi.set_xlim(1950, 2100)
ax_zgi.set_ylim(-0.75, 0.4)

# Map ERA5

rectangles = [
    (110, 180, -5, 5),   # WEP
    (180, 280, -5, 5)     # EEP
]


cf1, q1 = plot_trend_map(
    ax=ax_era5,
    sst_trend=trend_era5_dec,
    lon_range=(100, 290),
    sig_mask=None,
    title="b) ERA5 RSST and wind stress trend (1980-2024)",
    y_title=1.10,
    wind_scale=0.1,   
    tauu=trend_era5_taux_filt,
    tauv=trend_era5_tauy_filt,
    rectangles=rectangles)

ref_value = 0.0025  

ax_era5.quiverkey(
    q1,
    X=0.69,
    Y=1.07,
    U=ref_value,
    label=rf"{ref_value:.1e} $N\,m^{{-2}}\,dec^{{-1}}$",
    labelpos='E',
    coordinates='axes',
    fontproperties={'size': 14}  )

# Map CMIP6

cf2, q = plot_trend_map(
    ax_cmip6,
    trend_cmip6_mmm_dec,
    sig_sparse,
    tauu=None, tauv=None,
    lon_range=(100, 290),
    title="c) CMIP6 RSST and wind stress trend (1980-2024)",
    y_title=1.10,
    rectangles=rectangles
)

# Gridlines
for ax, gl_ax in [(ax_era5, ax_era5), (ax_cmip6, ax_cmip6)]:
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}


plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.08)

pos = ax_cmip6.get_position()
ax_cmip6.set_position([pos.x0, pos.y0 + 0.06, pos.width, pos.height])


# Colorbar 

cbar = fig.colorbar(
    cf2, cax=ax_cbar, orientation="horizontal")

cbar.set_label("°C/dec", fontsize=14)
cbar.ax.tick_params(labelsize=14)
cbar.formatter = FormatStrFormatter('%.2g')
cbar.update_ticks()
pos_cb = ax_cbar.get_position()
ax_cbar.set_position([pos_cb.x0, pos_cb.y0 + 0.1, pos_cb.width, pos_cb.height])


# plt.savefig("/home/astoppel/figure/ZGI/ZGI_trend_obs_cmip6_rsst_newboxes.pdf", bbox_inches='tight') 

plt.show()

