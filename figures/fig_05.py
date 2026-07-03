#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.



import xarray as xr
import xesmf as xe
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import ttest_ind, t, ttest_1samp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats





#### Grids

file_nemo = xr.open_dataset('/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-r4_merged_grid_T.nc').sel(time_counter=slice('1950-01-01', '1950-02-01'))
area_nemo = file_nemo['cell_area']

file_regular = xr.open_dataset('/data/astoppel/CMIP6/ensamble_historical/ipsl_CMIP6_TOS_1921_2014.nc')
grid_regular = {"lon": file_regular["lon"], "lat": file_regular["lat"]}




regioni_rsst = {
    'EP': {  # Eastern Equatorial Pacific
        'lat_range': (-5, 5),
        'lon_range': (180, 280)  
    },
    'WP': {  # Western Pacific  
        'lat_range': (-5, 5),
        'lon_range': (110, 180)   }}




####################################################
#### Part one : ERA5, IPSL, IPSL-FA RSST trend (map)+ EAST+WEST+ZG 
####################################################




#### Functions

def compute_regional_trend(rsst_era5, lat_range, lon_range):
    """Compute trend e stderr on region (°C/decade)."""
    lat_mask = (rsst_era5.lat >= lat_range[0]) & (rsst_era5.lat <= lat_range[1])
    lon_mask = (rsst_era5.lon >= lon_range[0]) & (rsst_era5.lon <= lon_range[1])
    
    series   = rsst_era5.where(lat_mask & lon_mask, drop=True).mean(dim=['lat', 'lon'])
    years    = series['year'].values
    yvals    = series.values
    slope, _, _, _, stderr = stats.linregress(years, yvals)
    return slope * 10, stderr * 10  # °C/decade


def compute_zg_trend(da):
    
    """Compute trend and stderr del ZG = WEP - EEP (°C/decade)."""
    
    eep_series = da.where(
        (da.lat >= -5) & (da.lat <= 5) & (da.lon >= 180) & (da.lon <= 280), drop=True).mean(dim=['lat', 'lon'])
    
    wep_series = da.where(
        (da.lat >= -5) & (da.lat <= 5) & (da.lon >= 110) & (da.lon <= 180), drop=True).mean(dim=['lat', 'lon'])
    
    zg_series = wep_series - eep_series
    
    slope, _, _, _, stderr = stats.linregress(zg_series['year'].values, zg_series.values)
    
    return slope * 10, stderr * 10  # °C/decade

def load_da(filename, newname):
    ds = xr.open_dataset(path + filename)
    da = ds["__xarray_dataarray_variable__"]
    return da.rename(newname)

def ensemble_ttest_members(trend_da, alpha=0.1):
    """
    One-sample t-test on ensemble members.
    H0: to test that the trend is different from zero
    """
    

    trend = trend_da
    
    
    t_stat, p_val = ttest_1samp(
        trend.values,
        popmean=0.0,
        axis=0,
        nan_policy="omit"
    )

    spatial_dims = [d for d in trend_da.dims if d != "member"]

    p_values = xr.DataArray(
        p_val,
        coords={d: trend_da.coords[d] for d in spatial_dims},
        dims=spatial_dims,
        name="p_value"
    )

    significant_mask = p_values < alpha

    return p_values, significant_mask




# ERA5 

era5 = xr.open_dataset("/data/astoppel/ERA5/ERA5_tos_1949_2019_gn.nc").sel(time_counter=slice("1950-01-01", "2019-12-31"))
era5_last = xr.open_dataset("/data/astoppel/ERA5/ERA5_tos_2020_2024_gn_Kelvin.nc") 
era5_last['tos'] = era5_last['tos'] - 273.15 
era5_all = xr.concat([era5, era5_last], dim="time_counter")
era5_sel = era5_all.sel(time_counter=slice('1980-01-01','2024-12-31'))

regridder = xe.Regridder(era5_sel, grid_regular, method="bilinear", reuse_weights=False, ignore_degenerate=True)
era5_sel_reg= regridder(era5_sel)

era5_ann = era5_sel_reg.groupby('time_counter.year').mean(dim='time_counter')

era5_trop_m  = era5_ann.sel(lat=slice(-30, 30)).mean(dim=['lat', 'lon'])
rsst_era5 = era5_ann - era5_trop_m

trend_era5 = trends_helpers.compute_trend(rsst_era5['tos']) #for map 


era5_eep_trend, era5_eep_stderr = compute_regional_trend(rsst_era5['tos'],    lat_range=(-5, 5), lon_range=(180, 280))
era5_wep_trend, era5_wep_stderr = compute_regional_trend(rsst_era5['tos'],    lat_range=(-5, 5), lon_range=(110, 180))
era5_zg_trend,  era5_zg_stderr  = compute_zg_trend(rsst_era5['tos'])





members = ['1','2','3','4','12','13','14','15','16','18','22','23','29','30','33'] 

path = "/home/astoppel/project_flux_adjustment/data/computation_1980_2024/"

# RSST IPSL
#1) members
trend_rsst_ipsl_m = load_da("trend_rsst_ipsl_members.nc", "trend_rsst_ipsl_m")
trend_rsst_ipsl_m = trend_rsst_ipsl_m*10

trend_rsst_ipsl_dict = {}
for i, m in enumerate(trend_rsst_ipsl_m.member):
    nome_membro = members[i]
    trend_rsst_ipsl_dict[nome_membro] = trend_rsst_ipsl_m.sel(member=m)

#2) mean
trend_rsst_ipsl_mean = load_da("trend_rsst_ipsl_mean.nc", "trend_rsst_ipsl_mean")
trend_rsst_ipsl_mean = trend_rsst_ipsl_mean*10


# RSST FA
#1) members
trend_rsst_FA_m = load_da("trend_rsst_FA_members.nc", "trend_rsst_FA_m")
trend_rsst_FA_m = trend_rsst_FA_m*10

trend_rsst_FA_dict = {}
for i, m in enumerate(trend_rsst_FA_m.member):
    nome_membro = members[i]
    trend_rsst_FA_dict[nome_membro] = trend_rsst_FA_m.sel(member=m)
    
#2) mean
trend_rsst_FA_mean = load_da("trend_rsst_FA_mean.nc", "trend_rsst_FA_mean")
trend_rsst_FA_mean = trend_rsst_FA_mean*10


# WIND STRESS FA: mean
tauu_fa_mean = load_da("trend_tauu_FA_mean.nc", "tauu_fa_mean")
tauv_fa_mean = load_da("trend_tauv_FA_mean.nc", "tauv_fa_mean")

# WIND STRESS IPSL: mean
tauu_ipsl_mean = load_da("trend_tauu_IPSL_mean.nc", "tauu_ipsl_mean")
tauv_ipsl_mean = load_da("trend_tauv_IPSL_mean.nc", "tauv_ipsl_mean")




#significant test for ipsl and ipsl-fa

_, sig_mask_ipsl  =  ensemble_ttest_members(trend_rsst_ipsl_m, alpha=0.10)

_, sig_mask_ipslfa =  ensemble_ttest_members(trend_rsst_FA_m, alpha=0.10)

step = 8
sig_sparse_ipsl_fa = np.zeros_like(sig_mask_ipslfa_regridded, dtype=bool)
sig_sparse_ipsl_fa[::step, ::step] = sig_mask_ipslfa_regridded[::step, ::step]

sig_sparse_ipsl = np.zeros_like(sig_mask_ipsl_regridded, dtype=bool)
sig_sparse_ipsl[::step, ::step] = sig_mask_ipsl_regridded[::step, ::step]





####################################################
#### Part two : regridd the trends
####################################################




#### Functions

def regrid_trend_data(trend_data, grid_target, method="bilinear"):
    """
    Regridda i dati trend sulla griglia target
    """
    trend_ds = trend_data.to_dataset(name='trend')
    
    regridder = xe.Regridder(
        trend_ds, grid_target,
        method=method,
        reuse_weights=False,
        ignore_degenerate=True
    )
    
    trend_regridded = regridder(trend_ds)['trend']
    return trend_regridded

def regrid_significance_mask(significance_mask, grid_target, method="nearest_s2d"):
    """
    Regridda una maschera di significatività sulla griglia target
    """
    # Converti booleano in float per xesmf
    mask_float = significance_mask.astype(float)
    mask_ds = mask_float.to_dataset(name='significance')
    
    regridder = xe.Regridder(
        trend_rsst_ipsl_dict['1'], grid_target,
        method='nearest_s2d',   
        reuse_weights=False,
        ignore_degenerate=True
    )
    
    mask_regridded = regridder(mask_ds)['significance']
    mask_regridded_bool = mask_regridded > 0.5
    
    return mask_regridded_bool




trend_rsst_FA_regridded = regrid_trend_data(trend_rsst_FA_mean, grid_regular)
trend_rsst_ipsl_regridded = regrid_trend_data(trend_rsst_ipsl_mean, grid_regular)
sig_mask_ipsl_regridded = regrid_significance_mask(sig_mask_ipsl, grid_regular)
sig_mask_ipslfa_regridded = regrid_significance_mask(sig_mask_ipslfa, grid_regular)




####################################################
#### Part three: IPSL, IPSL-FA RSST trend for ZG-boxplot
####################################################




#### Functions

def merge_historical_future(hist_list, fut_list, var_name, time_dim='time_counter'):
    merged = []
    for h_ds, f_ds in zip(hist_list, fut_list):

        h_ds = h_ds.rename({time_dim: 'time_counter'}) if time_dim != 'time_counter' else h_ds
        f_ds = f_ds.rename({time_dim: 'time_counter'}) if time_dim != 'time_counter' else f_ds
        
        hist = h_ds[var_name].sel(time_counter=slice('1950-01-01', '2014-12-31'))
        fut = f_ds[var_name].sel(time_counter=slice('2015-01-01', '2099-12-31'))
        
        merged.append(xr.concat([hist, fut], dim='time_counter'))
        
    return merged


def annual_mean_time_counter(ds_list):

    return [ds.groupby('time_counter.year').mean(dim='time_counter') for ds in ds_list]


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

def compute_rsst(da):
    
    trop_mask = (da.nav_lat >= -30) & (da.nav_lat <= 30)
    trop_mean = da.where(trop_mask).weighted(area_nemo.where(trop_mask).fillna(0.)).mean(dim=['y', 'x'])
    
    return da - trop_mean  




members = ['r1', 'r2', 'r3', 'r4', 'r14', 'r33', 'r12', 'r13', 'r15', 'r16', 'r18', 'r22', 'r23', 'r29', 'r30'] 

#compute the ipsl-fa ZG 

base_fa = "/scratchu/astoppel/IPSL_FA/"

hist_fa = load_ensemble(base_fa, "historical", 
                       "historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-{member}_merged_grid_T.nc", members)

fut_fa = load_ensemble(base_fa, "ssp585", 
                      "ssp585/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-ssp585-{member}_merged_grid_T.nc", members)

tos_fa = merge_historical_future(hist_fa, fut_fa, 'tos')
tos_fa_ann = annual_mean_time_counter(tos_fa)
tos_fa_ann_sel  = [da.sel(year=slice(1980, 2024)) for da in tos_fa_ann]
rsst_fa_ann  = [compute_rsst(da) for da in tos_fa_ann_sel]


#compute the ipsl ZG 

base_ctl ="/scratchu/astoppel/IPSL_CTL/"

hist_ctl = load_ensemble(base_ctl, "historical",
                        "historical/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_historical_{member}_gn_1950-2014.nc",
                        members)


fut_ctl = load_ensemble(base_ctl, "ssp585",
                        "ssp585/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_ssp585_{member}_gn_2015_2099.nc",
                        members)

tos_ctl = merge_historical_future(hist_ctl, fut_ctl, 'tos', 'time') #merge histo and future
tos_ctl_ann = annual_mean_time_counter(tos_ctl) #do the annual mean
tos_ctl_ann_sel = [da.sel(year=slice(1980, 2024)) for da in tos_ctl_ann] #select the period
rsst_ctl_ann = [compute_rsst(da) for da in tos_ctl_ann_sel] #compute rsst
rsst_ctl_ann_reg = [regridder(da) for da in rsst_ctl_ann] #regrid to regular grid




EEP = dict(lat_range=(-5, 5), lon_range=(180, 280))
WEP = dict(lat_range=(-5, 5), lon_range=(110, 180))

# Regridding dopo RSST
rsst_fa_ann_reg  = [regridder(da) for da in rsst_fa_ann]

# Trend per regione per membro
ipsl_eep  = [compute_regional_trend(da, **EEP)[0] for da in rsst_ctl_ann_reg]
ipsl_wep  = [compute_regional_trend(da, **WEP)[0] for da in rsst_ctl_ann_reg]
fa_eep    = [compute_regional_trend(da, **EEP)[0] for da in rsst_fa_ann_reg]
fa_wep    = [compute_regional_trend(da, **WEP)[0] for da in rsst_fa_ann_reg]

diff_ipsl = [compute_zg_trend(da)[0] for da in rsst_ctl_ann_reg]
diff_fa   = [compute_zg_trend(da)[0] for da in rsst_fa_ann_reg]

box_data = [ipsl_eep, fa_eep, ipsl_wep, fa_wep, diff_ipsl, diff_fa]




####################################################
#### Part four: Plot
####################################################




diff_levels = np.linspace(-0.2,0.2,9)

cmap = "RdBu_r"
def plot_trend_diff(ax, sst_trend, sig_mask_regridded, title, tauu=None, tauv=None, wind_scale=1, 
                    sig_color='black', sig_alpha=0.4, sig_linewidth=0.8, sampling_step=6,
                    lat_range=None, lon_range=None, global_map= False):  

    # Extension Map
    
    if global_map:
        ax.set_global()
    elif lat_range is not None or lon_range is not None:
        lon_min = lon_range[0]  
        lon_max = lon_range[1]  
        lat_min = lat_range[0]  
        lat_max = lat_range[1]  

        ax.set_extent(
            [lon_min, lon_max, lat_min, lat_max],
            crs=ccrs.PlateCarree())

    lon = sst_trend.lon
    lat = sst_trend.lat

    lon2d, lat2d = np.meshgrid(sst_trend.lon, sst_trend.lat)
    sig_positions = np.where(sig_mask_regridded)

    cf = ax.contourf(
        lon, lat, sst_trend,
        levels=diff_levels,
        cmap=cmap,
        extend='both',
        transform=ccrs.PlateCarree(),
        rasterized=True)

    cs = ax.contour(
        lon, lat, sst_trend,
        levels=diff_levels,
        colors='black',
        linewidths=0.3,
        transform=ccrs.PlateCarree())

    for col in cs.collections:
        col.set_rasterized(True)
        
    if len(sig_positions[0]) > 0:
        sig_data = np.zeros_like(sst_trend)
        sig_data[sig_positions] = 1
        
        contourf = ax.contourf(
            lon, lat, sig_data,
            levels=[0.5, 1.5],
            colors='none',  
            hatches=["."], 
            transform=ccrs.PlateCarree(),
            rasterized=True)
        
        for collection in contourf.collections:
            collection.set_edgecolor(sig_color)
            collection.set_linewidth(sig_linewidth)
            collection.set_alpha(sig_alpha)
            collection.set_facecolor('none')
            
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=1)

    ax.set_title(title, fontsize=20, loc ="left", y = 1.05)

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
        )
        return cf, q

    return cf, None




# COLOR SCHEME
colors   = [
    '#e06c6c','#5b9bd5',  #ipsl, Fa
    '#e06c6c','#5b9bd5',  
    '#e06c6c','#5b9bd5'  ]




t_crit = stats.t.ppf(0.95, df=43)  # 90% CI


fig = plt.figure(figsize=(10, 16))

gs = fig.add_gridspec(
    nrows=3,
    ncols=1,
    height_ratios=[1, 1, 1.4],
    hspace=0.0
)

ax_ipsl = fig.add_subplot(
    gs[0, 0],
    projection=ccrs.PlateCarree(central_longitude=180)
)
ax_fa = fig.add_subplot(
    gs[1, 0],
    projection=ccrs.PlateCarree(central_longitude=180)
)
ax_box_originali = fig.add_subplot(gs[2, 0])

# Maps IPSL / IPSL-FA

cf_ipsl, q1 = plot_trend_diff(
    ax_ipsl,
    trend_rsst_ipsl_regridded,
    sig_sparse_ipsl,
    r"a) IPSL: RSST, $\vec{\tau}$ trends ",
    tauu_ipsl_mean*10,
    tauv_ipsl_mean*10,
    wind_scale=0.05, #era 0.005,
    lat_range=(-30, 30),
    lon_range=(100, 290)
)

cf_fa, q2 = plot_trend_diff(
    ax_fa,
    trend_rsst_FA_regridded,
    sig_sparse_ipsl_fa,
    r"b) IPSL-FA: RSST, $\vec{\tau}$ trends ",
    tauu_fa_mean*10,
    tauv_fa_mean*10,
    wind_scale=0.05, #era 0.005,
    lat_range=(-30, 30),
    lon_range=(100, 290)
)

 
ref_value = 0.0025  

ax_fa.quiverkey(
    q2, X=0.65, Y=1.1, U=ref_value,
    label=rf"{ref_value:.1e} $N\,m^{{-2}}\,dec^{{-1}}$",
    labelpos='E',
    coordinates='axes',
    fontproperties={'size': 14}  
)

# Gridlines and boxes
for ax in [ax_ipsl, ax_fa]:
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

    for reg_name, reg in regioni_rsst.items():
        lon_min, lon_max = reg['lon_range']
        lat_min, lat_max = reg['lat_range']
        rect = mpatches.Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            linewidth=1.5,
            edgecolor='black',
            facecolor='none',
            transform=ccrs.PlateCarree(),
            zorder=5
        )
        ax.add_patch(rect)


fig.canvas.draw()  

pos0 = ax_ipsl.get_position()
pos1 = ax_fa.get_position()
pos2 = ax_box_originali.get_position()

gap_small = 0.08   # space between tra a e b  
gap_large = 0.12   # space between b e c  

# Panel a
new_y1 = pos0.y0 - pos0.height - gap_small
ax_fa.set_position([pos1.x0, new_y1, pos1.width, pos1.height])

# Panel b
new_y2 = new_y1 - pos2.height - gap_large
ax_box_originali.set_position([pos2.x0, new_y2, pos2.width, pos2.height])

# Colorbar 
pos_b = ax_fa.get_position()
cax_map = fig.add_axes([
    pos_b.x0,
    pos_b.y0 - 0.04,
    pos_b.width,
    0.018
])
cbar_map = fig.colorbar(cf_fa, cax=cax_map, orientation='horizontal')
cbar_map.set_label("RSST Trend (°C/dec)", fontsize=16)
cbar_map.ax.tick_params(labelsize=14)  # aggiungi questo
cbar_map.formatter = FormatStrFormatter('%.3g')
cbar_map.update_ticks()

# (c) BOXPLOT

positions_original = [1, 2, 3, 4]
positions_diff = [6, 7]
all_positions = positions_original + positions_diff

box_plot = ax_box_originali.boxplot(
    box_data,
    positions=all_positions,
    patch_artist=True,
    widths=0.45,
    medianprops={'visible': False},
    showmeans=True,
    meanline=True,
    meanprops={'color': 'black', 'linewidth': 1.2, 'linestyle': '--'}
)

for patch, color in zip(box_plot['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

for pos, data, color in zip(all_positions, box_data, colors):
    ax_box_originali.scatter(
        np.full_like(data, pos), data,
        color=color, alpha=0.6, s=30,
        edgecolors='white', linewidth=0.3, zorder=2
    )


# ERA5 with errorbar 

for x_pos, reg, trend, stderr, xmin, xmax, x_text, y_offset in [
    (1.5, 'EEP', era5_eep_trend, era5_eep_stderr, 0.75, 2.5,  2.0,  0.005),
    (3.5, 'WEP', era5_wep_trend, era5_wep_stderr, 2.5,  4.75, 4.5,  0.005),
    (6.5, 'ZG',  era5_zg_trend,  era5_zg_stderr,  5.25, 7.25, 5.5,  0.005),]:
    
    ax_box_originali.hlines(trend, xmin=xmin, xmax=xmax,
                            colors='black', linewidth=2, zorder=4)
    ax_box_originali.errorbar(
        x_pos, trend, yerr=[[t_crit * stderr], [t_crit * stderr]],
        fmt='none', color='black', capsize=5, linewidth=1.5, zorder=5
    )
    ax_box_originali.text(x_text, trend + y_offset, "ERA5",
                          ha='center', va='bottom', fontsize=16)
    
# STILE ASSI BOXPLOT

ax_box_originali.axvline(2.5, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax_box_originali.axvline(5.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax_box_originali.axhline(0,   color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

ax_box_originali.set_xticks([1, 2, 3, 4, 6, 7])
ax_box_originali.set_xticklabels(
    ["IPSL\n          EEP", "IPSL-FA", "IPSL\n          WEP", "IPSL-FA", "IPSL\n          ZG", "IPSL-FA"],
    fontsize=18
)
ax_box_originali.grid(True, alpha=0.2, axis='y')
ax_box_originali.tick_params(axis='y', labelsize=18)
ax_box_originali.text(
    0, 1.05, 'c) RSST trend (°C/dec)',
    transform=ax_box_originali.transAxes,
    fontsize=20
)

# plt.savefig("/home/astoppel/figure/trend/rsst_trend/ipsl_ipslfa_boxplot.pdf", bbox_inches='tight', facecolor='white')
plt.show()

