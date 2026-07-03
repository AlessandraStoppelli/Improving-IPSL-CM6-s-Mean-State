#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.



import xarray as xr
import numpy as np
import xesmf as xe
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FormatStrFormatter




#Grids & Masks

file_nemo = xr.open_dataset('/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-r4_merged_grid_T.nc').sel(time_counter=slice('1950-01-01', '1950-02-01'))
area_nemo = file_nemo['cell_area']

file_regular = xr.open_dataset('/data/astoppel/CMIP6/ensamble_historical/ipsl_CMIP6_TOS_1921_2014.nc')
grid_regular = {"lon": file_regular["lon"], "lat": file_regular["lat"]}

mask = xr.open_dataset('/data/astoppel/Nudging_Example/ORCA1_subbasins.nc').glomsk.to_dataset(name='msk')
mask_trop = mask.where((mask.nav_lat <= 30) & (mask.nav_lat >= -30))

mask_U_3d = xr.open_dataset('/data/astoppel/Nudging_Example/Umask_trans.nc').umask.to_dataset(name='umask')
mask_V_3d = xr.open_dataset('/data/astoppel/Nudging_Example/Vmask_trans.nc').vmask.to_dataset(name='vmask')
 
mask_eq_pac = mask.where(
    ((mask.nav_lat <= 5) & (mask.nav_lat >= -5)) &
    (((mask.nav_lon >= 140) & (mask.nav_lon <= 180)) |
     ((mask.nav_lon >= -180) & (mask.nav_lon <= -81))) )

mask_U_eq_pac = mask_U_3d.where(
    (mask_U_3d.nav_lat >= -5) & (mask_U_3d.nav_lat <= 5) &
    (((mask_U_3d.nav_lon >= 140) & (mask_U_3d.nav_lon <= 180)) |
     ((mask_U_3d.nav_lon >= -180) & (mask_U_3d.nav_lon <= -81))))




####################################################
#### Part one : corrections
####################################################




#### Functions

def shift_longitude_and_data(data, lon_360, shift_index=300):
    
    lon = lon_360
    
    data_shifted = xr.concat([data.isel(lon=slice(shift_index, None)),
                               data.isel(lon=slice(0, shift_index))], dim='lon')
    
    lon_shifted  = xr.concat([lon.isel(lon=slice(shift_index, None)),
                               lon.isel(lon=slice(0, shift_index)) + 360], dim="lon")
    
    data_shifted = data_shifted.assign_coords({'lon': lon_shifted})
    
    return data_shifted, lon_shifted

def filter_outliers(data, lower_percentile=0.1, upper_percentile=99.9):
    
    lower_bound = np.nanpercentile(data, lower_percentile)
    
    upper_bound = np.nanpercentile(data, upper_percentile)
    
    return np.where((data >= lower_bound) & (data <= upper_bound), data, np.nan)




#### Upload corrections

hf_file  = "/data/astoppel/HF_corr/Daily_clim_hfcorr_1979_corrected_smooth.nc"
mf_file  = "/data/astoppel/HF_corr/clim_tauu_diff_leap_trans_fillmiss2.nc"
mf_file_uy = "/data/astoppel/HF_corr/clim_tauv_diff_leap_trans_fillmiss2.nc"
 
HF_corr    = xr.open_dataset(hf_file)
Taux_corr  = xr.open_dataset(mf_file)
Tauy_corr  = xr.open_dataset(mf_file_uy)
 
hfcorr     = HF_corr['hfcorr'].mean(dim="time_counter")
tauuo_corr = Taux_corr['tauuo'].mean(dim="time_counter")
tauvo_corr = Tauy_corr['tauvo'].mean(dim="time_counter")

tauuo_corr_msk = tauuo_corr.where(mask_trop.msk == 1, other=0)
tauvo_corr_msk = tauvo_corr.where(mask_trop.msk == 1, other=0)
 
regridder_corr = xe.Regridder(tauuo_corr, grid_regular, 'bilinear', reuse_weights=False, ignore_degenerate=True)
 
hfcorr_reg      = regridder_corr(hfcorr)
tauuo_corr_reg  = np.array(regridder_corr(tauuo_corr_msk))
tauvo_corr_reg  = np.array(regridder_corr(tauvo_corr_msk))

tauuo_corr_reg_neg = -1 * tauuo_corr_reg
tauvo_corr_reg_neg = -1 * tauvo_corr_reg
 
lon_corr = hfcorr_reg.lon
lat_corr = hfcorr_reg.lat
 
# Equatorial profiles corrections
hf_corr_Eq     = hfcorr.where(mask_eq_pac['msk'] == 1)
tauuo_corr_Eq  = tauuo_corr.where(mask_U_eq_pac['umask'] == 1)
 
hf_corr_reg_Eq    = regridder_corr(hf_corr_Eq)
tauuo_corr_reg_Eq = regridder_corr(tauuo_corr_Eq)
 
hf_corr_reg_Eq_lon    = hf_corr_reg_Eq.sel(lat=slice(-5, 5)).mean(dim='lat')
tauuo_corr_reg_Eq_lon = -1 * tauuo_corr_reg_Eq.sel(lat=slice(-5, 5)).mean(dim='lat')
 




# Shift longitudini [0,360] -> to be centred on Pacific [120E...80W]

lon_360 = hf_corr_reg_Eq_lon.lon
 
hf_corr_msk_eq_shift, lon_shifted   = shift_longitude_and_data(hf_corr_reg_Eq_lon, lon_360)
tauuo_corr_msk_eq_shift, lon_shifted = shift_longitude_and_data(tauuo_corr_reg_Eq_lon, lon_360)

taux_corr_plot = filter_outliers(tauuo_corr_reg_neg)
tauy_corr_plot = filter_outliers(tauvo_corr_reg_neg)
 




####################################################
#### Part two : maps of bias 
####################################################




#### Functions

def apply_mask(da, mask):

    return (da * 0 + da.where(mask == 1))
    
def concat_historical_ssp(variables_histo, variabili_ssp):
    time_mean_out = {}
    eq_mean_out   = {}
    time_series_out_msk = {}
    
    for var in variables_histo.keys():
        time_mean_out[var] = {}
        eq_mean_out[var]   = {}
        time_series_out_msk[var] = {}
        
        for model in variables_histo[var]['var_names'].keys():
            if model == 'ERA5' and var in ['tauuo', 'tauvo']:
                continue
                
            if model == 'IPSL':
                path_hist = variables_histo[var]['path_ipsl']
                path_ssp  = variabili_ssp[var]['path_ipsl']
                
            elif model == 'IPSL-FA':
                path_hist = variables_histo[var]['path_ipsl_FA']
                path_ssp  = variabili_ssp[var]['path_ipsl_FA']
                
            elif model == 'ERA5':
                path_hist = variables_histo[var]['path_ERA5']
                path_ssp  = variabili_ssp[var]['path_ERA5']
                
            varname = variables_histo[var]['var_names'][model]
            da_hist = xr.open_dataset(path_hist)[varname]
            da_ssp  = xr.open_dataset(path_ssp)[varname]
            
            if model == 'IPSL-FA':
                da_hist = da_hist.rename({'time_counter': 'time'})
                da_ssp  = da_ssp.rename({'time_counter': 'time'})
                
            if model == 'ERA5' and var == 'tos':
                da_hist = da_hist.rename({'time_counter': 'time'})
                da_ssp  = da_ssp.rename({'time_counter': 'time'})
                da_ssp  = da_ssp - 273.15
                
            da_all = xr.concat([da_hist, da_ssp], dim='time')
            
            da_all = da_all.sel(time=slice('1980-01-01', '2024-12-31'))
            
            if var == 'tauuo':
                da_all_msk_oc = apply_mask(da_all, mask_U_eq_pac.umask)
            elif var == 'tauvo':
                da_all_msk_oc = apply_mask(da_all, mask_U_eq_pac.umask)
            elif var == 'tos':
                da_all_msk_oc = apply_mask(da_all, mask_eq_pac.msk)
            elif var == 'taux':
                target = time_mean_out['tauuo'][model]
                
                regridder = xe.Regridder(da_all, target, 'bilinear', reuse_weights=False, ignore_degenerate=True)
                da_all_reg = regridder(da_all)
                da_all_msk_oc = apply_mask(da_all_reg, mask_U_eq_pac.umask)
                
            lon_dependency = (
                da_all_msk_oc
                .weighted(area_nemo.fillna(0.))
                .mean(dim=['y', 'time'], skipna=True))
            
            time_mean_glob = da_all.mean(dim='time')
            time_mean_out[var][model] = time_mean_glob
            eq_mean_out[var][model]   = lon_dependency
            time_series_out_msk[var][model] = da_all_msk_oc
            
    return time_mean_out, eq_mean_out, time_series_out_msk




#### Upload sst and wind stress data

variables_histo = {
    'tos': {
        'path_ipsl':    '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_historical_MEM_gn_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-MEM_merged_grid_T.nc',
        'path_ERA5':    '/data/astoppel/ERA5/ERA5_tos_1949_2019_gn.nc',
        'var_names': {'IPSL': 'tos', 'IPSL-FA': 'tos', 'ERA5': 'tos'}
    },
    'tauuo': {
        'path_ipsl':    '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tauuo/tauuo_Oyr_IPSL-CM6A-LR_historical_MEM_gn_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Ugrid_yearmean/ensemble_mean_tauuo.nc',
        'path_ERA5':    '/data/astoppel/ERA5/ERA5_1m_ewss_194001_202412.nc',
        'var_names': {'IPSL': 'tauuo', 'IPSL-FA': 'tauuo', 'ERA5': 'tauuo'}
    },
    'tauvo': {
        'path_ipsl':    '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tauvo/tauvo_Oyr_IPSL-CM6A-LR_historical_MEM_gn_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Vgrid_yearmean/ensemble_mean_tauvo.nc',
        'path_ERA5':    '/data/astoppel/ERA5/ERA5_1m_nsss_194001_202412.nc',
        'var_names': {'IPSL': 'tauvo', 'IPSL-FA': 'tauvo', 'ERA5': 'tauvo'}
    },
    'taux': {
        'path_ipsl':    '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tauu/tauu_Ayr_IPSL-CM6A-LR_historical_MEM_gr_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_historical_annual_means.nc',
        'var_names': {'IPSL': 'tauu', 'IPSL-FA': 'taux'}
    }
}
 
variabili_ssp = {
    'tos': {
        'path_ipsl':    '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_ssp585_MEM_gn_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-ssp585-MEM_merged_grid_T.nc',
        'path_ERA5':    '/data/astoppel/ERA5/ERA5_tos_2020_2024_gn_Kelvin.nc',
        'var_names': {'IPSL': 'tos', 'IPSL-FA': 'tos', 'ERA5': 'tos'}
    },
    'tauuo': {
        'path_ipsl':    '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tauuo/tauuo_Oyr_IPSL-CM6A-LR_ssp585_MEM_gn_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Ugrid_yearmean/ensemble_mean_uo_ssp585.nc',
        'var_names': {'IPSL': 'tauuo', 'IPSL-FA': 'tauuo'}
    },
    'tauvo': {
        'path_ipsl':    '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tauvo/tauvo_Oyr_IPSL-CM6A-LR_ssp585_MEM_gn_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Vgrid_yearmean/S3-ssp585-MEM_merged_grid_V.nc',
        'var_names': {'IPSL': 'tauvo', 'IPSL-FA': 'tauvo'}
    },
    'taux': {
        'path_ipsl':    '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tauu/tauu_Ayr_IPSL-CM6A-LR_ssp585_MEM_gr_2015-2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_ssp585_annual_means.nc',
        'var_names': {'IPSL': 'tauu', 'IPSL-FA': 'taux'}
    }
}




#concatenating historical files, ssp files, masking them 

time_mean_dict, eq_mean_dict, time_series_dict = concat_historical_ssp(variables_histo, variabili_ssp)




# ERA5 wind stress

target = time_mean_dict['tauuo']['IPSL']
ERA5_ew = xr.open_dataset(variables_histo['tauuo']['path_ERA5'])['ewss'].sel(time=slice('1980-01-01', '2024-12-31'))
ERA5_ns = xr.open_dataset(variables_histo['tauvo']['path_ERA5'])['nsss'].sel(time=slice('1980-01-01', '2024-12-31'))
regridder_era5  = xe.Regridder(ERA5_ew, target, 'bilinear', reuse_weights=False, ignore_degenerate=True)

ERA5_ew_reg     = regridder_era5(ERA5_ew.chunk({'time': 10}))
ERA5_ns_reg     = regridder_era5(ERA5_ns.chunk({'time': 10}))

ERA5_ew_reg_msk = apply_mask(ERA5_ew_reg, mask_U_eq_pac.umask)
ERA5_ns_reg_msk = apply_mask(ERA5_ns_reg, mask_U_eq_pac.umask)

ERA5_tm_ew = ERA5_ew_reg.mean("time").compute()
ERA5_tm_ns = ERA5_ns_reg.mean("time").compute()

ERA5_lm_ew = (ERA5_ew_reg_msk.weighted(area_nemo.fillna(0.))).mean(dim=['y', 'time'], skipna=True).compute()




# Bias SST and wind stress on regular grid 

tos = time_mean_dict['tos']['IPSL']
regridder_tos      = xe.Regridder(tos, grid_regular, 'bilinear', reuse_weights=False, ignore_degenerate=True)

sst_bias_ipsl      = regridder_tos(time_mean_dict['tos']['IPSL']    - time_mean_dict['tos']['ERA5'])
sst_bias_ipsl_fa   = regridder_tos(time_mean_dict['tos']['IPSL-FA'] - time_mean_dict['tos']['ERA5'])

tauu_bias_ipsl     = np.array(regridder_tos(time_mean_dict['tauuo']['IPSL']    - ERA5_tm_ew))
tauu_bias_ipsl_fa  = np.array(regridder_tos(time_mean_dict['tauuo']['IPSL-FA'] - ERA5_tm_ew))

tauv_bias_ipsl     = np.array(regridder_tos(time_mean_dict['tauvo']['IPSL']    - ERA5_tm_ns))
tauv_bias_ipsl_fa  = np.array(regridder_tos(time_mean_dict['tauvo']['IPSL-FA'] - ERA5_tm_ns))
 
lon = sst_bias_ipsl.lon
lat = sst_bias_ipsl.lat

#fixing artifacts of the grid 

eq_mean_dict['taux']['IPSL-FA'][105] = eq_mean_dict['taux']['IPSL-FA'][104]
eq_mean_dict['taux']['IPSL-FA'][106] = eq_mean_dict['taux']['IPSL-FA'][107]




####################################################
#### Part three : Plot
####################################################




# X axis 

x_full   = eq_mean_dict['tos']['IPSL'].x
xticks_pac     = [47.5, 77.5, 107.5, 137.5, 167.5, 197.5]
xticklabels_pac = ['120°E', '150°E', '180°', '150°W', '120°W', '90°W']

xticks_shifted = [480, 510,  540, 570, 600, 630]
xticklabels_shifted = ['120°E','150°E', '180°','150°W', '120°W', '90°W']


xlim_pac = [27.5, 217.5]
xlim_pac_shifted = [460, 659]




fig = plt.figure(figsize=(15, 12))
 
gs_main = fig.add_gridspec( nrows=3, ncols=2, height_ratios=[1, 1, 1], width_ratios=[1.8, 1.3], hspace=0.15, wspace=0.28)
 
ax_prof_b = fig.add_subplot(gs_main[0, 1])  
ax_prof_d = fig.add_subplot(gs_main[1, 1])  
ax_prof_f = fig.add_subplot(gs_main[2, 1])  
 
cmap = plt.cm.get_cmap('RdBu_r')

lvl_hf   = np.linspace(-90, 90, 10)
lvl_sst  = np.linspace(-1.8, 1.8, 10)
 
proj_pac  = ccrs.PlateCarree(central_longitude=180)
proj_corr = ccrs.PlateCarree(central_longitude=120)
 
# 1) map and profile of corrections 

# a
ax_map1 = fig.add_subplot(gs_main[0, 0], projection=proj_corr)
 
ax_map1.set_title(r"a) Heat and Momentum fluxes adjustments $\delta Q_c$, $\delta\tau_c$ ", loc='left', fontsize=15)

cs1 = ax_map1.contourf(
    lon_corr, lat_corr, hfcorr_reg,
    levels=lvl_hf, cmap=cmap,
    transform=ccrs.PlateCarree(), extend='both',rasterized=True)

cf1 = ax_map1.contour(
    lon_corr, lat_corr, hfcorr_reg,
    levels=lvl_hf, colors='black', linewidths=0.3,
    transform=ccrs.PlateCarree() )

for col in cf1.collections:
    col.set_rasterized(True)


ax_map1.hlines(
    y=[5, -5], xmin=lon_corr.min(), xmax=lon_corr.max(),
    colors='k', linestyles='--', linewidth=1.5,
    transform=ccrs.PlateCarree(), zorder=3
)
    
step_x, step = 7, 7 
q1 = ax_map1.quiver(
    lon_corr[::step_x], lat_corr[::step],
    taux_corr_plot[::step, ::step_x],
    tauy_corr_plot[::step, ::step_x],
    transform=ccrs.PlateCarree(),
    scale=0.7, width=0.002, pivot='tail', zorder=1, color='black'
)

ax_map1.set_extent([100, 290, -30, 30], crs=ccrs.PlateCarree())
ax_map1.add_feature(cfeature.LAND, facecolor="white", zorder=2)
ax_map1.add_feature(cfeature.COASTLINE, linewidth=0.8)
gl1 = ax_map1.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl1.top_labels = False; gl1.right_labels = False
gl1.xlabel_style = {'size': 12}; gl1.ylabel_style = {'size': 12}
 
cax1 = fig.add_axes([0.15, 0.64, 0.38, 0.015])
cbar1 = fig.colorbar(cs1, cax=cax1, orientation='horizontal', extend='both')
cbar1.set_label(r"W m$^{-2}$", fontsize=14)
cbar1.ax.tick_params(labelsize=14)
cbar1.ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
 
# b 
ax_prof_b.set_title(r"b) $\delta Q_c$, $\delta\tau_{x,c}$ Equatorial Pacific", loc='left', fontsize=15)
ax_prof_b.set_ylabel(r"W m$^{-2}$", fontsize=14)
ax_prof_b.tick_params(labelsize=14)
ax_prof_b.set_xlim(xlim_pac_shifted)
ax_prof_b.set_ylim(-60, 60)
ax_prof_b.set_xticklabels([])
ax_prof_b.set_xticks([])
ax_prof_b.axhline(0, color='k', lw=0.8, alpha=0.3, ls='--')
ax_prof_b.plot(lon_shifted, hf_corr_msk_eq_shift, color='k', lw=2, label=r'$\delta Q_c$')
 
ax_prof_b_r = ax_prof_b.twinx()
ax_prof_b_r.set_ylim(-0.02, 0.02)
ax_prof_b_r.plot(lon_shifted, tauuo_corr_msk_eq_shift, color='r', lw=2, label=r'$\delta\tau_{x,c}$')
ax_prof_b_r.set_ylabel(r"N m$^{-2}$", fontsize=14, color='r')
ax_prof_b_r.tick_params(axis='y', labelsize=14, colors='r')
ax_prof_b_r.spines['right'].set_color('r')
 
lines1, labels1 = ax_prof_b.get_legend_handles_labels()
lines2, labels2 = ax_prof_b_r.get_legend_handles_labels()
ax_prof_b.legend(lines2 + lines1, labels2 + labels1, fontsize=14, loc='lower left')
 
# maps and profile of ipsl bias

#c)

ax_map2 = fig.add_subplot(gs_main[1, 0], projection=proj_pac)
 
ax_map2.set_title(r"c) IPSL SST and $\vec{\tau}$ bias (1980–2024)", loc='left', fontsize=15)
cs2 = ax_map2.contourf(
    lon, lat, sst_bias_ipsl,
    levels=lvl_sst, cmap=cmap,
    transform=ccrs.PlateCarree(), extend='both',
        rasterized=True)

cf2 = ax_map2.contour(
    lon, lat, sst_bias_ipsl,
    levels=lvl_sst, colors='black', linewidths=0.3,
    transform=ccrs.PlateCarree()
)

for col in cf2.collections:
    col.set_rasterized(True)
    
step = 7
ax_map2.quiver(
    lon[::step], lat[::step],
    tauu_bias_ipsl[::step, ::step],
    tauv_bias_ipsl[::step, ::step],
    transform=ccrs.PlateCarree(),
    scale=0.7, width=0.002, pivot='mid', color='black'
)
ax_map2.set_extent([100, 290, -30, 30], crs=ccrs.PlateCarree())
ax_map2.add_feature(cfeature.COASTLINE, lw=0.6)
ax_map2.hlines(y=[5, -5], xmin=100, xmax=290, colors='k', linestyles='--',
               linewidth=1.5, transform=ccrs.PlateCarree(), zorder=3)
gl2 = ax_map2.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl2.top_labels = False; gl2.right_labels = False
gl2.xlabel_style = {'size': 12}; gl2.ylabel_style = {'size': 12}
 
#d)

ax_prof_d.set_title("d) SST Equatorial Pacific (1980–2024)", loc='left', fontsize=15)
ax_prof_d.set_ylabel("°C", fontsize=14)
ax_prof_d.set_ylim([23.5, 30.5])
ax_prof_d.tick_params(labelsize=14)
ax_prof_d.plot(x_full, eq_mean_dict['tos']['ERA5'],    color='k',    lw=2, label='ERA5')
ax_prof_d.plot(x_full, eq_mean_dict['tos']['IPSL'],    color='red',  lw=2, label='IPSL (OCE)')
ax_prof_d.plot(x_full, eq_mean_dict['tos']['IPSL-FA'], color='blue', lw=2, label='IPSL-FA (OCE)')
ax_prof_d.legend(fontsize=14, loc='lower left')
ax_prof_d.set_xlim([27.5, 217.5])
ax_prof_d.set_xticklabels([])
ax_prof_d.set_xticks([])
 
# maps and profile of ipsl-fa bias

#e)

ax_map3 = fig.add_subplot(gs_main[2, 0], projection=proj_pac)
 
ax_map3.set_title(r"e) IPSL-FA SST and $\vec{\tau}$ bias (1980–2024)", loc='left', fontsize=15)
cs3 = ax_map3.contourf(
    lon, lat, sst_bias_ipsl_fa,
    levels=lvl_sst, cmap=cmap,
    transform=ccrs.PlateCarree(), extend='both',
        rasterized=True)

cf3 = ax_map3.contour(
    lon, lat, sst_bias_ipsl_fa,
    levels=lvl_sst, colors='black', linewidths=0.3,
    transform=ccrs.PlateCarree()
)


for col in cf3.collections:
    col.set_rasterized(True)


q3 = ax_map3.quiver(
    lon[::step], lat[::step],
    tauu_bias_ipsl_fa[::step, ::step],
    tauv_bias_ipsl_fa[::step, ::step],
    transform=ccrs.PlateCarree(),
    scale=0.7, width=0.002, pivot='mid', color='black'
)
ax_map3.quiverkey(
    q3, 0.85, 1.05, 0.025, 
    r"0.025 N m$^{-2}$",
    labelpos='E', coordinates='axes', fontproperties={'size': 12}
)
ax_map3.set_extent([100, 290, -30, 30], crs=ccrs.PlateCarree())
ax_map3.add_feature(cfeature.COASTLINE, lw=0.6)
ax_map3.hlines(y=[5, -5], xmin=100, xmax=290, colors='k', linestyles='--',
               linewidth=1.5, transform=ccrs.PlateCarree(), zorder=3)
gl3 = ax_map3.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
gl3.top_labels = False; gl3.right_labels = False
gl3.xlabel_style = {'size': 12}; gl3.ylabel_style = {'size': 12}
 
cax_sst = fig.add_axes([0.15, 0.1, 0.38, 0.015])
cbar_sst = fig.colorbar(cs2, cax=cax_sst, orientation='horizontal', extend='both')
cbar_sst.set_label("SST bias [°C]", fontsize=14)
cbar_sst.ax.tick_params(labelsize=11)
cbar_sst.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
 
#f)

ax_prof_f.set_title(r"f) $\tau_x$ Equatorial Pacific (1980–2024)", loc='left', fontsize=15)
ax_prof_f.set_ylabel(r'N m$^{-2}$', fontsize=14)
ax_prof_f.set_ylim([-0.065, 0.025])
ax_prof_f.tick_params(labelsize=14)
ax_prof_f.plot(x_full, ERA5_lm_ew,                          color='k',    lw=2)
ax_prof_f.plot(x_full, eq_mean_dict['tauuo']['IPSL'],        color='red',  lw=2)
ax_prof_f.plot(x_full, eq_mean_dict['tauuo']['IPSL-FA'],     color='blue', lw=2)
ax_prof_f.plot(x_full, eq_mean_dict['taux']['IPSL'],         color='red',  lw=2, linestyle='dotted', label='IPSL (ATM)')
ax_prof_f.plot(x_full, eq_mean_dict['taux']['IPSL-FA'],      color='blue', lw=2, linestyle='dotted', label='IPSL-FA (ATM)')
ax_prof_f.legend(fontsize=14, loc='upper left', ncol=1)
ax_prof_f.set_xlim([27.5, 217.5])
ax_prof_f.set_xticks(xticks_pac)
ax_prof_f.set_xticklabels(xticklabels_pac, fontsize=14)
 

# plt.savefig("/home/astoppel/figure/bias/combined_corrections_biases.pdf", bbox_inches='tight')
plt.show()

