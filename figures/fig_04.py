#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.



import numpy as np
import xesmf as xe
import xarray as xr
import matplotlib.pyplot as plt
import sys
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D




#Grids & Masks

file_nemo = xr.open_dataset('/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-r4_merged_grid_T.nc').sel(time_counter=slice('1950-01-01', '1950-02-01'))
area_nemo = file_nemo['cell_area']

mask = xr.open_dataset('/home/astoppel/glomask.nc').msk.to_dataset(name='msk')
mask_eq = mask.where(
    ((mask.nav_lat <= 5) & (mask.nav_lat >= -5)) &
    (((mask.nav_lon >= 140) & (mask.nav_lon <= 180)) |
        ((mask.nav_lon >= -180) & (mask.nav_lon <= -80))))




####################################################
#### Part one : data upload (ipsl, ipsl-fa, oras5)
####################################################




#### Functions

def apply_mask(da, mask):

    return (da * 0 + da.where(mask == 1))




# IPSL (wo)

IPSL_wo_hist = xr.open_dataset('/scratchu/astoppel/IPSL_CTL/historical/CMIP6/wo/wo_Oyr_IPSL-CM6A-LR_historical_MEM_gn_1950-2014.nc')['wo']
IPSL_wo_scen = xr.open_dataset('/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/wo/wo_Oyr_IPSL-CM6A-LR_ssp585_MEM_gn_2015-2099.nc')['wo']                                                                      
IPSL_wo = xr.concat([IPSL_wo_hist, IPSL_wo_scen], dim='time').sel(time=slice('1980-01-01', '2024-12-31'))

IPSL_wo_mask = apply_mask(IPSL_wo, mask_eq.msk)

ref_IPSL_wo_x  = IPSL_wo_mask.weighted(area_nemo.fillna(0.)).mean(dim=['y', 'time'], skipna=True)

del IPSL_wo_hist, IPSL_wo_scen, IPSL_wo, IPSL_wo_mask

# IPSL FA (wo)

FA_wo_hist = xr.open_dataset('/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Wgrid_yearmean/ensemble_mean_wo_historical.nc')['wo']
FA_wo_scen = xr.open_dataset('/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Wgrid_yearmean/ensemble_mean_wo_ssp585.nc')['wo']
FA_wo = xr.concat([FA_wo_hist, FA_wo_scen], dim='time_counter').sel(time_counter=slice('1980-01-01', '2024-12-31')) \
          .rename({'y_grid_W': 'y', 'x_grid_W': 'x'})

FA_wo_mask  = apply_mask(FA_wo, mask_eq.msk)

ref_IPSL_FA_wo_x = FA_wo_mask.weighted(area_nemo.fillna(0.)).mean(dim=['y', 'time_counter'], skipna=True)

del FA_wo_hist, FA_wo_scen, FA_wo, FA_wo_mask




# IPSL (thetao)

IPSL_thetao_hist = xr.open_dataset(
    '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/thetao/thetao_Oyr_IPSL-CM6A-LR_historical_MEM_gn_1950-2014.nc',
    chunks={'time': 5})['thetao']

IPSL_thetao_scen = xr.open_dataset(
    '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/thetao/thetao_Oyr_IPSL-CM6A-LR_ssp585_MEM_gn_2015-2099.nc',
    chunks={'time': 5})['thetao']

IPSL_thetao = xr.concat([IPSL_thetao_hist, IPSL_thetao_scen], dim='time').sel(time=slice('1980-01-01', '2024-12-31'))

IPSL_thetao_mask = apply_mask(IPSL_thetao, mask_eq.msk)

ref_IPSL_x  = IPSL_thetao_mask.weighted(area_nemo.fillna(0.)).mean(dim=['y', 'time'], skipna=True)
ref_IPSL_x = ref_IPSL_x.compute()

del IPSL_thetao_hist, IPSL_thetao_scen, IPSL_thetao, IPSL_thetao_mask


# IPSL FA (thetao)
FA_thetao_hist = xr.open_dataset(
    '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-MEM_merged_grid_T.nc',
    chunks={'time_counter': 5})['thetao']

FA_thetao_scen = xr.open_dataset(
    '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-ssp585-MEM_merged_grid_T.nc',
    chunks={'time_counter': 5})['thetao']

FA_thetao = xr.concat([FA_thetao_hist, FA_thetao_scen], dim='time_counter').sel(time_counter=slice('1980-01-01', '2024-12-31'))

FA_thetao_mask = apply_mask(FA_thetao, mask_eq.msk)

ref_IPSL_FA_x  = FA_thetao_mask.weighted(area_nemo.fillna(0.)).mean(dim=['y', 'time_counter'], skipna=True)
ref_IPSL_FA_x = ref_IPSL_FA_x.compute()

del FA_thetao_hist, FA_thetao_scen, FA_thetao, FA_thetao_mask




# ORAS5 (thetao)

Obsthetao = xr.open_dataset('/scratchu/astoppel/ORAS5/f5167edb8db31cbfeed05f67c6e149d2/votemper_1958_2024.nc', 
                            chunks={'time_counter':5 , 'deptht':5}).sel(time_counter=slice('1980-01-01', '2024-02-01'))

da_ORAS5 = Obsthetao['votemper']

regridder = xe.Regridder(da_ORAS5,  file_nemo, 'bilinear', reuse_weights=False, ignore_degenerate=True)

ORAS5_regridder = regridder(da_ORAS5)

du_ORAS5_mask = apply_mask(ORAS5_regridder, mask_eq.msk) 
        
ref_ORAS5_x= du_ORAS5_mask.weighted(area_nemo.fillna(0.)).mean(dim=['y','time_counter'], skipna=True)

# ref_ORAS5_x.deptht # sono uguali a quelli di IPSL quindi rinomino semplicemente 
ref_ORAS5_x = ref_ORAS5_x.rename({'deptht':'olevel'})

ref_ORAS5_x = ref_ORAS5_x.compute()

del da_ORAS5, ORAS5_regridder




# Slice depth
depth_slice = slice(0, 300)
FA_thetao_dep = ref_IPSL_FA_x.sel(olevel=depth_slice) 
IPSL_thetao_dep = ref_IPSL_x.sel(olevel=depth_slice) 
ORAS5_thetao_dep = ref_ORAS5_x.sel(olevel=depth_slice) 

#unit conversion
FA_wo_dep = ref_IPSL_FA_wo_x.sel(olevel=depth_slice)*86400 # m/s -> m/day    
IPSL_wo_dep = ref_IPSL_wo_x.sel(olevel=depth_slice)*86400 # m/s -> m/day    
diff_wo = FA_wo_dep - IPSL_wo_dep

#bias computation
bias_ipsl = IPSL_thetao_dep - ORAS5_thetao_dep
bias_ipsl_Fa = FA_thetao_dep - ORAS5_thetao_dep




####################################################
#### Part two : Plot
####################################################




x_full = bias_ipsl_Fa.x
fontsize_base = 16
labelsize_base = 14
line_value = 1.5
cmap_t = plt.cm.get_cmap('RdBu_r')

xticks = [ 77.5, 107.5, 137.5, 167.5, 197.5]
xticklabels = [ '150°E', '180°E', '150°W', '120°W', '90°W']

fig = plt.figure(figsize=(15, 9))

gs = GridSpec(
    2, 2,
    height_ratios=[1, 1],
    width_ratios=[1, 1],
    hspace=0.55,
    wspace=0.15
)

levels_thetao = np.linspace(-2.75, 2.75, 12)
levels_wo = np.linspace(-0.55, 0.55, 12)


ax_t1 = fig.add_subplot(gs[0, 1])
ax_t2 = fig.add_subplot(gs[0, 0])

#upper line (wo)
ax_t2.contourf(
    x_full, IPSL_wo_dep['olevel'],
    IPSL_wo_dep.transpose('olevel', 'x'),
    levels=levels_wo, cmap=cmap_t, extend='both'
)

cf1 = ax_t1.contourf(
    x_full, FA_wo_dep['olevel'],
    FA_wo_dep.transpose('olevel', 'x'),
    levels=levels_wo, cmap=cmap_t, extend='both'
)


ax_t2.contour(x_full, ORAS5_thetao_dep['olevel'], ORAS5_thetao_dep.transpose('olevel','x'),
           levels=[20], colors='black')
ax_t2.contour(x_full, IPSL_thetao_dep['olevel'], IPSL_thetao_dep.transpose('olevel','x'),
           levels=[20], colors='red' )
ax_t2.contour(x_full, FA_thetao_dep['olevel'], FA_thetao_dep.transpose('olevel','x'),
           levels=[20], colors='blue' )

ax_t1.contour(x_full, ORAS5_thetao_dep['olevel'], ORAS5_thetao_dep.transpose('olevel','x'),
           levels=[20], colors='black' )
ax_t1.contour(x_full, IPSL_thetao_dep['olevel'], IPSL_thetao_dep.transpose('olevel','x'),
           levels=[20], colors='red' )
ax_t1.contour(x_full, FA_thetao_dep['olevel'], FA_thetao_dep.transpose('olevel','x'),
           levels=[20], colors='blue' )

#colorbar
bax = fig.add_axes([0.25, 0.5, 0.5, 0.025])
cbar = fig.colorbar(cf1, cax=bax, orientation='horizontal', extend='both')
cbar.set_label("W [m/day]", fontsize=14)
cbar.ax.tick_params(labelsize=14)
cbar.set_ticks(cbar.get_ticks()[::1])
cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))


#bottom line (thetao bias)

gs_bottom = gs[1, :].subgridspec(1, 2, wspace=0.15)

ax  = fig.add_subplot(gs_bottom[0, 0])
ax2 = fig.add_subplot(gs_bottom[0, 1])


for axi in [ax, ax2, ax_t1, ax_t2]:
    axi.set_xlim([67.5, 207.5])
    axi.set_ylim(bias_ipsl['olevel'].max(), bias_ipsl['olevel'].min())
    axi.set_xticks(xticks)
    axi.set_xticklabels(xticklabels, fontsize=labelsize_base)
    axi.tick_params(labelsize=labelsize_base)


cfb = ax.contourf(
    x_full, bias_ipsl['olevel'],
    bias_ipsl.transpose('olevel', 'x'),
    levels=levels_thetao, cmap=cmap_t, extend='both'
)

ax.contour(x_full, ORAS5_thetao_dep['olevel'], ORAS5_thetao_dep.transpose('olevel','x'),
           levels=[20], colors='black' )
ax.contour(x_full, IPSL_thetao_dep['olevel'], IPSL_thetao_dep.transpose('olevel','x'),
           levels=[20], colors='red'  )
ax.contour(x_full, FA_thetao_dep['olevel'], FA_thetao_dep.transpose('olevel','x'),
           levels=[20], colors='blue'  )

ax.set_ylabel("Depth [m]", fontsize=fontsize_base)

handles = [
    Line2D([0], [0], color='black', lw=line_value, label='ORAS5'),
    Line2D([0], [0], color='red',   lw=line_value, label='IPSL'),
    Line2D([0], [0], color='blue',  lw=line_value, label='IPSL-FA'),
]

ax.legend(handles=handles, fontsize=12, loc='lower right')


ax2.contourf(
    x_full, bias_ipsl_Fa['olevel'],
    bias_ipsl_Fa.transpose('olevel', 'x'),
    levels=levels_thetao, cmap=cmap_t, extend='both'
)

ax2.contour(x_full, IPSL_thetao_dep['olevel'], IPSL_thetao_dep.transpose('olevel','x'),
            levels=[20], colors='red', linewidth=line_value)
ax2.contour(x_full, FA_thetao_dep['olevel'], FA_thetao_dep.transpose('olevel','x'),
            levels=[20], colors='blue', linewidth=line_value)
ax2.contour(x_full, ORAS5_thetao_dep['olevel'], ORAS5_thetao_dep.transpose('olevel','x'),
            levels=[20], colors='black', linewidth=line_value)

ax_t2.set_title("a) Vertical velocity, IPSL (1980–2024)", fontsize=fontsize_base, loc='left')
ax_t1.set_title("b) Vertical velocity, IPSL-FA (1980–2024)", fontsize=fontsize_base, loc='left')
ax.set_title("c) Temperature bias, IPSL − ORAS5 (1980–2024)", fontsize=fontsize_base, loc='left')
ax2.set_title("d) Temperature bias, IPSL-FA − ORAS5 (1980–2024)", fontsize=fontsize_base, loc='left')


#colorbar

cax = fig.add_axes([0.25, 0.05, 0.5, 0.025])
cbar = fig.colorbar(cfb, cax=cax, orientation='horizontal', extend='both')
cbar.set_label("T Bias [°C]", fontsize=14)
cbar.ax.tick_params(labelsize=14)
cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ticks = np.linspace(cfb.norm.vmin, cfb.norm.vmax, 12)
cbar.set_ticks(ticks)

# plt.savefig("/home/astoppel/figure/bias/thetao_wo_1980_2024.pdf", bbox_inches='tight')







