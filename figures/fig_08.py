#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.



import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.lines import Line2D




#### Grids and masks

file_nemo = xr.open_dataset('/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-r4_merged_grid_T.nc').sel(time_counter=slice('1950-01-01', '1950-02-01'))
area_nemo = file_nemo['cell_area']

# file_regular = xr.open_dataset('/data/astoppel/CMIP6/ensamble_historical/ipsl_CMIP6_TOS_1921_2014.nc')
# grid_regular = {"lon": file_regular["lon"], "lat": file_regular["lat"]}




mask = xr.open_dataset('/data/astoppel/Nudging_Example/ORCA1_subbasins.nc').glomsk.to_dataset(name='msk')

mask_trop = mask.where((mask.nav_lat <= 30) & (mask.nav_lat >= -30) )

mask_EEP = mask.where(
    ((mask.nav_lat <= 5) & (mask.nav_lat >= -5)) &
    ((mask.nav_lon >= -180) & (mask.nav_lon <= -80)) )

mask_WEP = mask.where(
    ((mask.nav_lat <= 5) & (mask.nav_lat >= -5)) &
    ((mask.nav_lon >= 110) & (mask.nav_lon <= 180)) ) 

boxes = {
    'EEP': mask_EEP.msk,
    'WEP': mask_WEP.msk
}

vertical_config = {
    'uo': 0,
    'vo': 0,
    'wo': 50
}

mask_wep = mask_WEP.msk
mask_eep = mask_EEP.msk
mask_30 = mask_trop.msk




####################################################
#### Part one : Preprocessing the variables
####################################################




#### Functions

def select_period(ds, start_year, end_year,  dim=None):
    
    if dim is None:
        if 'time_counter' in ds.dims:
            dim = 'time_counter'
        elif 'time' in ds.dims:
            dim = 'time'
        else:
            raise ValueError("Non riesco a trovare una dimensione temporale (time_counter o time)")

    return ds.sel(**{dim: slice(f"{start_year}-01-01", f"{end_year}-12-31")})

def apply_mask(da, mask):

    if not isinstance(da, xr.DataArray):
        raise TypeError("da deve essere un xarray.DataArray")
    
    return (da * 0 + da.where(mask == 1))
    
def preprocess(path, varname, var, start_year, end_year, model, mask=None):

    mean = xr.open_dataset(path)

    time_series = select_period(mean[varname], start_year, end_year)
    
    if var in ['wo'] and model in ['IPSL-FA']:
        time_series = time_series.rename({'x_grid_W': 'x', 'y_grid_W': 'y'})

    if var in ['pr'] :
        
        regridder = xe.Regridder(time_series, file_nemo, 'bilinear',
                                 reuse_weights=False,
                                 ignore_degenerate=True)
        
        time_series = regridder(time_series)

    if mask is not None:
        time_series = apply_mask(time_series, mask)

    return time_series






variables_histo = {
    'tos': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_historical_{member}_gn_1950-2014.nc',
        'path_ipsl-fa': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-{member}_merged_grid_T.nc',
        'var_names': {
            'IPSL': 'tos',
            'IPSL-FA': 'tos',
        }
    },
    'tauuo': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tauuo/tauuo_Oyr_IPSL-CM6A-LR_historical_{member}_gn_1950-2014.nc',
        'path_ipsl-fa': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Ugrid_yearmean/S3-tau200-{member}_merged_grid_U.nc',
        'var_names': {
            'IPSL': 'tauuo',
            'IPSL-FA': 'tauuo',
        }
    }, 
    'pr': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/pr/pr_Ayr_IPSL-CM6A-LR_historical_{member}_gr_1950-2014.nc',
        'path_ipsl-fa': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-{member}_merged_grid_T.nc',
        'var_names': {           
            'IPSL': 'pr',
            'IPSL-FA': 'rain',
        }
    },
    'uo':{
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/uo/uo_Oyr_IPSL-CM6A-LR_historical_{member}_gn_1950-2014.nc',
        'path_ipsl-fa': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Ugrid_yearmean/uo_surface/S3-tau200-{member}_merged_grid_U.nc',
        'var_names': {
            'IPSL': 'uo',
            'IPSL-FA': 'uo'
        }
    }, 
     'wo': {
         'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/wo/wo_Oyr_IPSL-CM6A-LR_historical_{member}_gn_1950-2014.nc',
         'path_ipsl-fa': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Wgrid_yearmean/S3-tau200-{member}_merged_grid_W.nc',
         'var_names': {
             'IPSL': 'wo',
             'IPSL-FA': 'wo'
         }
     }
}

variables_histo['rsst'] = variables_histo['tos']





variables_ssp = {
    'tos': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_ssp585_{member}_gn_2015_2099.nc',
        'path_ipsl-fa': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-ssp585-{member}_merged_grid_T.nc',
        'var_names': {
            'IPSL': 'tos',
            'IPSL-FA': 'tos'
        }
    },
    'tauuo': {
        'path_ipsl': "/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tauuo/tauuo_Oyr_IPSL-CM6A-LR_ssp585_{member}_gn_2015_2099.nc",
        'path_ipsl-fa': "/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Ugrid_yearmean/S3-ssp585-{member}_merged_grid_U.nc",
        'var_names': {
            'IPSL': 'tauuo',
            'IPSL-FA': 'tauuo'
        }
    }, 
    'pr': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/pr/pr_Ayr_IPSL-CM6A-LR_ssp585_{member}_gr_2015_2099.nc',
        'path_ipsl-fa': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-ssp585-{member}_merged_grid_T.nc',
        'var_names': {           
            'IPSL': 'pr',
            'IPSL-FA': 'rain',
        }
    },
     'uo': {
         'path_ipsl': "/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/uo/uo_Oyr_IPSL-CM6A-LR_ssp585_{member}_gn_2015_2099.nc",
         'path_ipsl-fa': "/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Ugrid_yearmean/S3-ssp585-{member}_merged_grid_U.nc",
         'var_names': {
             'IPSL': 'uo',
             'IPSL-FA': 'uo'
         }
     },
     'wo': {
         'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/wo/wo_Oyr_IPSL-CM6A-LR_ssp585_{member}_gn_2015-2099.nc',
         'path_ipsl-fa': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Wgrid_yearmean/S3-ssp585-{member}_merged_grid_W.nc',
         'var_names': {
             'IPSL': 'wo',
             'IPSL-FA': 'wo'
         }
     }
}

variables_ssp['rsst'] = variables_ssp['tos']




#### uploading, selecting period, masking all the variables for each box

members = ['r1','r2','r3','r4','r12','r13','r14','r15','r16','r18','r22','r23','r29','r30','r33']

climatologia_ref = {}
histo = {}
ssp = {}
anomalia = {}
absolute = {}  

for var in variables_histo.keys():
    print(var)

    climatologia_ref[var] = {}
    histo[var] = {}
    ssp[var] = {}
    anomalia[var] = {}
    absolute[var] = {}

    info_h = variables_histo[var]
    info_s = variables_ssp[var]

    for model in ['IPSL', 'IPSL-FA']:
        key = f'path_{model.lower()}'

        climatologia_ref[var][model] = {}
        histo[var][model]            = {}
        ssp[var][model]              = {}
        anomalia[var][model]         = {}
        absolute[var][model]         = {}

        for box_name, mask in boxes.items():
            print(f'  {model} / {box_name}')

            histo[var][model][box_name]            = {}
            ssp[var][model][box_name]              = {}
            anomalia[var][model][box_name]         = {}
            absolute[var][model][box_name]         = {}

            clim_members = []

            for member in members:
                print(f'    {member}')
                try:
                    path_h = info_h[key].format(member=member)
                    path_s = info_s[key].format(member=member)

                    # =========================
                    # HISTORICAL
                    # =========================
                    ts_h = preprocess(path_h, info_h['var_names'][model], var, 1950, 2014, model, mask)
                    if 'time_counter' in ts_h.dims:
                        ts_h = ts_h.rename({'time_counter': 'time'})

                    if var in ['rsst', 'zos']:
                        ts_h_trop = preprocess(path_h, info_h['var_names'][model], var, 1950, 2014, model, mask_30)
                        if 'time_counter' in ts_h_trop.dims:
                            ts_h_trop = ts_h_trop.rename({'time_counter': 'time'})
                        trop_h = ts_h_trop.weighted(area_nemo.fillna(0.)).mean(dim=["y", "x"], skipna=True)
                        ts_h = ts_h - trop_h

                    if var in vertical_config:
                        depth = vertical_config[var]
                        if depth == 0:
                            ts_h = ts_h.sel(olevel=0, method='nearest')
                        else:
                            ts_h = ts_h.sel(olevel=slice(0, depth)).mean('olevel')

                    ts_h_x = ts_h.weighted(area_nemo.fillna(0.)).mean('y', skipna=True)
                    histo[var][model][box_name][member] = ts_h_x

                    ref = select_period(ts_h_x, 1950, 1979)
                    clim_member = ref.mean('time')
                    clim_members.append(clim_member)

                    # =========================
                    # SSP
                    # =========================
                    ts_s = preprocess(path_s, info_s['var_names'][model], var, 2015, 2099, model, mask)
                    if 'time_counter' in ts_s.dims:
                        ts_s = ts_s.rename({'time_counter': 'time'})

                    if var in ['rsst', 'zos']:
                        ts_s_trop = preprocess(path_s, info_s['var_names'][model], var, 2015, 2099, model, mask_30)
                        if 'time_counter' in ts_s_trop.dims:
                            ts_s_trop = ts_s_trop.rename({'time_counter': 'time'})
                        trop_s = ts_s_trop.weighted(area_nemo.fillna(0.)).mean(dim=["y", "x"], skipna=True)
                        ts_s = ts_s - trop_s

                    if var in vertical_config:
                        depth = vertical_config[var]
                        if depth == 0:
                            ts_s = ts_s.sel(olevel=0, method='nearest')
                        else:
                            ts_s = ts_s.sel(olevel=slice(0, depth)).mean('olevel')

                    ts_s_x = ts_s.weighted(area_nemo.fillna(0.)).mean('y', skipna=True)
                    ssp[var][model][box_name][member] = ts_s_x

                    # =========================
                    # CONCAT + ANOMALIA + ABSOLUTE
                    # =========================
                    ts_all = xr.concat([ts_h_x, ts_s_x], dim='time').sortby('time')
                    anomalia[var][model][box_name][member] = ts_all - clim_member
                    absolute[var][model][box_name][member] = ts_all

                except Exception as e:
                    print(f'      SKIP {var}/{model}/{box_name}/{member}: {e}')
                    continue

            if clim_members:
                climatologia_ref[var][model][box_name] = xr.concat(clim_members, dim='member').mean('member')





####################################################
#### Part two : Computing the Confidence interval
####################################################




#### Functions
def get_CI_90(dati_array):
    std = np.nanstd(dati_array, axis=0, ddof=1)
    n_membri = 15
    t_critico = stats.t.ppf(0.95, df=n_membri-1)
    return t_critico * (std / np.sqrt(n_membri))


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
        dims=["year"],                 
        coords={"year": yearly_da.year}  
    )




media_smoothed = {}
media_smoothed_sig = {} 
percentili = {}

for var in anomalia:
    print(f"Elaborating: {var}")
    
    media_smoothed[var] = {}
    media_smoothed_sig[var] = {}  
    percentili[var] = {}
    
    for modello in anomalia[var]:
        media_smoothed[var][modello] = {}
        media_smoothed_sig[var][modello] = {}   
        percentili[var][modello] = {}
        
        for zona in anomalia[var][modello]:
            
            dizionario_membri = anomalia[var][modello][zona]
            lista_membri = list(dizionario_membri.values())
            
            anni = lista_membri[0].time.dt.year.values
            n_anni = len(anni)
            n_membri = 15
            
            tabella_dati = np.zeros((n_membri, n_anni))
            for idx_membro in range(n_membri):
                dati_membro = lista_membri[idx_membro]
                dati_temporali = dati_membro.mean(dim='x')
                tabella_dati[idx_membro, :] = dati_temporali.values
            
            media = np.nanmean(tabella_dati, axis=0)
            ci = get_CI_90(tabella_dati)  
            
            if var in ['pr', 'wo']:
                media = media * 86400
                ci = ci * 86400
            
            media_da = xr.DataArray(media, dims=['year'], coords={'year': anni})
            ci_da = xr.DataArray(ci, dims=['year'], coords={'year': anni})
            
            media_smoothed[var][modello][zona] = rolling_mean(media_da)
            
            ci_smooth = rolling_mean(ci_da)
            
             
            media_smoothed_sig[var][modello][zona] = media_smoothed[var][modello][zona].where(
                np.abs(media_smoothed[var][modello][zona]) > ci_smooth)
            
            percentile_5 = np.percentile(tabella_dati, 5, axis=0)
            percentile_95 = np.percentile(tabella_dati, 95, axis=0)
            
            if var in ['pr', 'wo']:
                percentile_5 = percentile_5 * 86400
                percentile_95 = percentile_95 * 86400
           
            
            percentile_5_da = xr.DataArray(percentile_5, dims=['year'], coords={'year': anni})
            percentile_95_da = xr.DataArray(percentile_95, dims=['year'], coords={'year': anni})
            
            percentile_5_smooth = rolling_mean(percentile_5_da)
            percentile_95_smooth = rolling_mean(percentile_95_da)
            
            percentili[var][modello][zona] = {
                'min': percentile_5_smooth,   
                'max': percentile_95_smooth   
            }









variables_to_plot = ['rsst', 'pr', 'tauuo', 'uo', 'wo'] 

scale_dict = {
    'tauuo': 1e3,    
    'uo':    1e2,    
    'wo':    1e2,    
}

ylabel_dict = {
    'rsst':  r'RSST (°C)',
    'pr':    r'Precip (mm/day)',
    'tauuo': r'$\tau_x$ ($\times10^{-3}$ N/m²)',
    'uo':    r'$u$ ($\times10^{-2}$ m/s)',
    'wo':    r'$w$ ($\times10^{-2}$ m/s)',
}

colors = {
    'IPSL':    'red',
    'IPSL-FA': 'blue',
}

plt.rcParams.update({
    'font.family':       'sans-serif',
    'font.size':         20,
    'axes.labelsize':    20,
    'xtick.labelsize':   20,
    'ytick.labelsize':   20,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth':    0.8,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
})



fig = plt.figure(figsize=(22, 9))
gs = GridSpec(2, 5, figure=fig, hspace=0.15, wspace=0.15,
              left=0.06, right=0.99, top=0.93, bottom=0.08)
axes = np.array([[fig.add_subplot(gs[j, i]) for i in range(5)] for j in range(2)])


ylims = {
    0: (-1,   2),
    1: (-1.5,   5.5),
    2: (-6,   14),
    3: (-10.0,   25.0),
    4: (-4.5,  4),
}


for j, box_name in enumerate(['WEP', 'EEP']):
    for i, var in enumerate(variables_to_plot):
        ax = axes[j, i]
        scale = scale_dict.get(var, 1)
        
        for model in ['IPSL', 'IPSL-FA']:
            if box_name not in media_smoothed[var][model]:
                continue
            
            mean = media_smoothed[var][model][box_name]
            mean_sig = media_smoothed_sig[var][model][box_name]
            years = mean.year.values
            color = colors[model]
            
            # SHADING: Percentili 5-95%
            percentile_min = percentili[var][model][box_name]['min']
            percentile_max = percentili[var][model][box_name]['max']
            ax.fill_between(years, 
                           percentile_min * scale, 
                           percentile_max * scale,
                           color=color, alpha=0.15, zorder=1)
            
            # line : mean
            ax.plot(years, mean.values * scale,
                   color=color, linewidth=1.5, label=model, zorder=3)
            
            # bigger line: mean significative 
            sig_mask = ~np.isnan(mean_sig.values)
            if np.any(sig_mask):
                ax.plot(years[sig_mask], mean_sig.values[sig_mask] * scale,
                       color=color, linewidth=4, alpha=0.9, zorder=4)
        
        ax.set_ylim(ylims[i])
        
        if i == 0:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
        else:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
        
        ax.axvspan(2005, 2034, color='silver', alpha=0.3, lw=0, zorder=0)
        ax.axvspan(2070, 2099, color='silver', alpha=0.3, lw=0, zorder=0)
        ax.grid(True, alpha=0.2, linestyle=':')
        
        if j == 0:
            ax.set_title(ylabel_dict[var], fontsize=24)
            ax.set_xticklabels([])
            ax.tick_params(axis='x', length=0)
            
            # Legend
            if i == 0:
                legend_elements = []
                for model in ['IPSL', 'IPSL-FA']:
                    color = colors[model]

                    legend_elements.append(Patch(facecolor=color, alpha=0.15, 
                                               label=f'{model} (5th-95th perc.)'))
                    
                    legend_elements.append(Line2D([0], [0], color=color, linewidth=1.5, 
                                                 label=f'{model}'))

                
                ax.legend(handles=legend_elements, loc='upper left', fontsize=18, frameon=False)
        
        if j == len(['WEP', 'EEP']) - 1:
            ax.set_xlabel('Year', fontsize=22)
            ax.tick_params(axis='x', pad=10)
        
        if i == 0:
            ax.set_ylabel(box_name, fontsize=22)

for ax, label in zip(axes.T.flat, 'abcdefghij'):
    ax.text(0.0, 1.1, f'{label})', transform=ax.transAxes,
            fontsize=22, va='top', ha='left')
    ymin = ax.get_ylim()[0]
    ax.plot([1950, 1979], [ymin, ymin], color='black', linewidth=4,
            solid_capstyle='butt', clip_on=False, zorder=5)
    ax.axhline(0,   color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

fig.suptitle("Projected equatorial Pacific changes",
             fontsize=24, y=1.05)

# plt.savefig('/home/astoppel/figure/cambiamenti/ipsl_fa_time_series_anomaly.pdf',
#             bbox_inches='tight', facecolor='white')
plt.show()






