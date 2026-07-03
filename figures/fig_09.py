#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.



import xarray as xr
import xesmf as xe
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch




#### Grid and mask

file_nemo = xr.open_dataset('/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-r4_merged_grid_T.nc').sel(time_counter=slice('1950-01-01', '1950-02-01'))
area_nemo = file_nemo['cell_area']

mask = xr.open_dataset('/data/astoppel/Nudging_Example/ORCA1_subbasins.nc').glomsk.to_dataset(name='msk') #lon [-180,180]

mask_Trop = mask.where((mask.nav_lat <= 30) & (mask.nav_lat >= -30))

regions = {
    'EP': mask.where(((mask.nav_lat <= 5) & (mask.nav_lat >= -5)) &
                     ((mask.nav_lon >= -180) & (mask.nav_lon <= -80))),
    'WP': mask.where(((mask.nav_lat <= 5) & (mask.nav_lat >= -5)) &
                     ((mask.nav_lon >= 110) & (mask.nav_lon <= 180)))}




Present_period = slice('1950-01-01', '1979-12-31')




####################################################
#### Part one : Preprocessing the variables
####################################################




#### Functions

def apply_mask(da, mask):
    
    return (da * 0 + da.where(mask == 1))
    

def concat_historical_ssp(variables_histo, variables_ssp):
    
    time_series_TP = {}  
    
    for var in variables_histo.keys():
        time_series_TP[var] = {}
        print(var)
        for model in variables_histo[var]['var_names'].keys():
            
            if model == 'IPSL':
                path_hist = variables_histo[var]['path_ipsl']
                path_ssp  = variables_ssp[var]['path_ipsl']
                time_dim_hist = 'time'
                time_dim_ssp = 'time'

            elif model == 'IPSL-FA':
                path_hist = variables_histo[var]['path_ipsl_FA']
                path_ssp  = variables_ssp[var]['path_ipsl_FA']
                time_dim_hist = 'time_counter'
                time_dim_ssp = 'time_counter'

            varname = variables_histo[var]['var_names'][model]
            
            ds_hist = xr.open_dataset(path_hist)
            ds_ssp = xr.open_dataset(path_ssp)
            
            da_hist = ds_hist[varname]
            da_ssp = ds_ssp[varname]
            
            if model == 'IPSL-FA':
                da_hist = da_hist.rename({time_dim_hist: 'time'})
                da_ssp = da_ssp.rename({time_dim_ssp: 'time'})
            
            da_hist = da_hist.sel(time=slice('1950-01-01', '2014-12-31'))
            da_ssp = da_ssp.sel(time=slice('2015-01-01', '2099-12-31'))
            
            da_all = xr.concat([da_hist, da_ssp], dim='time')
            
            regridder = xe.Regridder(da_all, file_nemo, 'bilinear', reuse_weights=False, ignore_degenerate=True)
            var_reg = regridder(da_all)
            
            da_all_TP = apply_mask(var_reg, mask_Trop.msk)
            
            time_series_TP[var][model] = da_all_TP
            
            ds_hist.close()
            ds_ssp.close()
    
    return time_series_TP




variables_histo = {
    'tos': { #SST
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_historical_MEM_gn_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-MEM_merged_grid_T.nc',
        'var_names': {
            'IPSL': 'tos',
            'IPSL-FA': 'tos'
        }
    },
    'hfls': { #LH
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/hfls/hfls_Ayr_IPSL-CM6A-LR_historical_MEM_gr_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_historical_annual_means.nc',
        'var_names': {
            'IPSL': 'hfls',
            'IPSL-FA': 'flat'
        }
    },
    'hfss': { #SH
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/hfss/hfss_Ayr_IPSL-CM6A-LR_historical_MEM_gr_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_historical_annual_means.nc',
        'var_names': {
            'IPSL': 'hfss',
            'IPSL-FA': 'sens'
        }
    },
    'rlds': { #LWD
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/rlds/rlds_Ayr_IPSL-CM6A-LR_historical_MEM_gr_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_historical_annual_means.nc',
        'var_names': {
            'IPSL': 'rlds',
            'IPSL-FA': 'LWdnSFC'
        }
    },
    'rlus': { #LWU
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/rlus/rlus_Ayr_IPSL-CM6A-LR_historical_MEM_gr_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_historical_annual_means.nc',
        'var_names': {
            'IPSL': 'rlus',
            'IPSL-FA': 'LWupSFC'
        }
    },
    'rsds': { #SWD
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/rsds/rsds_Ayr_IPSL-CM6A-LR_historical_MEM_gr_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_historical_annual_means.nc',
        'var_names': {
            'IPSL': 'rsds',
            'IPSL-FA': 'SWdnSFC'
        }
    },
    'rsus': { #SWU
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/historical/CMIP6/rsus/rsus_Ayr_IPSL-CM6A-LR_historical_MEM_1950-2014.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_historical_annual_means.nc',
        'var_names': {
            'IPSL': 'rsus',
            'IPSL-FA': 'SWupSFC'
        }
    }
}




variables_ssp = {
    'tos': { 
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tos/tos_Oyr_IPSL-CM6A-LR_ssp585_MEM_gn_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-ssp585-MEM_merged_grid_T.nc',
        'var_names': {
            'IPSL': 'tos',
            'IPSL-FA': 'tos'
        }
    },
    'hfls': { 
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/hfls/hfls_Ayr_IPSL-CM6A-LR_ssp585_MEM_gr_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_ssp585_annual_means.nc',
        'var_names': {
            'IPSL': 'hfls',
            'IPSL-FA': 'flat'
        }
    },
    'hfss': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/hfss/hfss_Ayr_IPSL-CM6A-LR_ssp585_MEM_gr_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_ssp585_annual_means.nc',
        'var_names': {
            'IPSL': 'hfss',
            'IPSL-FA': 'sens'
        }
    },
    'rlds': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/rlds/rlds_Ayr_IPSL-CM6A-LR_ssp585_MEM_gr_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_ssp585_annual_means.nc',
        'var_names': {
            'IPSL': 'rlds',
            'IPSL-FA': 'LWdnSFC'
        }
    },
    'rlus': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/rlus/rlus_Ayr_IPSL-CM6A-LR_ssp585_MEM_gr_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_ssp585_annual_means.nc',
        'var_names': {
            'IPSL': 'rlus',
            'IPSL-FA': 'LWupSFC'
        }
    },
    'rsds': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/rsds/rsds_Ayr_IPSL-CM6A-LR_ssp585_MEM_gr_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_ssp585_annual_means.nc',
        'var_names': {
            'IPSL': 'rsds',
            'IPSL-FA': 'SWdnSFC'
        }
    },
    'rsus': {
        'path_ipsl': '/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/rsus/rsus_Ayr_IPSL-CM6A-LR_ssp585_MEM_gr_2015_2099.nc',
        'path_ipsl_FA': '/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/histmth/S3-tau200-MEM_ssp585_annual_means.nc',
        'var_names': {
            'IPSL': 'rsus',
            'IPSL-FA': 'SWupSFC'
        }
    }
}




time_series_trop_pacific = concat_historical_ssp(variables_histo, variables_ssp)


# hfls= Surface upward latent heat w/m2 $\Rightarrow$ sign changed by *-1   
# hfss= Surface upward sensible heat w/m2 $\Rightarrow$ sign changed by *-1 
# 
# rlds= Surface downwelling long wave radiation w/m2 $\Rightarrow$  sign same  
# rlus= Surface upwelling long wave radiation w/m2 $\Rightarrow$ sign changed by *-1
# 
# rsds= Surface downwelling shortwave radiation w/m2 $\Rightarrow$ sign same  
# rsus= Surface upwelling shortwave radiationn w/m2 $\Rightarrow$ sign changed by *-1
# 
# tos= Sea Surface Temperature $\Rightarrow$ sign same  
# rsst= relative Sea Surface Temperature $\Rightarrow$ sign same   



#some variables has to be changed sign to respect the convection

# IPSL
for var in ['hfls', 'hfss', 'rsus', 'rlus']:
    if var in time_series_trop_pacific and 'IPSL' in time_series_trop_pacific[var]:
        time_series_trop_pacific[var]['IPSL'] = -1 * time_series_trop_pacific[var]['IPSL']

# IPSL-FA
for var in ['rsus', 'rlus']:
    if var in time_series_trop_pacific and 'IPSL-FA' in time_series_trop_pacific[var]:
        time_series_trop_pacific[var]['IPSL-FA'] = -1 * time_series_trop_pacific[var]['IPSL-FA']




#### dictionary to save present climatology and change

present_clim, change = {}, {}

for var in time_series_trop_pacific.keys():
    present_clim[var] = {}
    change[var] = {}
    
    for model in time_series_trop_pacific[var].keys():

        da_trop = time_series_trop_pacific[var][model]
        
        present_clim[var][model] = da_trop.sel(time=Present_period).mean("time")
        
        change[var][model]       = da_trop - present_clim[var][model]  




#### computation of Q net and Oceanic processes 

def delta_qnet(change, model):
    return (
          change['rsds'][model]
        + change['rsus'][model]
        + change['rlds'][model]
        + change['rlus'][model]
        + change['hfss'][model]
        + change['hfls'][model]
    )

models = ['IPSL-FA', 'IPSL']  


delta_D0 = {} 

for model in models:

    delta_D0[model] = - delta_qnet(change,model)




#### computing delta net components, alpha 

delta_LH_o, delta_LH_f = {}, {}

alpha, alpha_LH  = {}, {}

delta_F = {} 

delta_SW, delta_SH, delta_LW = {}, {}, {}

for model in models:
    
    delta_SW[model]  = change['rsds'][model] + change['rsus'][model]
    delta_LW[model]  = change['rlds'][model] + change['rlus'][model]  
    delta_SH[model]  = change['hfss'][model]
    
    # LHF: The ocean part is calculated as γ1(0.06) × present day LH × SST change 
    # alpha_lh positive,  dQ = dQ_f - alpha * dT -> it's a neg fb, has to be a cooling
    
    alpha_LH[model] =  -1 * 0.06 * (present_clim['hfls'][model]) 
    delta_LH_o[model] = alpha_LH[model]  * change['tos'][model] 
    delta_LH_f[model] = change['hfls'][model] + delta_LH_o[model]  
    
    alpha[model] =  alpha_LH[model]  
    
    delta_F[model] = delta_SW[model] + delta_LW[model] + delta_SH[model]  + delta_LH_f[model] 




#### Relative Change 

change_mean, rel_change  = {}, {}

for var in change.keys():
    change_mean[var] = {}
    rel_change[var] = {} 
    for model in change[var].keys():
            change_mean[var][model]= change[var][model].weighted(area_nemo.fillna(0.)).mean(dim=['y', 'x'], skipna=True)
            rel_change[var][model] = change[var][model] - change_mean[var][model]
            

delta_D0_mean, delta_D0_rel  = {}, {} 
delta_F_mean, delta_F_rel = {}, {} 
alpha_mean, alpha_rel = {}, {} 

delta_SH_rel, delta_LH_f_rel = {}, {} 
delta_LW_rel, delta_SW_rel = {}, {} 

delta_T_mean, delta_T_rel, feedback_ocean = {}, {}, {}


for model in models:
    
    #relative change oceanic component 
    delta_D0_mean[model] =  -delta_qnet(change_mean,model)
    delta_D0_rel[model]  =  -delta_qnet(rel_change,model)

    #relative change forcing 
    delta_F_mean[model] = delta_F[model].weighted(area_nemo.fillna(0.)).mean(dim=['y', 'x'], skipna=True)
    delta_F_rel[model]  = delta_F[model] - delta_F_mean[model]

    delta_SH_rel[model]   = delta_SH[model]   - delta_SH[model].weighted(area_nemo.fillna(0.)).mean(dim=['y', 'x'], skipna=True)
    delta_LH_f_rel[model] = delta_LH_f[model] - delta_LH_f[model].weighted(area_nemo.fillna(0.)).mean(dim=['y', 'x'], skipna=True)
    delta_SW_rel[model]   = delta_SW[model]   - delta_SW[model].weighted(area_nemo.fillna(0.)).mean(dim=['y', 'x'], skipna=True)
    delta_LW_rel[model]   = delta_LW[model]   - delta_LW[model].weighted(area_nemo.fillna(0.)).mean(dim=['y', 'x'], skipna=True)

    # relative alpha
    alpha_mean[model] = alpha[model].weighted(area_nemo.fillna(0.)).mean(dim=['y', 'x'], skipna=True)
    alpha_rel[model]  = alpha[model] - alpha_mean[model]

    delta_T_mean[model] = (delta_F_mean[model] / alpha_mean[model] + delta_D0_mean[model]/ alpha_mean[model]) 

    star = alpha_rel[model] / alpha_mean[model]
    
    #first order approx 
    feedback_ocean[model] = -1 * star * (delta_F_rel[model] + delta_D0_rel[model]) + (alpha_rel[model]*star -1 * alpha_rel[model]) * delta_T_mean[model]







#### computing the three main contributors
cont_forcing  = {}
cont_ocean    = {}
cont_feedback = {}

cont_delta_SW_rel, cont_delta_LW_rel, cont_delta_SH_rel, cont_delta_LH_rel = {}, {}, {}, {}


for model in models:
    
    cont_forcing[model]  = delta_F_rel[model] / alpha_mean[model]  
    cont_ocean[model]    = delta_D0_rel[model]  / alpha_mean[model]  
    cont_feedback[model] = feedback_ocean[model]/ alpha_mean[model] 
    
    cont_delta_SW_rel[model] = delta_SW_rel[model]/ alpha_mean[model]
    cont_delta_LW_rel[model] = delta_LW_rel[model]/ alpha_mean[model]
    cont_delta_SH_rel[model] = delta_SH_rel[model] / alpha_mean[model]
    cont_delta_LH_rel[model] = delta_LH_f_rel[model] / alpha_mean[model]




#### mean values into the EEP and WEP boxes 

area_weights = area_nemo.fillna(0.)

cont_forcing_reg      = {}
cont_ocean_reg        = {}
cont_feedback_reg     = {}

cont_SW_reg = {}
cont_LW_reg = {}
cont_SH_reg = {}
cont_LH_reg = {}

delta_T_rel_reg_online = {}

for model in models:
    
    cont_forcing_reg[model]      = {}
    cont_ocean_reg[model]        = {}
    cont_feedback_reg[model]     = {}

    cont_SW_reg[model] = {}
    cont_LW_reg[model] = {}
    cont_SH_reg[model] = {}
    cont_LH_reg[model] = {}

    delta_T_rel_reg_online[model] = {}


    for region_key, region_mask in regions.items():
        mask = region_mask.msk

        # componenti radiative / turbolente (relative)  
        cont_SW_reg[model][region_key] = (
            apply_mask(cont_delta_SW_rel[model], mask)
            .weighted(area_weights).mean(dim=['y', 'x'], skipna=True)
        )

        cont_LW_reg[model][region_key] = (
            apply_mask(cont_delta_LW_rel[model], mask)
            .weighted(area_weights).mean(dim=['y', 'x'], skipna=True)
        )

        cont_SH_reg[model][region_key] = (
            apply_mask(cont_delta_SH_rel[model], mask)
            .weighted(area_weights).mean(dim=['y', 'x'], skipna=True)
        )

        cont_LH_reg[model][region_key] = (
            apply_mask(cont_delta_LH_rel[model], mask)
            .weighted(area_weights).mean(dim=['y', 'x'], skipna=True)
        )

        #  forcing / ocean / feedback (relative) 
        cont_forcing_reg[model][region_key] = (
            apply_mask(cont_forcing[model], mask)
            .weighted(area_weights).mean(dim=['y', 'x'], skipna=True)
        )

        cont_ocean_reg[model][region_key] = (
            apply_mask(cont_ocean[model], mask)
            .weighted(area_weights).mean(dim=['y', 'x'], skipna=True)
        )

        cont_feedback_reg[model][region_key] = (
            apply_mask(cont_feedback[model], mask)
            .weighted(area_weights).mean(dim=['y', 'x'], skipna=True)
        )

        # --- ΔT relativo (online) ---
        delta_T_rel_reg_online[model][region_key] = (
            apply_mask(rel_change['tos'][model], mask)
            .weighted(area_weights).mean(dim=['y', 'x'], skipna=True)
        )




#### buildind dictionary

terms = {}
delta_T_rel_reg = {}

for model in ['IPSL-FA', 'IPSL']:
    terms[model] = {}
    delta_T_rel_reg[model] = {}

    for region_key in regions.keys():

        #  ΔT relativo regionale (ricostruito)  
        delta_T_rel_reg[model][region_key] = (
            cont_forcing_reg[model][region_key] +
            cont_ocean_reg[model][region_key] +
            cont_feedback_reg[model][region_key]
        )

        # Dictionary relative terms
        terms[model][region_key] = {
            'delta_online': delta_T_rel_reg_online[model][region_key],
            'delta_reconstructed': delta_T_rel_reg[model][region_key],
            'ocean': cont_ocean_reg[model][region_key],
            'feedback': cont_feedback_reg[model][region_key],
            'forcing': cont_forcing_reg[model][region_key],
            'SW': cont_SW_reg[model][region_key],
            'LW': cont_LW_reg[model][region_key],
            'SH': cont_SH_reg[model][region_key],
            'LH': cont_LH_reg[model][region_key]
        }





####################################################
#### Part three : Plot
####################################################




def draw_split_arrows(ax, x_pos, termini_group, vals, arrow_kw, x_offset=0.04, shaded=False):
    """Frecce cumulate: positive a sinistra, negative a destra di x_pos."""
    pos_terms = [t for t in termini_group if vals[t] >= 0]
    neg_terms = [t for t in termini_group if vals[t] < 0]
    sum_pos = sum(vals[t] for t in pos_terms)
 
    cumsum = 0.0
    for t in pos_terms:
        v = vals[t]
        if abs(v) > 1e-6:
            if not shaded:
                # Versione non shaded
                ax.arrow(x_pos - x_offset, cumsum, 0, v, 
                        facecolor='black', edgecolor='black', linewidth=4, **arrow_kw)
                ax.arrow(x_pos - x_offset, cumsum, 0, v, 
                        facecolor=colori_termini[t], edgecolor=colori_termini[t], 
                        linewidth=2, **arrow_kw)
            else:
                # Versione shaded - crea una copia di arrow_kw senza alpha se presente
                arrow_kw_shaded = arrow_kw.copy()
                
                ax.arrow(x_pos - x_offset, cumsum, 0, v, 
                        facecolor='white', edgecolor='gray', linewidth=4, **arrow_kw_shaded)

                arrow_kw_shaded['alpha'] = 1 #0.6

                ax.arrow(x_pos - x_offset, cumsum, 0, v, 
                        facecolor=colori_termini[t], edgecolor=colori_termini[t], 
                        linewidth=2, **arrow_kw_shaded)
        cumsum += v
 
    cumsum = sum_pos
    for t in neg_terms:
        v = vals[t]
        if abs(v) > 1e-6:
            if not shaded:
                # Versione non shaded
                ax.arrow(x_pos + x_offset, cumsum, 0, v, 
                        facecolor='black', edgecolor='black', linewidth=4, **arrow_kw)
                
                ax.arrow(x_pos + x_offset, cumsum, 0, v, 
                        facecolor=colori_termini[t], edgecolor=colori_termini[t], 
                        linewidth=2, **arrow_kw)
            else:
                # Versione shaded 
                arrow_kw_shaded = arrow_kw.copy()
                
                ax.arrow(x_pos + x_offset, cumsum, 0, v, 
                        facecolor='white', edgecolor='gray', linewidth=4, **arrow_kw_shaded)
                
                arrow_kw_shaded['alpha'] = 1 # 0.6
                
                ax.arrow(x_pos + x_offset, cumsum, 0, v, 
                        facecolor=colori_termini[t], edgecolor=colori_termini[t], 
                        linewidth=2, **arrow_kw_shaded)
                
        cumsum += v




# Labels LaTeX
termini_labels = {
    'delta_online':        r"$\Delta T'$",
    'delta_reconstructed': r"$\Delta T_{ZL}'$",
    'forcing':             r"$\Delta Q_{for}'^*$",
    'ocean':               r"$\Delta O'^*$",
    'feedback':            r"$fdb'^*$",
    'SW':                  r"$\Delta SW'^*$",
    'LW':                  r"$\Delta LW'^*$",
    'SH':                  r"$\Delta SH'^*$",
    'LH':                  r"$\Delta LH_{for}'^*$",
    'residuo': r"$\Delta LW'^* + \Delta SH'^* + \Delta LH_{for}'^*$"
}

colori_termini = {
    'forcing':  '#009E73',
    'ocean':    '#F18F01',  
    'feedback': '#F433AB', 
    'SW':       '#fff897',
    'LW':       '#CAFFBF',      # Verde pastello (più brillante)
    'SH':       '#A0E7E5',      # Turchese pastello
    'LH':       '#FFB3D9',      # Rosa pastello
}

modelli  = ['IPSL-FA', 'IPSL']

color_model = {'IPSL': '#e06c6c', 'IPSL-FA': '#5b9bd5'}


 
bacini_k = {'WEP': 'WP', 'EEP': 'EP'}

# Axes limits
ylims_arr = {
    '2005–2034': [-0.3, 0.5],
    '2070–2099': [-0.9,  1.8] 
}

head_length_arr = {
    '2005–2034': 0.04,
    '2070–2099': 0.14,
}

plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize':11,
    'ytick.labelsize':11,
})


per_near   = slice('2005-01-01', '2034-12-31')
per_far    = slice('2070-01-01', '2099-12-31')

configs = [
    ('2005–2034', per_near, 'WEP', 'a)'),
    ('2070–2099', per_far,  'EEP', 'b)'),]

x_positions = {
    'IPSL': {
        'box_dt':     0.05,
        'box_online': 0.13,
        'arr_solid':  0.24,
        'arr_rad':    0.40,     
    },
    'IPSL-FA': {
        'box_dt':     0.55,     
        'box_online': 0.63,      
        'arr_solid':  0.74,      
        'arr_rad':    0.90,      
    },
}



box_width = 0.07
group_rad = ['SW', 'LW', 'SH', 'LH']
group_solid     = ['feedback', 'forcing', 'ocean']




fig, axs = plt.subplots(1, 2, figsize=(11, 4)) 

for ax, (periodo_label, periodo_slice, bacino, plabel) in zip(axs, configs):
    bk = bacini_k[bacino]
    

    arrow_kw = dict(
        length_includes_head=True,
        head_width=0.05,
        head_length=head_length_arr[periodo_label],
    )
    arrow_kw_rad = dict(
        length_includes_head=True,
        head_width=0.05,
        head_length=head_length_arr[periodo_label],
    )

    for model in modelli:
        xp    = x_positions[model]
        first = (model == 'IPSL-FA' and bacino == 'WEP')

        mean_dt     = terms[model][bk]['delta_reconstructed'].sel(time=periodo_slice).mean().item()
        mean_online = terms[model][bk]['delta_online'].sel(time=periodo_slice).mean().item()

        # Barra ΔT'
        ax.bar(xp['box_dt'], mean_dt, width=box_width,
               color=color_model[model], edgecolor='black', linewidth=0.8,
               label=f"$\\Delta T'$ ({model})" if first else None)

        # Barra ΔT_online (hatch)
        ax.bar(xp['box_online'], mean_online, width=box_width,
               color=color_model[model], edgecolor='black', linewidth=0.8,
               hatch='///', alpha=0.5,
               label=f"$\\Delta T_{{\\rm online}}'$ ({model})" if first else None)

        # Frecce group_solid
        vals_solid = {t: terms[model][bk][t].sel(time=periodo_slice).mean().item()
                      for t in group_solid}
        
        draw_split_arrows(ax, xp['arr_solid'], group_solid, vals_solid,
                          arrow_kw, x_offset=0.03)

        # Frecce radiativi
        vals_rad = {t: terms[model][bk][t].sel(time=periodo_slice).mean().item()
                    for t in group_rad}
        draw_split_arrows(ax, xp['arr_rad'], group_rad, vals_rad,
                          arrow_kw_rad, x_offset=0.03, shaded=True)

    # Separatori
    ax.axvline(0.50, color='black', linestyle='--', linewidth=0.9, alpha=0.4)  # Aumentato da 0.44
    ax.axhline(0,    color='black', linewidth=0.6, alpha=0.6)
    for x_sep in [0.18, 0.32, 0.68, 0.82]:  # Adattati proporzionalmente
        ax.axvline(x_sep, color='gray', linestyle=':', linewidth=0.5, alpha=0.3)

    ax.set_ylim(*ylims_arr[periodo_label])
    ax.set_xlim(0, 1.0)  # Aumentato da 0.92

    
    ax.set_xticks([
        np.mean([x_positions['IPSL']['box_dt'],    x_positions['IPSL']['arr_rad']]),
        np.mean([x_positions['IPSL-FA']['box_dt'], x_positions['IPSL-FA']['arr_rad']]),
    ])
    ax.set_xticklabels(['IPSL', 'IPSL-FA'], fontsize=12)
    ax.tick_params(axis='x', length=0)
    ax.grid(True, alpha=0.2, axis='y', linestyle='--')

    ax.set_title(f"{plabel} {bacino} — {periodo_label}", fontsize=14,
                 loc='left', x = -0.05, y=1.01)
    
    ax.set_ylabel(r'°C', fontsize=12)
 
# ── Legenda ───────────────────────────────────────────────────────────────────
bar_handles, bar_lbls = axs[0].get_legend_handles_labels()

arrow_handles = [
    Patch(facecolor=colori_termini[t], edgecolor='black', linewidth=0.6,
          label=termini_labels[t])
    for t in group_solid
] + [
    Patch(facecolor=colori_termini[t], edgecolor='gray', linewidth=0.6,
          alpha=1, label=termini_labels[t]) #alpha = 0.6 
    for t in group_rad
]


fig.legend(
    arrow_handles,
    [h.get_label() for h in arrow_handles],
    fontsize=10, frameon=True, ncol=3,
    bbox_to_anchor=(0.33, 0.9),
    handlelength=1.4, handletextpad=0.5,
    labelspacing=0.4, columnspacing=1.0,
)

# ── Legenda ΔT nel secondo pannello ──────────────────────────────────────────

dt_handles = [
    Patch(facecolor=color_model['IPSL'],    edgecolor='black', linewidth=0.8,
          label="$\\Delta T_{ZL}'$ IPSL"),
    Patch(facecolor=color_model['IPSL-FA'], edgecolor='black', linewidth=0.8,
          label="$\\Delta T_{ZL}'$ IPSL-FA"),
    Patch(facecolor=color_model['IPSL'],    edgecolor='black', linewidth=0.8,
          hatch='///', alpha=0.5, label="$\\Delta T'$ IPSL"),
    Patch(facecolor=color_model['IPSL-FA'], edgecolor='black', linewidth=0.8,
          hatch='///', alpha=0.5, label="$\\Delta T'$ IPSL-FA"),
]

axs[1].legend(
    dt_handles, [h.get_label() for h in dt_handles],
    fontsize=10, frameon=True, ncol=2,
    loc='lower left',
    handlelength=1.3, handletextpad=0.3,
    labelspacing=0.3, columnspacing=0.7,
)

plt.tight_layout()

# plt.savefig('/home/astoppel/figure/ZLi14/fig_arrows_WEPEEP_relTrop.pdf',
#             bbox_inches='tight', facecolor='white')
plt.show()






