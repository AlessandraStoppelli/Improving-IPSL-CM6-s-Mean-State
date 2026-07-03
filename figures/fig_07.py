#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.



import xarray as xr
import numpy as np
import xesmf as xe 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats




ref_period = ['1950-01-01', '1979-12-31']

future_periods = ['1980-01-01','2099-12-31']

members_FA = members_ipsl = ['r1','r2','r3','r4','r12','r13','r14','r15','r16','r18','r22','r23','r29','r30','r33']

regions_rsst = {
    'EP': {'lat_range': (-5, 5), 'lon_range': (180, 280)},
    'WP': {'lat_range': (-5, 5), 'lon_range': (110, 180)},
}




#Grids & Masks

file_nemo = xr.open_dataset('/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-r4_merged_grid_T.nc').sel(time_counter=slice('1950-01-01', '1950-02-01'))
area_nemo = file_nemo['cell_area']

file_regular = xr.open_dataset('/data/astoppel/CMIP6/ensamble_historical/ipsl_CMIP6_TOS_1921_2014.nc')
grid_regular = {"lon": file_regular["lon"], "lat": file_regular["lat"]}

mask = xr.open_dataset('/data/astoppel/Nudging_Example/ORCA1_subbasins.nc').glomsk.to_dataset(name='msk')

mask_eq = mask.where(
    ((mask.nav_lat <= 5) & (mask.nav_lat >= -5)) &
    (((mask.nav_lon >= 110) & (mask.nav_lon <= 180)) |
     ((mask.nav_lon >= -180) & (mask.nav_lon <= -80))))

nino34_4_mask = mask.where(
    (mask['nav_lat'] >= -5) & (mask['nav_lat'] <= 5) &
    ((mask['nav_lon'] >= 160) | (mask['nav_lon'] <= -120)))




####################################################
#### Part one : RSST CHANGE & VARIANCE DECOMPOSITION
####################################################




#### Functions

def load_member_sst(member, file_past, file_future, time_var='time_counter',
                     var_name='tos', start=None, end=None):

    ds_past = xr.open_dataset(file_past)
    ds_future = xr.open_dataset(file_future)

    sst_concat = xr.concat([ds_past[var_name], ds_future[var_name]], dim=time_var)

    if time_var != 'time_counter':
        sst_concat = sst_concat.rename({time_var: 'time_counter'})

    if start is not None and end is not None:
        sst_concat = sst_concat.sel(time_counter=slice(start, end))

    sst_concat = sst_concat.expand_dims(member=[member])

    return sst_concat


def build_ensemble(members, base_path_past, base_path_future,
                    file_pattern_past="S3-tau200-r{m}_merged_grid_T.nc",
                    file_pattern_future="S3-ssp585-r{m}_merged_grid_T.nc",
                    time_var='time', var_name='tos', start=None, end=None):

    sst_list = []

    for m in members:
        file_past = base_path_past + "/" + file_pattern_past.format(m=m)
        file_future = base_path_future + "/" + file_pattern_future.format(m=m)

        sst_m = load_member_sst(
            member=m,
            file_past=file_past,
            file_future=file_future,
            time_var=time_var,
            var_name=var_name,
            start=start,
            end=end
        )

        sst_list.append(sst_m)

    ensemble = xr.concat(sst_list, dim='member')

    return ensemble


def compute_rsst_change_per_member(data_list, period1, period2, area=None):

    changes_all = []

    for m in range(len(data_list)):
        member_da = data_list[m]

        # 1. Seleziona i tropici
        tropical_mask = (member_da.nav_lat >= -30) & (member_da.nav_lat <= 30)
        tropical_da = member_da.where(tropical_mask, drop=True)

        if area is not None:
            tropical_area = area.where(tropical_mask, drop=True)

            data_p1 = tropical_da.sel(time_counter=slice(period1[0], period1[1]))
            trop_mean1 = data_p1.weighted(tropical_area.fillna(0.)).mean(dim=["y", "x"], skipna=True)
            trop_mean1 = trop_mean1.mean(dim='time_counter')

            data_p2 = tropical_da.sel(time_counter=slice(period2[0], period2[1]))
            trop_mean2 = data_p2.weighted(tropical_area.fillna(0.)).mean(dim=["y", "x"], skipna=True)
            trop_mean2 = trop_mean2.mean(dim='time_counter')

        else:
            data_p1 = tropical_da.sel(time_counter=slice(period1[0], period1[1]))
            trop_mean1 = data_p1.mean(dim=['nav_lat', 'nav_lon']).mean(dim='time_counter')

            data_p2 = tropical_da.sel(time_counter=slice(period2[0], period2[1]))
            trop_mean2 = data_p2.mean(dim=['nav_lat', 'nav_lon']).mean(dim='time_counter')

        # 2. Calcola RSST per ciascun periodo
        rsst1 = member_da.sel(time_counter=slice(period1[0], period1[1])).mean(dim='time_counter') - trop_mean1
        rsst2 = member_da.sel(time_counter=slice(period2[0], period2[1])).mean(dim='time_counter') - trop_mean2

        # 3. Calcola il cambiamento di RSST
        rsst_change = rsst2 - rsst1

        changes_all.append(rsst_change.values)

    # Crea il DataArray finale
    changes_array = np.array(changes_all)

    changes_da = xr.DataArray(
        changes_array,
        dims=['member', 'y', 'x'],
        coords={
            'member': range(len(data_list)),
            'y': data_list[0]['y'],
            'x': data_list[0]['x'],
            'nav_lon': data_list[0].nav_lon,
            'nav_lat': data_list[0].nav_lat
        }
    )

    return changes_da



def regrid_change_data(change_data, grid_target, method="bilinear"):
    """Regridda i dati di cambiamento sulla griglia target"""
    change_ds = change_data.to_dataset(name='change')
    
    regridder = xe.Regridder(
        change_ds, grid_target,
        method=method,
        reuse_weights=False,
        ignore_degenerate=True
    )
    
    change_regridded = regridder(change_ds)['change']
    return change_regridded

def regrid_dictionary(rsst_dict, grid_target):
    return {k: regrid_change_data(v, grid_target)
            for k, v in rsst_dict.items()}


def extract_region_change_rsst_rg(data, lat_range, lon_range):

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range

    return (
        data
        .sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        .mean(dim=["lat", "lon"])
    )


def get_CI_from_list(values_list):
    values = np.array(values_list)
    std = np.std(values, ddof=1)
    Var = np.var(values, ddof=1)
    n = len(values)
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    margin_of_error = t_critical * (std / np.sqrt(n))
    chi2_lower = stats.chi2.ppf(0.975, n-1)
    chi2_upper = stats.chi2.ppf(0.025, n-1)
    ci_var_lower = (n-1) * Var / chi2_lower
    ci_var_upper = (n-1) * Var / chi2_upper
    return Var, std, ci_var_lower, ci_var_upper, margin_of_error




#### upload each member of IPSL and IPSL-FA

sst_ensemble_FA = build_ensemble(
    members=members_FA,
    base_path_past="/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean",
    base_path_future="/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Tgrid_yearmean",
    file_pattern_past="S3-tau200-{m}_merged_grid_T.nc",
    file_pattern_future="S3-ssp585-{m}_merged_grid_T.nc",
    time_var='time_counter',
    var_name='tos',
    start='1950-01-01',
    end='2099-12-31'
)
sst_ensemble_ipsl = build_ensemble(
    members=members_ipsl,
    base_path_past="/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tos",
    base_path_future="/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tos",
    file_pattern_past="tos_Oyr_IPSL-CM6A-LR_historical_{m}_gn_1950-2014.nc",
    file_pattern_future="tos_Oyr_IPSL-CM6A-LR_ssp585_{m}_gn_2015_2099.nc",
    time_var='time',
    var_name='tos',
    start='1950-01-01',
    end='2099-12-31'
)

fa_members_list   = [sst_ensemble_FA.isel(member=i)   for i in range(len(members_FA))]
ipsl_members_list = [sst_ensemble_ipsl.isel(member=i) for i in range(len(members_ipsl))]





#### Compute RSST change 

    
rsst_change_fa = compute_rsst_change_per_member(fa_members_list, ref_period, future_periods, area=area_nemo)
rsst_change_fa_mean = rsst_change_fa.mean(dim='member')

rsst_change_ipsl = compute_rsst_change_per_member(ipsl_members_list, ref_period, future_periods, area=area_nemo)
rsst_change_ipsl_mean = rsst_change_ipsl.mean(dim='member')

rsst_changes = {'FA': rsst_change_fa_mean, 'IPSL': rsst_change_ipsl_mean}




#### Regridd per member
#### Regridd per mean 

# Dictionary per member 
change_rsst_ipsl_dict = {}
change_rsst_FA_dict = {}

for i, m in enumerate(rsst_change_ipsl.member):
    nome = members_ipsl[i]
    change_rsst_ipsl_dict[nome] = rsst_change_ipsl.sel(member=m)
    change_rsst_FA_dict[nome] = rsst_change_fa.sel(member=m)

# Regrid of members
change_rsst_ipsl_regridded_memb = regrid_dictionary(change_rsst_ipsl_dict, grid_regular)
change_rsst_FA_regridded_memb = regrid_dictionary(change_rsst_FA_dict, grid_regular)

# Regrid of the mean
rsst_fa_reg = regrid_change_data(rsst_changes["FA"], grid_regular)
rsst_ipsl_reg = regrid_change_data(rsst_changes["IPSL"], grid_regular)

# rsst_fa_reg[:, 73] = rsst_fa_reg[:, 72]
# rsst_ipsl_reg[:, 73] = rsst_ipsl_reg[:, 72]





#extract rsst change for regions 

change_rsst = {}
datasets = {
    "IPSL":    change_rsst_ipsl_regridded_memb,
    "IPSL-FA": change_rsst_FA_regridded_memb,
}

for reg_name, reg_coords in regions_rsst.items():
    change_rsst[reg_name] = {}
    for model_name, model_dict in datasets.items():
        change_rsst[reg_name][model_name] = {}
        for mem, data in model_dict.items():
            change_rsst[reg_name][model_name][mem] = extract_region_change_rsst_rg(
                data, reg_coords["lat_range"], reg_coords["lon_range"]
            )




#### compute results of cor cov... 

eep_ipsl    = [change_rsst["EP"]["IPSL"][m]    for m in members_ipsl]
wep_ipsl    = [change_rsst["WP"]["IPSL"][m]    for m in members_ipsl]
eep_ipsl_fa = [change_rsst["EP"]["IPSL-FA"][m] for m in members_FA]
wep_ipsl_fa = [change_rsst["WP"]["IPSL-FA"][m] for m in members_FA]

g_ipsl = [change_rsst["WP"]["IPSL"][m]    - change_rsst["EP"]["IPSL"][m]    for m in members_ipsl]
g_fa   = [change_rsst["WP"]["IPSL-FA"][m] - change_rsst["EP"]["IPSL-FA"][m] for m in members_FA]

COR_ipsl = np.corrcoef(wep_ipsl, eep_ipsl)[0, 1]
COR_fa   = np.corrcoef(wep_ipsl_fa, eep_ipsl_fa)[0, 1]

eep_ipsl_var, eep_ipsl_std, *_ = get_CI_from_list(eep_ipsl)
wep_ipsl_var, wep_ipsl_std, *_ = get_CI_from_list(wep_ipsl)
eep_fa_var,   eep_fa_std,   *_ = get_CI_from_list(eep_ipsl_fa)
wep_fa_var,   wep_fa_std,   *_ = get_CI_from_list(wep_ipsl_fa)

sigma_g_ipsl = np.sqrt(wep_ipsl_std**2 + eep_ipsl_std**2 - 2*wep_ipsl_std*eep_ipsl_std*COR_ipsl)
sigma_g_fa   = np.sqrt(wep_fa_std**2   + eep_fa_std**2   - 2*wep_fa_std*eep_fa_std*COR_fa)

VG_ipsl  = sigma_g_ipsl**2
COV_ipsl = wep_ipsl_std * eep_ipsl_std * COR_ipsl
terms_ipsl = np.array([VG_ipsl, wep_ipsl_std**2, eep_ipsl_std**2, -2*COV_ipsl])

VG_fa  = sigma_g_fa**2
COV_fa = wep_fa_std * eep_fa_std * COR_fa
terms_fa = np.array([VG_fa, wep_fa_std**2, eep_fa_std**2, -2*COV_fa])




####################################################
#### Part two : Correlations between Wind stress and SST
####################################################




#### Functions

def build_ensemble_s2(
    members, base_path_past, base_path_future,
    file_pattern_past, file_pattern_future,
    period,
    time_var='time', var_name='tos'
):
    ds_list = []
    for m in members:  
        ds_past   = xr.open_dataset(f"{base_path_past}/{file_pattern_past.format(m=m)}",   chunks={time_var: 12})
        ds_future = xr.open_dataset(f"{base_path_future}/{file_pattern_future.format(m=m)}", chunks={time_var: 12})
        ds_concat = xr.concat([ds_past[var_name], ds_future[var_name]], dim=time_var)
        if time_var in ds_concat.dims and time_var != 'time':
            ds_concat = ds_concat.rename({time_var: 'time'})
        ds_concat_msk = ds_concat.where(mask_eq.msk == 1)
        ds_msk_x = ds_concat_msk.weighted(area_nemo.fillna(0.)).mean(dim=['y'], skipna=True)
        if var_name == 'tauuo':
            ds_concat_msk_tau = ds_concat.where(nino34_4_mask.msk == 1)
            ds_msk_x_tau = ds_concat_msk_tau.weighted(area_nemo.fillna(0.)).mean(dim=['x','y'], skipna=True)
            ds_msk_x_2 = ds_msk_x_tau.sel(time=period)
        else:
            ds_msk_x_2 = ds_msk_x.sel(time=period)
        ds_msk_x_2 = ds_msk_x_2.expand_dims(member=[m]).compute()
        ds_list.append(ds_msk_x_2)
    return xr.concat(ds_list, dim='member')




period = slice(future_periods[0], future_periods[1])

sst_ens_FA_s2 = build_ensemble_s2(
    members_FA,
    "/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/tos_monthly",
    "/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Tgrid_yearmean/tos_monthly",
    "tos-{m}_historical_monthly_means.nc",
    "tos-{m}_ssp585_monthly_means.nc",
    period=period,
    time_var='time_counter'
)

sst_ens_ipsl_s2 = build_ensemble_s2(
    members_ipsl,
    "/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tos/Omon",
    "/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tos/Omon",
    "tos_Omon_IPSL-CM6A-LR_historical_{m}_gn_195001-201412.nc",
    "tos_Omon_IPSL-CM6A-LR_ssp585_{m}_gn_201501-210012.nc",
    period=period,
)

tauuo_ens_fa = build_ensemble_s2(
    members_FA,
    "/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Ugrid_yearmean/tauuo_monthly",
    "/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Ugrid_yearmean/tauuo_monthly",
    "tauuo-{m}_historical_monthly_means.nc",
    "tauuo-{m}_ssp585_monthly_means.nc",
    period=period,
    var_name='tauuo',
    time_var='time_counter'
)

tauuo_ens_ipsl = build_ensemble_s2(
    members_ipsl,
    "/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tauuo/Omon",
    "/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tauuo/Omon",
    "tauuo_Omon_IPSL-CM6A-LR_historical_{m}_gn_195001-201412.nc",
    "tauuo_Omon_IPSL-CM6A-LR_ssp585_{m}_gn_201501-210012.nc",
    period=period,
    var_name='tauuo',
)

sst_distance_ipsl_mem   = sst_ens_ipsl_s2 - sst_ens_ipsl_s2.mean(dim='member')
sst_distance_fa_mem     = sst_ens_FA_s2   - sst_ens_FA_s2.mean(dim='member')
tauuo_distance_ipsl_mem = tauuo_ens_ipsl  - tauuo_ens_ipsl.mean(dim='member')
tauuo_distance_fa_mem   = tauuo_ens_fa    - tauuo_ens_fa.mean(dim='member')

tau_ipsl_al, sst_ipsl_al = xr.align(tauuo_distance_ipsl_mem, sst_distance_ipsl_mem, join="inner")
corr_x_ipsl = xr.corr(tau_ipsl_al, sst_ipsl_al, dim=("time", "member"))

tau_fa_al, sst_fa_al = xr.align(tauuo_distance_fa_mem, sst_distance_fa_mem, join="inner")
corr_x_FA = xr.corr(tau_fa_al, sst_fa_al, dim=("time", "member"))




####################################################
#### Part three : Plot
####################################################




def multicolored_bar_combined(ax, terms, xpos, colors, model, bar_label=None):
    bottom_pos = bottom_neg = 0
    lab = bar_label
    for v, c in zip(terms, colors):
        if v >= 0:
            ax.bar(xpos, v, width, bottom=bottom_pos,
                   edgecolor="black", linewidth=0.8, color=c, alpha= 1,
                   label=lab if bottom_pos == 0 else None)
            bottom_pos += v
        else:
            ax.bar(xpos, v, width, bottom=bottom_neg,
                   edgecolor="black", linewidth=0.8, color=c,alpha= 1,
                   label=lab if bottom_neg == 0 else None)
            bottom_neg += v




fontsize_base = 20  
labelsize_base = 20  
fig, axes = plt.subplots(1, 2, figsize=(18, 5),  
                         gridspec_kw={'width_ratios': [1, 1]})
fig.subplots_adjust(wspace=0.35)
colors_ipsl = ['#754F5B',    '#21A179',   "#FFC300"]
labels_bar  = [  r"$Var(WEP)$",     r"$Var(EEP)$",     r"$-2\,\mathrm{Cov}(WEP,EEP)$"]
labels_var  = ["var(ZG) IPSL", "var(ZG) IPSL-FA"]
colors_var  = ['#e06c6c', '#5b9bd5']
x     = np.array([0.4, 0.65, 1.1, 1.35])
width = 0.25

arrow_kw = dict(length_includes_head=True, head_width=0.07, head_length=0.001)
ax1 = axes[0]
ax2 = axes[1]

# Prima erano dizionari indicizzati per periodo, ora sono variabili singole
terms_ipsl = terms_ipsl
terms_fa   = terms_fa

# Creare i patch per la legenda
legend_patches = [Rectangle((0, 0), 1, 1, facecolor=colors_var[0], edgecolor='black', linewidth=1),
                  Rectangle((0, 0), 1, 1, facecolor=colors_var[1], edgecolor='black', linewidth=1)] + \
                [Rectangle((0, 0), 1, 1, facecolor=colors_ipsl[i], edgecolor='black', linewidth=1) 
                 for i in range(len(colors_ipsl))]

legend_labels = labels_var + labels_bar

# Left Panel : variance decomposition
ax1.set_xlim(0, 1.5)
multicolored_bar_combined(ax1, [terms_ipsl[0]], x[0], ['#e06c6c'], 
                          model="ipsl", bar_label="")
multicolored_bar_combined(ax1, [terms_fa[0]],   x[2], ['#5b9bd5'], 
                          model="ipsl-fa", bar_label="")

#first arrow IPSL - Var(WEP)
ax1.arrow(x[1], 0, 0, terms_ipsl[1], facecolor='black', edgecolor='black', linewidth=4, **arrow_kw)
ax1.arrow(x[1], 0, 0, terms_ipsl[1], facecolor=colors_ipsl[0], edgecolor=colors_ipsl[0], linewidth=2, **arrow_kw)

# second arrow IPSL - Var(EEP)
ax1.arrow(x[1], terms_ipsl[1], 0, terms_ipsl[2], facecolor='black', edgecolor='black', linewidth=4, **arrow_kw)
ax1.arrow(x[1], terms_ipsl[1], 0, terms_ipsl[2], facecolor=colors_ipsl[1], edgecolor=colors_ipsl[1], linewidth=2, **arrow_kw)

#  third arrow IPSL - -2Cov(WEP,EEP)
ax1.arrow(x[1]+0.1, terms_ipsl[1]+terms_ipsl[2], 0, terms_ipsl[3], facecolor='black', edgecolor='black', linewidth=4, **arrow_kw)
ax1.arrow(x[1]+0.1, terms_ipsl[1]+terms_ipsl[2], 0, terms_ipsl[3], facecolor=colors_ipsl[2], edgecolor=colors_ipsl[2], linewidth=2, **arrow_kw)

# first arrow FA - Var(WEP)
ax1.arrow(x[3], 0, 0, terms_fa[1], facecolor='black', edgecolor='black', linewidth=4, **arrow_kw)
ax1.arrow(x[3], 0, 0, terms_fa[1], facecolor=colors_ipsl[0], edgecolor=colors_ipsl[0], linewidth=2, **arrow_kw)

# second arrow FA - Var(EEP)
ax1.arrow(x[3], terms_fa[1], 0, terms_fa[2], facecolor='black', edgecolor='black', linewidth=4, **arrow_kw)
ax1.arrow(x[3], terms_fa[1], 0, terms_fa[2], facecolor=colors_ipsl[1], edgecolor=colors_ipsl[1], linewidth=2, **arrow_kw)

# third arrow FA - -2Cov(WEP,EEP)
ax1.arrow(x[3]+0.1, terms_fa[1]+terms_fa[2], 0, terms_fa[3], facecolor='black', edgecolor='black', linewidth=4, **{**arrow_kw, 'head_length': 0.0003})
ax1.arrow(x[3]+0.1, terms_fa[1]+terms_fa[2], 0, terms_fa[3], facecolor=colors_ipsl[2], edgecolor=colors_ipsl[2], linewidth=2, **{**arrow_kw, 'head_length': 0.0003})

ax1.set_title(r"a) ZG variance decomposition (2070-2099)", loc='left', fontsize=fontsize_base+4, x=-0.12, y = +1.05)
ax1.set_xticks([0.4, 1.1])
ax1.set_xticklabels(['IPSL', 'IPSL-FA'], fontsize=labelsize_base)
ax1.tick_params(axis='x', length=0)
ax1.tick_params(axis='y', labelsize=labelsize_base, width=0.8)
ax1.grid(True, axis='y', alpha=0.5, linestyle='-', linewidth=0.5)
ax1.axvline(x=0.85, color='k', linestyle='--', linewidth=1)
ax1.legend(handles=legend_patches, labels=legend_labels, fontsize=labelsize_base-1, ncol=1, loc='upper left',
           handlelength=1, columnspacing=0.7, handletextpad=0.2, frameon=True)

# Right panel: Correlation tau_x / SST ----
corr_ipsl = corr_x_ipsl
corr_fa   = corr_x_FA
ax2.set_title(
    r"b) Corr ( $\tau_x(t)_{\rm nino\,3.4+4}$,  $SST(x,t)_{\rm EqPac}$), 2070–2099",
    loc='left', fontsize=fontsize_base+4,  x=-0.12, y = +1.05)
ax2.tick_params(labelsize=labelsize_base)
ax2.plot(sst_ens_ipsl_s2.x, corr_ipsl, color='red',  lw=2, label='IPSL')
ax2.plot(sst_ens_ipsl_s2.x, corr_fa,   color='blue', lw=2, label='IPSL-FA')
ax2.legend(fontsize=labelsize_base-1, loc='lower right')
ax2.set_xlim(37.5, 207.5)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax2.set_xticks([47.5, 77.5, 107.5, 137.5, 167.5, 197.5])
ax2.set_xticklabels(['120°E', '150°E', '180°', '150°W', '120°W', '90°W'], fontsize=labelsize_base)
ax2.axvline(x=107.5, color='k', linestyle='--', linewidth=1)
ax2.grid(True, alpha=0.5, linestyle='-', linewidth=0.5)
plt.tight_layout()


# plt.savefig("/home/astoppel/figure/mechanism/var_decomp_corr.pdf", bbox_inches='tight')
plt.show()






