#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.



import xarray as xr
import xesmf as xe 

import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats




ref_period = ['1950-01-01', '1979-12-31']

future_periods = {
    'present': ['2005-01-01', '2034-12-31'],
    'far_future': ['2070-01-01', '2099-12-31']}


members_FA = members_ipsl = ['r1','r2','r3','r4','r12','r13','r14','r15','r16','r18','r22','r23','r29','r30','r33'] 

color_scheme = {
    "FA_box" : '#5b9bd5',  
    "IPSL_box" : '#e06c6c'}  




#Grids & Masks

file_nemo = xr.open_dataset('/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Tgrid_yearmean/S3-tau200-r4_merged_grid_T.nc').sel(time_counter=slice('1950-01-01', '1950-02-01'))
area_nemo = file_nemo['cell_area']

file_regular = xr.open_dataset('/data/astoppel/CMIP6/ensamble_historical/ipsl_CMIP6_TOS_1921_2014.nc')
grid_regular = {"lon": file_regular["lon"], "lat": file_regular["lat"]}




#### Functions

def load_member_sst(member, file_past, file_future, time_var='time_counter',
                    var_name='tos', start=None, end=None):
    
    ds_past = xr.open_dataset(file_past)
    ds_future = xr.open_dataset(file_future)

    sst_concat = xr.concat([ds_past[var_name], ds_future[var_name]], dim=time_var)

    if time_var != 'time_counter':
        sst_concat = sst_concat.rename({time_var: 'time_counter'})

    if start and end:
        sst_concat = sst_concat.sel(time_counter=slice(start, end))

    return sst_concat.expand_dims(member=[member])


def build_ensemble(members, base_path_past, base_path_future,
                   file_pattern_past="S3-tau200-r{m}_merged_grid_T.nc",
                   file_pattern_future="S3-ssp585-r{m}_merged_grid_T.nc",
                   time_var='time', var_name='tos', start=None, end=None):

    sst_list = []

    for m in members:
        file_past = f"{base_path_past}/{file_pattern_past.format(m=m)}"
        file_future = f"{base_path_future}/{file_pattern_future.format(m=m)}"

        sst_m = load_member_sst(
            member=m,
            file_past=file_past,
            file_future=file_future,
            time_var=time_var,
            var_name=var_name,
            start=start,
            end=end
        )
        
        if sst_m is not None:
            sst_list.append(sst_m)
        else:
            print(f"Membro {m} fallito")


    return xr.concat(sst_list, dim='member')

def compute_rsst_change_per_member(data_list, period1, period2, area=None):

    changes_all = []

    for m, member_da in enumerate(data_list):
        
        # 1. Seleziona i tropici
        tropical_mask = (member_da.nav_lat >= -30) & (member_da.nav_lat <= 30)
        tropical_da = member_da.where(tropical_mask, drop=True)
        
        if area is not None:
            tropical_area = area.where(tropical_mask, drop=True)
            # Media tropicale pesata per periodo 1
            trop_mean1 = tropical_da.sel(time_counter=slice(period1[0], period1[1])).weighted(tropical_area.fillna(0.)).mean(dim=["y", "x"], skipna=True).mean(dim='time_counter')
            # Media tropicale pesata per periodo 2
            trop_mean2 = tropical_da.sel(time_counter=slice(period2[0], period2[1])).weighted(tropical_area.fillna(0.)).mean(dim=["y", "x"], skipna=True).mean(dim='time_counter')
        else:
            trop_mean1 = tropical_da.sel(time_counter=slice(period1[0], period1[1])).mean(dim=['nav_lat', 'nav_lon']).mean(dim='time_counter')
            trop_mean2 = tropical_da.sel(time_counter=slice(period2[0], period2[1])).mean(dim=['nav_lat', 'nav_lon']).mean(dim='time_counter')
        
        # 2. Calcola RSST per ogni periodo
        rsst1 = member_da.sel(time_counter=slice(period1[0], period1[1])).mean(dim='time_counter') - trop_mean1
        rsst2 = member_da.sel(time_counter=slice(period2[0], period2[1])).mean(dim='time_counter') - trop_mean2
        # 3. Differenza RSST
        rsst_change = rsst2 - rsst1
        changes_all.append(rsst_change.values)
        
    changes_da = xr.DataArray(
        np.array(changes_all),
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


def regrid_wind_data(wind_data, grid_target, method="bilinear"):

    wind_ds = wind_data.to_dataset(name='wind')
    
    regridder = xe.Regridder(
        wind_ds, grid_target,
        method=method,
        reuse_weights=False,
        ignore_degenerate=True
    )
    
    wind_regridded = regridder(wind_ds)['wind']
    return wind_regridded
    

def compute_wind_mean_change(file_past_U, file_future_U, file_past_V, file_future_V, period1, period2, grid_target, time_dim='time_counter'):

    # Compute reference period
    ds_U1 = xr.open_dataset(file_past_U)
    ds_V1 = xr.open_dataset(file_past_V)
    
    tauu1 = ds_U1['tauuo'].sel(**{time_dim: slice(period1[0], period1[1])}).mean(dim=time_dim)
    tauv1 = ds_V1['tauvo'].sel(**{time_dim: slice(period1[0], period1[1])}).mean(dim=time_dim)
    
    # Compute reference period 2
    ds_U2 = xr.open_dataset(file_future_U)  
    ds_V2 = xr.open_dataset(file_future_V)
    
    tauu2 = ds_U2['tauuo'].sel(**{time_dim: slice(period2[0], period2[1])}).mean(dim=time_dim)
    tauv2 = ds_V2['tauvo'].sel(**{time_dim: slice(period2[0], period2[1])}).mean(dim=time_dim)
    
    # Compute difference
    tauu_change = tauu2 - tauu1
    tauv_change = tauv2 - tauv1
    
    # Regrid  
    tauu_regridded = regrid_wind_data(tauu_change, grid_target)
    tauv_regridded = regrid_wind_data(tauv_change, grid_target)

    # Close dataset
    ds_U1.close(); ds_V1.close(); ds_U2.close(); ds_V2.close()
    
    return tauu_regridded, tauv_regridded









#### Upload SST data


# IPSL-FA
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

# IPSL-CTL
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




#### Compute RSST change

fa_members_list = [sst_ensemble_FA.isel(member=i) for i in range(len(members_FA))]
ipsl_members_list = [sst_ensemble_ipsl.isel(member=i) for i in range(len(members_ipsl))]

rsst_changes = {}
rsst_change_fa, rsst_change_ipsl = {}, {}

for period_name, period_range in future_periods.items():
    print(f"\n Periodo: {period_name} ({period_range[0]} to {period_range[1]}) ")
    
    # IPSL-FA
    rsst_change_fa[period_name] = compute_rsst_change_per_member(
        fa_members_list, ref_period, period_range, area=area_nemo
    )
    rsst_change_fa_mean = rsst_change_fa[period_name].mean(dim='member')
    
    # IPSL-CTL  
    rsst_change_ipsl[period_name] = compute_rsst_change_per_member(
        ipsl_members_list, ref_period, period_range, area=area_nemo
    )
    rsst_change_ipsl_mean = rsst_change_ipsl[period_name].mean(dim='member')
    
    rsst_change_fa_mean[:,73] = rsst_change_fa_mean[:,72]
    rsst_change_ipsl_mean[:,73] = rsst_change_ipsl_mean[:,72]
    
    rsst_changes[period_name] = {
        'FA': rsst_change_fa_mean,
        'IPSL': rsst_change_ipsl_mean
    }




#### Wind stress changes


members_FA_wind = members_ipsl_wind = ['MEM']
wind_changes = {}

for period_name, period_range in future_periods.items():
    
    # IPSL-FA
    tauu_fa_changes = {}
    tauv_fa_changes = {}
    
    for m in members_FA_wind:
        tauu_fa, tauv_fa = compute_wind_mean_change(
            file_past_U=f"/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Ugrid_yearmean/ensemble_mean_tauuo.nc",
            file_future_U=f"/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Ugrid_yearmean/ensemble_mean_uo_ssp585.nc", 
            file_past_V=f"/scratchu/astoppel/IPSL_FA/historical/NEMO_LMD_Annual_Mean/Vgrid_yearmean/ensemble_mean_tauvo.nc",
            file_future_V=f"/scratchu/astoppel/IPSL_FA/ssp585/NEMO_LMD_Annual_Mean/Vgrid_yearmean/S3-ssp585-MEM_merged_grid_V.nc",     
            period1=ref_period, 
            period2=period_range, 
            grid_target=grid_regular
        )
        tauu_fa_changes[m] = tauu_fa
        tauv_fa_changes[m] = tauv_fa

    # IPSL-CTL
    tauu_ipsl_changes = {}
    tauv_ipsl_changes = {}
    
    for m in members_ipsl_wind:
        tauu_ipsl, tauv_ipsl = compute_wind_mean_change(
            file_past_U=f"/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tauuo/tauuo_Oyr_IPSL-CM6A-LR_historical_MEM_gn_1950-2014.nc",
            file_future_U=f"/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tauuo/tauuo_Oyr_IPSL-CM6A-LR_ssp585_MEM_gn_2015_2099.nc",
            file_past_V=f"/scratchu/astoppel/IPSL_CTL/historical/CMIP6/tauvo/tauvo_Oyr_IPSL-CM6A-LR_historical_MEM_gn_1950-2014.nc",
            file_future_V=f"/scratchu/astoppel/IPSL_CTL/ssp585/CMIP6/tauvo/tauvo_Oyr_IPSL-CM6A-LR_ssp585_MEM_gn_2015_2099.nc",
            time_dim='time',   
            period1=ref_period, 
            period2=period_range, 
            grid_target=grid_regular
        )
        tauu_ipsl_changes[m] = tauu_ipsl
        tauv_ipsl_changes[m] = tauv_ipsl
    
    tauu_fa_mean = xr.concat(list(tauu_fa_changes.values()), dim='member').mean(dim='member')
    tauv_fa_mean = xr.concat(list(tauv_fa_changes.values()), dim='member').mean(dim='member')
    tauu_ipsl_mean = xr.concat(list(tauu_ipsl_changes.values()), dim='member').mean(dim='member')  
    tauv_ipsl_mean = xr.concat(list(tauv_ipsl_changes.values()), dim='member').mean(dim='member')

    
    lat_slice = slice(-50, 50)
    
    tauu_fa_msk   = tauu_fa_mean.sel(lat=lat_slice)
    tauv_fa_msk   = tauv_fa_mean.sel(lat=lat_slice)
    tauu_ipsl_msk = tauu_ipsl_mean.sel(lat=lat_slice)
    tauv_ipsl_msk = tauv_ipsl_mean.sel(lat=lat_slice)

    wind_changes[period_name] = {
        'FA': (tauu_fa_msk, tauv_fa_msk),
        'IPSL': (tauu_ipsl_msk, tauv_ipsl_msk)
    }




###### Regridding

def regrid_change_data(change_data, grid_target, method="bilinear"):

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

    regridded_dict = {}
    
    for nome_membro, rsst_data in rsst_dict.items():
        
        rsst_regridded =  regrid_change_data(rsst_data, grid_target)
        
        regridded_dict[nome_membro] = rsst_regridded
    
    return regridded_dict

def filter_outliers(data, lower_percentile=2, upper_percentile=98):

    lower_bound = np.nanpercentile(data, lower_percentile)
    upper_bound = np.nanpercentile(data, upper_percentile)
    
    return data.where( (data <= upper_bound) & (data >= lower_bound) )




change_rsst_ipsl_dict, change_rsst_FA_dict = {}, {}
change_rsst_ipsl_regridded_memb,change_rsst_FA_regridded_memb = {}, {}

for per_name, per_range in future_periods.items():

    # create dictionary
    change_rsst_ipsl_dict[per_name] = {}
    change_rsst_FA_dict[per_name] = {}

    for i, m in enumerate(rsst_change_ipsl[per_name].member):
        nome_membro = members_ipsl[i]
        change_rsst_ipsl_dict[per_name][nome_membro] = rsst_change_ipsl[per_name].sel(member=m)
        change_rsst_FA_dict[per_name][nome_membro] = rsst_change_fa[per_name].sel(member=m)


    # Regridd data 
    change_rsst_ipsl_regridded_memb[per_name] = regrid_dictionary(change_rsst_ipsl_dict[per_name], grid_regular)
    change_rsst_FA_regridded_memb[per_name] = regrid_dictionary(change_rsst_FA_dict[per_name], grid_regular)


#rsst mem
rsst_fa_reg = {}
rsst_ipsl_reg = {}

for per_name, per_range in future_periods.items():

    rsst_fa_reg[per_name] = regrid_change_data(rsst_changes[per_name]["FA"], grid_regular)
    rsst_ipsl_reg[per_name] = regrid_change_data(rsst_changes[per_name]["IPSL"], grid_regular)

    rsst_fa_reg[per_name][:, 73] = rsst_fa_reg[per_name][:, 72]
    rsst_ipsl_reg[per_name][:, 73] = rsst_ipsl_reg[per_name][:, 72]





tauu_fa_fil , tauu_ipsl_fil = {},  {}
tauv_fa_fil , tauv_ipsl_fil = {}, {}

for per_name, per_range in future_periods.items():

    tauu_fa[per_name], tauv_fa[per_name] = wind_changes[per_name]["FA"]
    tauu_ipsl[per_name], tauv_ipsl [per_name]= wind_changes[per_name]["IPSL"]

    tauu_fa_fil[per_name] = filter_outliers(tauu_fa[per_name])
    tauv_fa_fil[per_name] = filter_outliers(tauv_fa[per_name])

    tauu_ipsl_fil[per_name] = filter_outliers(tauu_ipsl[per_name])
    tauv_ipsl_fil[per_name] = filter_outliers(tauv_ipsl[per_name])




##### Extract region RSST change




def extract_region_change_rsst(rsst_change, lat_range, lon_range):

    lat_mask = (rsst_change.nav_lat >= lat_range[0]) & (rsst_change.nav_lat <= lat_range[1])
    
    lon_mask = (rsst_change.nav_lon >= lon_range[0]) & (rsst_change.nav_lon <= lon_range[1])
    
    region_data = rsst_change.where(lat_mask & lon_mask, drop=True)
    
    area_box = area_nemo.where(lat_mask & lon_mask, drop=True)

    return region_data.weighted(area_box.fillna(0.)).mean(dim=['y', 'x'])




regioni_rsst_180_180 = {
    'TIO': {  #   (Brady) (30, 100) ?
        'lat_range': (-10, 10),'lon_range': (35, 100)},

    'ATL': {  #  (McGregor Nature) -50, 20 ?
        'lat_range': (-10, 10),'lon_range': (-50, 15)},

    'SO': {  #   (Dong 22)
        'lat_range': (-62, -47), 'lon_range': (-140, -70)},

    'SEP':{ 
        'lat_range': (-35, -10), 'lon_range': (-140, -70)},

    'NH' :{
        'lat_range': (0, 70), 'lon_range': (-180, 180)},
    'SH' :{
        'lat_range': (-70, 0), 'lon_range': (-180, 180)}}




rsst_change = {}

for per_name in future_periods:
    datasets = {
        "IPSL": change_rsst_ipsl_dict[per_name],
        "IPSL-FA": change_rsst_FA_dict[per_name],
    }

    rsst_change[per_name] = {
        reg_name: {
            model_name: {
                member_name: extract_region_change_rsst(
                    change_data,
                    reg_coords["lat_range"],
                    reg_coords["lon_range"]
                )
                for member_name, change_data in model_dict.items()
            }
            for model_name, model_dict in datasets.items()
        }
        for reg_name, reg_coords in regioni_rsst_180_180.items()  
    }




###CI

from scipy import stats

regioni_box = ["ATL", "TIO", "SEP", "SO", "NH-SH"]

risultati = []

for reg in regioni_box:
    
    if reg == "NH-SH":
        fa = [n.item() - s.item()
            for n, s in zip(rsst_change[per_name]["NH"]["IPSL-FA"].values(),
                            rsst_change[per_name]["SH"]["IPSL-FA"].values())]
        
        ip = [n.item() - s.item()
            for n, s in zip(rsst_change[per_name]["NH"]["IPSL"].values(),
                            rsst_change[per_name]["SH"]["IPSL"].values())]
    else:
        ip = [v.item() for v in rsst_change[per_name][reg]["IPSL"].values()]
        fa = [v.item() for v in rsst_change[per_name][reg]["IPSL-FA"].values()]
    
    media_ip = np.mean(ip)
    std_ip = np.std(ip, ddof=1)
    n_ip = len(ip)
    t_crit = stats.t.ppf(0.95, df=n_ip-1)  # 90% confidence
    ci_ip = t_crit * (std_ip / np.sqrt(n_ip))
    
    sig_ip = (media_ip - ci_ip > 0) or (media_ip + ci_ip < 0)
    
    media_fa = np.mean(fa)
    std_fa = np.std(fa, ddof=1)
    n_fa = len(fa)
    ci_fa = t_crit * (std_fa / np.sqrt(n_fa))
    
    sig_fa = (media_fa - ci_fa > 0) or (media_fa + ci_fa < 0)
    
    risultati.append({
        "regione": reg,
        "IPSL_sig": sig_ip,
        "IPSL_media": media_ip,
        "FA_sig": sig_fa,
        "FA_media": media_fa
    })




########
#PLOT




def plot_change(ax, title, 
                sst_change= None, tauu=None, tauv=None,
                rsst_fa=None, rsst_ipsl=None,
                tauu_ipsl=None, tauv_ipsl=None,
                tauu_fa=None, tauv_fa=None, #generali 
                difference=False,
                wind_scale=1, period="present",
                lon_range=None, lat_range=None):
    
    if difference == True:

        if period == "present":
            change_levels = np.linspace(-0.3, 0.3, 13)
        else:
            change_levels = np.linspace(-0.6, 0.6, 13)

        sst_change = rsst_fa - rsst_ipsl

        #lo posso fare anche senza vento il plot
        if tauu_fa is not None and tauv_fa is not None: 
            tauu = tauu_fa - tauu_ipsl
            tauv = tauv_fa - tauv_ipsl
        else:
            tauu = tauv = None

    else:
        if period == "present":
            change_levels = np.linspace(-0.5, 0.5, 11)
        else:
            change_levels = np.linspace(-1.2, 1.2, 13)

    if lon_range is not None and lat_range is not None:
        # Per SST
        if sst_change is not None:
            sst_change = sst_change.sel(
                lon=slice(lon_range[0], lon_range[1]),
                lat=slice(lat_range[0], lat_range[1])
            )
        
        # Per i venti
        if tauu is not None:
            tauu = tauu.sel(
                lon=slice(lon_range[0], lon_range[1]),
                lat=slice(lat_range[0], lat_range[1])
            )
        if tauv is not None:
            tauv = tauv.sel(
                lon=slice(lon_range[0], lon_range[1]),
                lat=slice(lat_range[0], lat_range[1])
            )

    lon = sst_change.lon
    lat = sst_change.lat

    cmap = plt.get_cmap("RdBu_r")

    cf = ax.contourf(
        lon, lat, sst_change,
        levels=change_levels,
        cmap=cmap,
        extend='both',
        transform=ccrs.PlateCarree(),
            rasterized=True
        )


    cs = ax.contour(
        lon, lat, sst_change,
        levels=change_levels[::3],
        colors='black',
        linewidths=0.3,
        transform=ccrs.PlateCarree()
    )
    for col in cs.collections:
        col.set_rasterized(True)
        

    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=2)
    ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)
    ax.set_global()

    ax.set_title(title, fontsize=20, pad=10, loc = 'left')

    q = None
    if tauu is not None and tauv is not None:
        step = 5
        step_x = 8 
        
        q = ax.quiver(
            lon[::step_x], lat[::step],
            tauu.values[::step, ::step_x], 
            tauv.values[::step, ::step_x],
            transform=ccrs.PlateCarree(),
            color='black',
            scale=wind_scale,
            width=0.002,
            headwidth=3,
            headlength=3,
            pivot='mid'
        )

    return cf, q




wind_scale_present = 0.2
wind_scale_future = 0.5
period_list = ["present"]
periodo_numeri = "2005-2034"

for per_name in period_list:
    fig = plt.figure(figsize=(10, 14))
    gs = gridspec.GridSpec(
        2, 1,                         
        height_ratios=[1, 1],
        hspace=0.17,
        top=0.95, bottom=0.30
    )       
    proj_global = ccrs.Mollweide(central_longitude=120)
    
    wind_scale = wind_scale_present if per_name == "present" else wind_scale_future
    ref_value = 0.005 if per_name == "present" else 0.01  # coerente con wind_scale

    tauu_fa = tauu_fa_fil[per_name]
    tauv_fa = tauv_fa_fil[per_name]
    tauu_ipsl = tauu_ipsl_fil[per_name]
    tauv_ipsl = tauv_ipsl_fil[per_name]

    # (a) IPSL — gs[0]
    ax1 = fig.add_subplot(gs[0], projection=proj_global)
    cf1, q1 = plot_change(
        ax1,
        sst_change=rsst_ipsl_reg[per_name],
        title="a) IPSL",
        tauu=tauu_ipsl,
        tauv=tauv_ipsl,
        wind_scale=wind_scale,
        period=per_name
    )
    ax1.set_title(ax1.get_title(), fontsize=23)
    ax1.quiverkey(
        q1, X=0.55, Y=1.05, U=ref_value,
        label=rf"{ref_value:.1e} $N\,m^{{-2}}$",
        labelpos='E', coordinates='axes',
        fontproperties={'size': 14}
    )

    # (b) IPSL-FA — gs[1]
    ax2 = fig.add_subplot(gs[1], projection=proj_global)
    cf2, q2 = plot_change(
        ax2,
        sst_change=rsst_fa_reg[per_name],
        title="b) IPSL-FA",
        tauu=tauu_fa,
        tauv=tauv_fa,
        wind_scale=wind_scale,
        period=per_name
    )
    ax2.set_title(ax2.get_title(), fontsize=23)
    
    ax2.quiverkey(
        q2, X=0.55, Y=1.05, U=ref_value,
        label=rf"{ref_value:.1e} $N\,m^{{-2}}$",
        labelpos='E', coordinates='axes',
        fontproperties={'size': 14})


    cbar = plt.colorbar(cf1, ax=[ax1, ax2], orientation="horizontal", fraction=0.03, pad=0.02, shrink=0.8)
    cbar.set_label("ΔRSST (°C)", fontsize=16)
    

    # RETTANGOLI  
    axes_to_draw = [ax1, ax2]#, ax3]

    for ax in axes_to_draw:
        for reg_name, reg in regioni_rsst_180_180.items():
            if reg_name in ['NH', 'SH']:
                continue
                
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
                zorder=3
            )
            ax.add_patch(rect)
            
            ax.text(
                lon_max ,   
                lat_max + 3,  
                reg_name,
                transform=ccrs.PlateCarree(),
                fontsize=13,
                fontweight='bold',
                color='black',
                ha='center',   
                va='bottom',
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor='white',
                    alpha=0.85,
                    edgecolor='black',
                    linewidth=0.8
                ),
                zorder=4
            )


    # (d) BOXPLOT 

    ax_box = fig.add_axes([0.12, 0.04, 0.8, 0.22])

    regioni_box = ["ATL", "TIO", "SEP", "SO", "NH-SH"]
    box_data, colors, positions = [], [], []
    pos = 1

    for reg in regioni_box:
        if reg == "NH-SH":
            fa = [n.item() - s.item()
                for n, s in zip(rsst_change[per_name]["NH"]["IPSL-FA"].values(),
                                rsst_change[per_name]["SH"]["IPSL-FA"].values())]
            
            ip = [n.item() - s.item()
                for n, s in zip(rsst_change[per_name]["NH"]["IPSL"].values(),
                                rsst_change[per_name]["SH"]["IPSL"].values())]
            
            box_data += [ip, fa]
            colors += [ color_scheme["IPSL_box"], color_scheme["FA_box"]]
            positions += [pos, pos + 1]
            pos += 2
        else:
            box_data.append([v.item() for v in rsst_change[per_name][reg]["IPSL"].values()])
            colors.append(color_scheme["IPSL_box"])
            positions.append(pos)
            pos += 1
        
            box_data.append([v.item() for v in rsst_change[per_name][reg]["IPSL-FA"].values()])
            colors.append(color_scheme["FA_box"])
            positions.append(pos)
            pos += 1

    bp = ax_box.boxplot(
        box_data,
        positions=positions,
        patch_artist=True,
        widths=0.45,
        medianprops={'visible': False},
        showmeans=True,
        meanline=True,
        meanprops={'color': 'black', 'linewidth': 1.2, 'linestyle': '--'})

    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.85)

    for pos, data, c in zip(positions, box_data, colors):
        ax_box.scatter(np.full(len(data), pos), data,
                       color=c, s=35, alpha=0.6,
                       edgecolors='white', linewidth=0.5)

    for hlines in [ 2.5, 4.5, 6.5, 8.5 ]:
        ax_box.axvline(hlines, color='gray', linestyle='--', linewidth=1.5)
    ax_box.axhline(0, color='gray', linestyle='--', linewidth=1.5 )

    ax_box.set_ylabel("ΔRSST (°C)", fontsize=14)

    xt = []
    for res in risultati:
        reg = res["regione"]
        # Label per IPSL
        label_ip = reg if not res["IPSL_sig"] else reg + "*"
        # Label per IPSL-FA
        label_fa = reg if not res["FA_sig"] else reg + "*"
        xt += [label_ip, label_fa]
    

    ax_box.set_xticks(positions)
    ax_box.set_xticklabels(xt, fontsize=16)
    ax_box.grid(axis='y', alpha=0.5, linestyle='--')
    ax_box.tick_params(axis='y', labelsize=14)  

    ax_box.legend( handles=[mpatches.Patch(facecolor=color_scheme["IPSL_box"], label="IPSL", alpha=0.7),
            mpatches.Patch(facecolor=color_scheme["FA_box"], label="IPSL-FA", alpha=0.7), ], loc="lower left", fontsize=16 )

    ax_box.set_title( "c)", loc = 'left', fontsize=20, pad=15)
    
    # plt.savefig("/home/astoppel/figure/cambiamenti/rsst_change/ipsl_ipslfa_boxplot_nf.pdf", bbox_inches='tight', facecolor='white')
    plt.show()






