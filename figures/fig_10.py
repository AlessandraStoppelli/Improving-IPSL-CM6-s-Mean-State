#!/usr/bin/env python
# coding: utf-8

# NOTE: file paths below reflect the internal cluster environment used for this analysis.
# They will be updated to match the final archived dataset upon publication.
# See ../DATA_PATHS_REFERENCE.md for the full list of data files this script depends on.



import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob




##### here i uploaded already data computed from previus script ("helper_fig_10.py")




regions    = ['wep', 'eep']

models     = ['ipsl', 'fa']

components = ['u_advection', 'v_advection', 'w_advection']

periods = {
    'near': slice('2005', '2034'),
    'far':  slice('2070', '2099'),}

NORM = 50

mean_vars = {
    'u_advection': ['u_bar_dTdx_mean', 'u_prime_dTdx_mean', 'u_prime_dTprime_dx_mean'],
    'v_advection': ['v_bar_dTdy_mean', 'v_prime_dTdy_mean', 'v_prime_dTprime_dy_mean'],
    'w_advection': ['w_bar_dTdz_mean', 'w_prime_dTdz_mean', 'w_prime_dTprime_dz_mean'],
}




#### Uploading data

data = {}

for region in regions:
    data[region] = {}
    
    for model in models:
        data[region][model] = {}
        
        for comp in components:
            
            # percorso del file
            file_path = f"/data/astoppel/advection/time_evolution_test_cell_area_MEM/{comp}_{model}_{region}_timeseries_MEM.nc"

            # Cerchiamo se il file esiste
            found_files = glob.glob(file_path)
            
            if found_files:
                # Apriamo il primo file trovato
                ds = xr.open_dataset(found_files[0])
                
                print(f"{region} | {model} | {comp} -> variabili: {list(ds.data_vars)}")
                
                # Salviamo il tempo e tutte le variabili del dataset
                saved_data = {'time': ds.time}
                for var in ds.data_vars:
                    saved_data[var] = ds[var]
                
                data[region][model][comp] = saved_data
                
                ds.close()




#### time mean on the period

stats = {}

for region in regions:
    stats[region] = {}
    
    for model in models:
        stats[region][model] = {}
        
        for comp in components:
            if comp not in data[region][model]:
                continue
                
            d = data[region][model][comp]
            
            stats[region][model][comp] = {}
            
            for period_name, period_slice in periods.items():
                
                stats[region][model][comp][period_name] = {}
                
                for var, da in d.items():
                    
                    if var != 'time':
                        
                        stats[region][model][comp][period_name][var] = da.sel(time=period_slice).mean(dim=['time'])




### sum of data 
sum_data = {}

for region in regions:
    sum_data[region] = {}
    for model in models:
        total_mean = None
        for comp, vars_list in mean_vars.items():
            for var_name in vars_list:
                da_mean = data[region][model][comp][var_name] 
                
                total_mean = da_mean if total_mean is None else total_mean + da_mean
                
        sum_data[region][model] = {'mean': total_mean / NORM}
        




####################################################
#### Part two : Computing the Confidence interval
####################################################




var_to_comp = {
    'u_bar_dTdx':         'u_advection',
    'u_prime_dTdx':       'u_advection',
    'u_prime_dTprime_dx': 'u_advection',
    'v_bar_dTdy':         'v_advection',
    'v_prime_dTdy':       'v_advection',
    'v_prime_dTprime_dy': 'v_advection',
    'w_bar_dTdz':         'w_advection',
    'w_prime_dTdz':       'w_advection',
    'w_prime_dTprime_dz': 'w_advection',
}

labels_neg = {
    'u_bar_dTdx':         r"$-\overline{u}\frac{\partial \Delta T}{\partial x}$",
    'u_prime_dTdx':       r"$-\Delta u\frac{\partial \overline{T}}{\partial x}$",
    'u_prime_dTprime_dx': r"$-\Delta u\frac{\partial \Delta T}{\partial x}$",
    'v_bar_dTdy':         r"$-\overline{v}\frac{\partial \Delta T}{\partial y}$",
    'v_prime_dTdy':       r"$-\Delta v\frac{\partial \overline{T}}{\partial y}$",
    'v_prime_dTprime_dy': r"$-\Delta v\frac{\partial \Delta T}{\partial y}$",
    'w_bar_dTdz':         r"$-\overline{w}\frac{\partial \Delta T}{\partial z}$",
    'w_prime_dTdz':       r"$-\Delta w\frac{\partial \overline{T}}{\partial z}$",
    'w_prime_dTprime_dz': r"$-\Delta w\frac{\partial \Delta T}{\partial z}$",
}




fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.5), dpi=300)
period_region = [('near', 'wep'), ('far', 'eep')]
period_titles = ['(a) WEP – Near-future (2005–2034)', '(b) EEP – Far-future (2070–2099)']

bar_width = 0.16
gap = 0.08
spacing = 2 * bar_width + gap + 0.15

configs = [
    ('ipsl', '#e06c6c', ''),
    ('fa',   '#5b9bd5', ''),
]

terms = list(var_to_comp.keys())
term_groups = [terms[i:i+3] for i in range(0, len(terms), 3)]
first_w_term = term_groups[2][0]
other_w_terms = term_groups[2][1:]
group_labels = ['$u$-advection', '$v$-advection', '$w$-advection']

for ax, (period, region), title in zip(axes, period_region, period_titles):
    # Plot u, v, w
    for g_idx in range(3):
        group = term_groups[g_idx]
        x0 = g_idx * spacing
        for b_idx, (model, color, hatch) in enumerate(configs):
            mean = sum(
                stats[region][model][var_to_comp[t]][period][f'{t}_mean'].values / NORM
                for t in group
            )
            first = (g_idx == 0 and period == 'near')
            label_map = {'ipsl': 'IPSL', 'fa': 'IPSL-FA'}
            ax.bar(x0 + b_idx * bar_width, mean, width=bar_width,
                   color=color, edgecolor='black', linewidth=0.6,
                   label=label_map[model] if first else None, alpha=0.9)
    
    # first term W-ADVECTION 
    x0_w_breakdown = 3 * spacing + 0.05  
    for b_idx, (model, color, hatch) in enumerate(configs):
        mean_w1 = stats[region][model][var_to_comp[first_w_term]][period][f'{first_w_term}_mean'].values / NORM
        ax.bar(x0_w_breakdown + b_idx * bar_width, mean_w1, width=bar_width,
               color=color, edgecolor='black', linewidth=0.6, alpha=0.9)
    
    # dashed line of separation
    ax.axvline(x0_w_breakdown - 0.15, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
    
    # Plot Sum 
    x0_sum = 3 * spacing + 0.6
    for b_idx, (model, color, hatch) in enumerate(configs):
        mean_sum = sum(
            stats[region][model][var_to_comp[t]][period][f'{t}_mean'].values / NORM
            for t in terms  
        )
        
        ax.bar(x0_sum + b_idx * bar_width, mean_sum, width=bar_width,
               color=color, edgecolor='black', linewidth=0.6, alpha=0.9)
    
    # Formatting
    tick_pos = [g_idx * spacing + bar_width / 2 for g_idx in range(3)] + \
               [x0_w_breakdown + bar_width / 2, x0_sum + bar_width / 2]

    
    tick_labs = group_labels + [labels_neg[first_w_term], 'Sum']
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labs, fontsize=9)
    ax.tick_params(labelsize=9, length=4, width=0.7)
    
    ax.axhline(0, color='black', linewidth=0.8, zorder=0)
    ax.grid(axis='y', linestyle='-', alpha=0.15, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    
    ax.set_ylabel('ΔT (°C)', fontsize=11, fontweight='normal')
    ax.set_title(title, fontsize=11, loc='left', fontweight='normal', pad=10)
    
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(0.8)
    
    ax.minorticks_on()
    ax.tick_params(which='minor', length=2, width=0.5)

axes[1].set_ylim(-150, 150)
axes[0].legend(frameon=False, fontsize=10, loc='upper left', handlelength=1.5)

plt.tight_layout(pad=0.5)

# plt.savefig("/home/astoppel/figure/mechanism/time_evolution/bar_advection_final.pdf", bbox_inches='tight') 

plt.show()






