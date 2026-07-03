# Stoppelli et al. (2026) — Code for Journal of Climate submission

This repository contains the analysis and plotting scripts used to produce the figures and tables
in Improving IPSL-CM6’s Mean State Yields a Transient La Niña-like Response Followed by Amplified El Niño-like Warming, submitted to the *Journal of Climate*.

## Repository structure

```
repo/
├── figures/                    # one script per figure/table
│   ├── fig_01.py
│   ├── fig_03.py
│   ├── fig_04.py
│   ├── fig_05.py
│   ├── fig_06_table_1.py
│   ├── fig_07.py
│   ├── fig_08.py
│   ├── fig_09.py
│   ├── fig_10_table_S1.py      # requires output of helper_fig_10.py (see below)
│   └── fig_11.py
├── environment.yml             # conda environment with all dependencies
├── DATA_PATHS_REFERENCE.md     # full list of data files referenced by each script
└── README.md
```


## Data

The scripts reference NetCDF files from CMIP6, IPSL-CM6A-LR, the flux-adjusted IPSL-FA
simulation, and observational/reanalysis products (ERA5, ORAS5, HadISST, ERSSTv5, COBE2).

Paths in these scripts currently reflect the internal file structure of the authors'
computing cluster at the time of analysis 

Once the processed data are archived, this README and the scripts will be updated to point to the archived filenames.


## Citation

If you use this code, please cite:

> Stoppelli et al., 2026: Improving IPSL-CM6’s Mean State Yields a Transient La Niña-like Response Followed by Amplified El Niño-like Warming. *Journal of Climate*, in review.

A permanent, citable snapshot of this repository (with DOI) will be linked here upon
publication.

