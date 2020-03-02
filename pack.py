"""
Concatenate the individual CSV files produced by the LCMV and DICS simulations.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import fname

noise = 0.1

def fix(x):
    """An orientation error of 180 degrees is actually really good."""
    if x == np.nan:
        return np.nan
    elif x > 90:
        return 180 - x
    else:
        return x


dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/lcmv_results/new_max_power_ori/lcmv_results-vertex{vertex:04d}-noise{noise:.1f}.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = noise
        dfs.append(df)
    except Exception as e:
        print(e)
lcmv = pd.concat(dfs, ignore_index=True)
lcmv['pick_ori'].fillna('none', inplace=True)
lcmv['weight_norm'].fillna('none', inplace=True)
lcmv['ori_error'].fillna(-1, inplace=True)
lcmv['ori_error'] = lcmv['ori_error'].map(fix)
lcmv.to_csv('lcmv_new_max_ori.csv')

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/lcmv_results/old_max_power_ori/lcmv_results-vertex{vertex:04d}-noise{noise:.1f}.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = noise
        dfs.append(df)
    except Exception as e:
        print(e)
lcmv = pd.concat(dfs, ignore_index=True)
lcmv['pick_ori'].fillna('none', inplace=True)
lcmv['weight_norm'].fillna('none', inplace=True)
lcmv['ori_error'].fillna(-1, inplace=True)
lcmv['ori_error'] = lcmv['ori_error'].map(fix)
lcmv.to_csv('lcmv_old_max_ori.csv')

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/dics_results/new_max_power_ori/dics_results-vertex{vertex:04d}-noise{noise:.1f}.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = noise
        dfs.append(df)
    except Exception as e:
        print(e)
dics = pd.concat(dfs, ignore_index=True)
dics['pick_ori'].fillna('none', inplace=True)
dics['weight_norm'].fillna('none', inplace=True)
dics['ori_error'].fillna(-1, inplace=True)
dics['ori_error'] = dics['ori_error'].map(fix)
dics.to_csv('dics_new_max_ori.csv')

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/dics_results/old_max_power_ori/dics_results-vertex{vertex:04d}-noise{noise:.1f}.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = noise
        dfs.append(df)
    except Exception as e:
        print(e)
dics = pd.concat(dfs, ignore_index=True)
dics['pick_ori'].fillna('none', inplace=True)
dics['weight_norm'].fillna('none', inplace=True)
dics['ori_error'].fillna(-1, inplace=True)
dics['ori_error'] = dics['ori_error'].map(fix)
dics.to_csv('dics_old_max_ori.csv')
