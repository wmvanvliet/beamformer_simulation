"""
Concatenate the individual CSV files produced by the LCMV and DICS simulations.
"""
import pandas as pd
from tqdm import tqdm

noise = 0.1

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/lcmv_results/lcmv_results-vertex{vertex:04d}-noise{noise:.1f}.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = noise
        dfs.append(df)
    except Exception as e:
        print(e)
lcmv = pd.concat(dfs, ignore_index=True)
lcmv['pick_ori'].fillna('none', inplace=True)
lcmv['weight_norm'].fillna('none', inplace=True)
lcmv['ori_error'].fillna(-1, inplace=True)
lcmv.to_csv('lcmv.csv')

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/dics_results/dics_results-vertex{vertex:04d}-noise{noise:.1f}.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = noise
        dfs.append(df)
    except Exception as e:
        print(e)
dics = pd.concat(dfs, ignore_index=True)
dics['pick_ori'].fillna('none', inplace=True)
dics['weight_norm'].fillna('none', inplace=True)
dics['ori_error'].fillna(-1, inplace=True)
dics.to_csv('dics.csv')
