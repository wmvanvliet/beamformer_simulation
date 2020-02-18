import pandas as pd
from tqdm import tqdm
import config

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/lcmv_results/new_max_power_ori/lcmv_results-vertex{vertex:04d}-noise0.1.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = 0.1
        dfs.append(df)
    except Exception as e:
        print(e)
lcmv = pd.concat(dfs, ignore_index=True)
lcmv.to_csv('lcmv_new_max_ori.csv')

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/dics_results/new_max_power_ori/dics_results-vertex{vertex:04d}-noise0.1.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = 0.1
        dfs.append(df)
    except Exception as e:
        print(e)
dics = pd.concat(dfs, ignore_index=True)
dics.to_csv('dics_new_max_ori.csv')

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/lcmv_results/old_max_power_ori/lcmv_results-vertex{vertex:04d}-noise0.1.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = 0.1
        dfs.append(df)
    except Exception as e:
        print(e)
lcmv = pd.concat(dfs, ignore_index=True)
lcmv.to_csv('lcmv_old_max_ori.csv')

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(f'/m/nbe/scratch/epasana/beamformer_simulation/data/dics_results/old_max_power_ori/dics_results-vertex{vertex:04d}-noise0.1.csv', index_col=0)
        df['vertex'] = vertex
        df['noise'] = 0.1
        dfs.append(df)
    except Exception as e:
        print(e)
dics = pd.concat(dfs, ignore_index=True)
dics.to_csv('dics_old_max_ori.csv')
