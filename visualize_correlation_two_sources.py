import os.path as op
from itertools import product

import mne
import numpy as np
import pandas as pd
from jumeg.jumeg_volume_plotting import plot_vstc_sliced_old
from tqdm import tqdm

import config
from config import fname
from utils import set_directory

###############################################################################
# Compute the settings grid
###############################################################################

regs = [0.05, 0.1, 0.5]
sensor_types = ['joint', 'grad', 'mag']
pick_oris = [None, 'max-power']
weight_norms = ['unit-noise-gain', 'nai', None]
use_noise_covs = [True, False]
depths = [True, False]

settings = list(product(regs, sensor_types, pick_oris, weight_norms,
                        use_noise_covs, depths))

###############################################################################
# Load volume source space
###############################################################################

info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))

fwd = mne.read_forward_solution(fname.fwd_man)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)

vsrc = fwd['src']
vertno = vsrc[0]['vertno']

# needs to be set for plot_vstc_sliced_old to work
if vsrc[0]['subject_his_id'] is None:
    vsrc[0]['subject_his_id'] = 'sample'

###############################################################################
# Get data from csv files
###############################################################################

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(fname.lcmv_results_2s(vertex=vertex), index_col=0)
        df['vertex'] = vertex
        df['noise'] = config.noise
        dfs.append(df)
    except Exception as e:
        print(e)
lcmv = pd.concat(dfs, ignore_index=True)
lcmv['pick_ori'].fillna('none', inplace=True)
lcmv['weight_norm'].fillna('none', inplace=True)

cbar_range_dist = [0, lcmv['nb_dist'].dropna().to_numpy().max()]

###############################################################################
# HTML settings
###############################################################################

html_header = '''
<html>
    <head>
        <meta charset="UTF-8">
        <link rel="stylesheet" type="text/css" href="style.css">
    </head>
    <body>
    <table id="results">
    <tr>
        <th>reg</th>
        <th>sensor type</th>
        <th>pick_ori</th>
        <th>inversion</th>
        <th>weight_norm</th>
        <th>normalize_fwd</th>
        <th>use_noise_cov</th>
        <th>P2P distance</th>
        <th>Fancy metric</th>
    </tr>
'''

html_footer = '''
        <script src="tablefilter/tablefilter.js"></script>
        <script src="filter.js"></script>
    </body>
</html>
'''

html_table = ''

image_folder = 'lcmv_two_sources'
image_path = op.join('html', image_folder)
set_directory(image_path)

for i, setting in enumerate(config.lcmv_settings):
    # construct query
    setting = tuple(['none' if s is None else s for s in setting])
    q = ("reg==%.1f and sensor_type=='%s' and pick_ori=='%s' and inversion=='%s' and "
         "weight_norm=='%s' and normalize_fwd==%s and use_noise_cov==%s" % setting)

    print(q)

    sel = lcmv.query(q).dropna()

    if len(sel) < 1000:
        print('Not enough voxels. Did this run fail?')
        continue

    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, use_noise_cov = setting

    ###############################################################################
    # Create correlation stc from simulated data
    ###############################################################################

    vert_sel = sel['vertex'].get_values()
    data_nb_dist_sel = sel['nb_dist'].get_values()
    data_corr_sel = sel['corr'].get_values()

    # do I want to add small value for thresholding in the plot, e.g., 0.001
    # -> causes points with localization error equal to zero to be black in the plot
    offset = 0.001
    data_dist = np.zeros(shape=(vertno.shape[0], 1)) + offset

    # get for each vertex the neighbor with the smallest pairwise distance where corr < 0.5 ** 0.5
    for vert in np.unique(vert_sel):
        dist_to_nbs = data_nb_dist_sel[np.where(vert_sel == vert)]
        corr_with_nbs = data_corr_sel[np.where(vert_sel == vert)]
        distance = dist_to_nbs[np.where(corr_with_nbs < 0.5 ** 0.5)].min()

        data_dist[vert][0] = distance + offset

    vstc_dist = mne.VolSourceEstimate(data=data_dist, vertices=vertno, tmin=0,
                                      tstep=1 / info['sfreq'], subject='sample')

    ###############################################################################
    # Plot
    ###############################################################################
    fn_image = '%03d_lcmv_dist_2sources_ortho.png' % i
    fp_image = op.join(image_path, fn_image)

    plot_vstc_sliced_old(vstc_dist, vsrc, vstc_dist.tstep,
                         subjects_dir=fname.subjects_dir,
                         time=vstc_dist.tmin, cut_coords=(0, 0, 0),
                         display_mode='ortho', figure=None,
                         axes=None, colorbar=True, cmap='magma',
                         symmetric_cbar='auto', threshold=0,
                         cbar_range=cbar_range_dist,
                         save=True, fname_save=fp_image)

    ###############################################################################
    # Plot
    ###############################################################################

    html_table += '<tr><td>' + '</td><td>'.join([str(s) for s in setting]) + '</td>'
    html_table += '<td><img src="' + op.join(image_folder, fn_image) + '"></td>'

    with open('html/lcmv_vol.html', 'w') as f:
        f.write(html_header)
        f.write(html_table)
        f.write(html_footer)
