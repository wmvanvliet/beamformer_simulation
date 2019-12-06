from itertools import product

import mne
import numpy as np
import pandas as pd
from jumeg.jumeg_volume_plotting import plot_vstc_sliced_old
from tqdm import tqdm

import config
from config import vfname
from utils import set_directory

###############################################################################
# Load volume source space
###############################################################################

info = mne.io.read_info(vfname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))

fwd = mne.read_forward_solution(vfname.fwd_discrete_man)
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
for vertex in tqdm(range(3765), total=3765):
    try:
        df = pd.read_csv(vfname.dics_results(noise=config.noise, vertex=vertex))
        df['vertex'] = vertex
        df['noise'] = config.noise
        dfs.append(df)
    except Exception as e:
        print(e)
dics = pd.concat(dfs, ignore_index=True)
dics['pick_ori'].fillna('none', inplace=True)
dics['weight_norm'].fillna('none', inplace=True)

cbar_range_dist = [0, dics['dist'].dropna().get_values().max()]
cbar_range_eval = [0, dics['eval'].dropna().get_values().max()]

###############################################################################
# Construct dics settings list
###############################################################################

regs = [0.05, 0.1, 0.5]
sensor_types = ['grad', 'mag']
pick_oris = ['none', 'normal', 'max-power']
inversions = ['single', 'matrix']
weight_norms = ['unit-noise-gain', 'none']
normalize_fwds = [True, False]
real_filters = [True, False]
settings = list(product(regs, sensor_types, pick_oris, inversions,
                        weight_norms, normalize_fwds, real_filters))

html_header = '''
    <html>
    <head>
        <link rel="stylesheet" type="text/css" href="style.css">
        <script src="filter.js"></script>
    </head>
    <body>
    <table>
    <tr>
        <th>reg</th>
        <th>sensor type</th>
        <th>pick_ori</th>
        <th>inversion</th>
        <th>weight_norm</th>
        <th>normalize_fwd</th>
        <th>real_filter</th>
        <th>P2P distance</th>
        <th>Fancy metric</th>
    </tr>
    <tr>
        <td><input type="text" onkeyup="filter(0, this)" placeholder="reg"></td>
        <td><input type="text" onkeyup="filter(1, this)" placeholder="sensor type"></td>
        <td><input type="text" onkeyup="filter(2, this)" placeholder="pick_ori"></td>
        <td><input type="text" onkeyup="filter(3, this)" placeholder="inversion"></td>
        <td><input type="text" onkeyup="filter(4, this)" placeholder="weight_norm"></td>
        <td><input type="text" onkeyup="filter(5, this)" placeholder="normalize_fwd"></td>
        <td><input type="text" onkeyup="filter(6, this)" placeholder="real_filter"></td>
        <td></td>
        <td></td>
    </tr>
'''

html_footer = '</body></table>'

html_table = ''

set_directory('html/dics')

for i, setting in enumerate(settings):
    # construct query
    q = ("reg==%.1f and sensor_type=='%s' and pick_ori=='%s' and "
         "inversion=='%s' and weight_norm=='%s' and normalize_fwd==%s and real_filter==%s" % setting)

    print(q)

    sel = dics.query(q).dropna()

    if len(sel) < 1000:
        continue

    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, real_filters = setting

    # Skip some combinations
    if weight_norm == 'unit-noise-gain' and normalize_fwd == True:
        continue
    if weight_norm == 'none' and normalize_fwd == False:
        continue

    ###############################################################################
    # Create dist stc from simulated data
    ###############################################################################

    vert_sel = sel['vertex'].get_values()
    data_dist_sel = sel['dist'].get_values()
    data_eval_sel = sel['eval'].get_values()

    data_dist = np.zeros(shape=(vertno.shape[0], 1))

    # do I want to add small value for thresholding in the plot, e.g., 0.001
    # -> causes points with localization error equal to zero to be black in the plot
    data_dist[vert_sel, 0] = data_dist_sel + 0.001

    vstc_dist = mne.VolSourceEstimate(data=data_dist, vertices=vertno, tmin=0,
                                      tstep=1 / info['sfreq'], subject='sample')

    data_eval = np.zeros(shape=(vertno.shape[0], 1))

    # do I want to add small value for thresholding in the plot, e.g., 0.001
    # -> causes points with localization error equal to zero to be black in the plot
    data_eval[vert_sel, 0] = data_eval_sel + 0.001

    vstc_eval = mne.VolSourceEstimate(data=data_eval, vertices=vertno, tmin=0,
                                      tstep=1 / info['sfreq'], subject='sample')

    ###############################################################################
    # Plot
    ###############################################################################

    fn_image = 'html/dics/%03d_dist_ortho.png' % i

    plot_vstc_sliced_old(vstc_dist, vsrc, vstc_dist.tstep,
                         subjects_dir=vfname.subjects_dir,
                         time=vstc_dist.tmin, cut_coords=(0, 0, 0),
                         display_mode='ortho', figure=None,
                         axes=None, colorbar=True, cmap='magma',
                         symmetric_cbar='auto', threshold=0,
                         cbar_range=cbar_range_dist,
                         save=True, fname_save=fn_image)

    fn_image = 'html/dics/%03d_eval_ortho.png' % i

    plot_vstc_sliced_old(vstc_eval, vsrc, vstc_eval.tstep,
                         subjects_dir=vfname.subjects_dir,
                         time=vstc_eval.tmin, cut_coords=(0, 0, 0),
                         display_mode='ortho', figure=None,
                         axes=None, colorbar=True, cmap='magma',
                         symmetric_cbar='auto', threshold=0,
                         cbar_range=cbar_range_eval,
                         save=True, fname_save=fn_image)

    ###############################################################################
    # Plot
    ###############################################################################

    html_table += '<tr><td>' + '</td><td>'.join([str(s) for s in setting]) + '</td>'
    html_table += '<td><img src="dics/%03d_dist_ortho.png"></td>' % i
    html_table += '<td><img src="dics/%03d_eval_ortho.png"></td>' % i

    with open('html/dics_vol.html', 'w') as f:
        f.write(html_header)
        f.write(html_table)
        f.write(html_footer)

