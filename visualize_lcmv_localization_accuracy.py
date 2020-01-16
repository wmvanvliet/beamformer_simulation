import os.path as op
import warnings

import mne
import numpy as np
import pandas as pd
from jumeg.jumeg_volume_plotting import plot_vstc_sliced_old
from tqdm import tqdm

import config
from config import fname
from utils import set_directory

warnings.filterwarnings('ignore', category=DeprecationWarning)

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
        df = pd.read_csv(fname.lcmv_results(vertex=vertex), index_col=0)
        df['vertex'] = vertex
        df['noise'] = config.noise
        dfs.append(df)
    except Exception as e:
        print(e)
lcmv = pd.concat(dfs, ignore_index=True)
lcmv['pick_ori'].fillna('none', inplace=True)
lcmv['weight_norm'].fillna('none', inplace=True)

cbar_range_dist = [0, lcmv['dist'].dropna().to_numpy().max()]
cbar_range_eval = [0, lcmv['eval'].dropna().to_numpy().max()]
cbar_range_corr = [0, lcmv['corr'].dropna().to_numpy().max()]

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
        <th>reduce_rank</th>
        <th>P2P distance</th>
        <th>Fancy metric</th>
        <th>Time course correlation</th>
    </tr>
'''

html_footer = '''
        <script src="tablefilter/tablefilter.js"></script>
        <script>
            var filtersConfig = {
                base_path: 'tablefilter/',
                col_0: 'checklist',
                col_1: 'checklist',
                col_2: 'checklist',
                col_3: 'checklist',
                col_4: 'checklist',
                col_5: 'checklist',
                col_6: 'checklist',
                col_7: 'checklist',
                col_8: 'none',
                col_9: 'none',
                col_10: 'none',
                filters_row_index: 1,
                enable_checklist_reset_filter: false,
                alternate_rows: true,
                col_types: [
                    'number', 'string', 'string',
                    'string', 'string', 'string',
                    'string', 'string', 'image',
                    'image', 'image'
                ],
                col_widths: [
                    '80px', '150px', '130px',
                    '110px', '170px', '150px',
                    '150px', '150px', '210px',
                    '210px', '210px'
                ]
            };

            var tf = new TableFilter('results', filtersConfig);
            tf.init();

            for (div of document.getElementsByClassName("div_checklist")) {
                div.style.height = 100;
            }
        </script>
    </body>
</html>
'''

html_table = ''

image_folder = 'lcmv'
image_path = op.join('html', image_folder)
set_directory(image_path)

for i, setting in enumerate(config.lcmv_settings):
    # construct query
    setting = tuple(['none' if s is None else s for s in setting])

    q = ("reg==%.2f and sensor_type=='%s' and pick_ori=='%s' and inversion=='%s' and "
         "weight_norm=='%s' and normalize_fwd==%s and use_noise_cov==%s and reduce_rank==%s") % setting

    print(q)

    sel = lcmv.query(q).dropna()

    if len(sel) < 1000:
        print('Not enough voxels. Did this run fail?')
        continue

    ###############################################################################
    # Create dist stc from simulated data
    ###############################################################################

    vert_sel = sel['vertex'].to_numpy()
    data_dist_sel = sel['dist'].to_numpy()
    data_eval_sel = sel['eval'].to_numpy()
    data_corr_sel = sel['corr'].to_numpy()

    # do I want to add small value for thresholding in the plot, e.g., 0.001
    # -> causes points with localization error equal to zero to be black in the plot
    offset = 0.001

    data_dist = np.zeros(shape=(vertno.shape[0], 1))
    data_dist[vert_sel, 0] = data_dist_sel + offset

    vstc_dist = mne.VolSourceEstimate(data=data_dist, vertices=vertno, tmin=0,
                                      tstep=1 / info['sfreq'], subject='sample')

    data_eval = np.zeros(shape=(vertno.shape[0], 1))
    data_eval[vert_sel, 0] = data_eval_sel + offset

    vstc_eval = mne.VolSourceEstimate(data=data_eval, vertices=vertno, tmin=0,
                                      tstep=1 / info['sfreq'], subject='sample')

    data_corr = np.zeros(shape=(vertno.shape[0], 1))
    data_corr[vert_sel, 0] = data_corr_sel + offset

    vstc_corr = mne.VolSourceEstimate(data=data_corr, vertices=vertno, tmin=0,
                                      tstep=1 / info['sfreq'], subject='sample')

    ###############################################################################
    # Plot
    ###############################################################################

    fn_image_dist = '%03d_lcmv_dist_ortho.png' % i
    fp_image_dist = op.join(image_path, fn_image_dist)

    plot_vstc_sliced_old(vstc_dist, vsrc, vstc_dist.tstep,
                         subjects_dir=fname.subjects_dir,
                         time=vstc_dist.tmin, cut_coords=config.cut_coords,
                         display_mode='ortho', figure=None,
                         axes=None, colorbar=True, cmap='magma_r',
                         symmetric_cbar='auto', threshold=0,
                         cbar_range=cbar_range_dist,
                         save=True, fname_save=fp_image_dist)

    fn_image_eval = '%03d_lcmv_eval_ortho.png' % i
    fp_image_eval = op.join(image_path, fn_image_eval)

    plot_vstc_sliced_old(vstc_eval, vsrc, vstc_eval.tstep,
                         subjects_dir=fname.subjects_dir,
                         time=vstc_eval.tmin, cut_coords=config.cut_coords,
                         display_mode='ortho', figure=None,
                         axes=None, colorbar=True, cmap='magma',
                         symmetric_cbar='auto', threshold=0,
                         cbar_range=cbar_range_eval,
                         save=True, fname_save=fp_image_eval)

    fn_image_corr = '%03d_lcmv_corr_ortho.png' % i
    fp_image_corr = op.join(image_path, fn_image_corr)

    plot_vstc_sliced_old(vstc_corr, vsrc, vstc_corr.tstep,
                         subjects_dir=fname.subjects_dir,
                         time=vstc_corr.tmin, cut_coords=config.cut_coords,
                         display_mode='ortho', figure=None,
                         axes=None, colorbar=True, cmap='magma',
                         symmetric_cbar='auto', threshold=0,
                         cbar_range=cbar_range_corr,
                         save=True, fname_save=fp_image_corr)

    ###############################################################################
    # Plot
    ###############################################################################

    html_table += '<tr><td>' + '</td><td>'.join([str(s) for s in setting]) + '</td>'
    html_table += '<td><img src="' + op.join(image_folder, fn_image_dist) + '"></td>'
    html_table += '<td><img src="' + op.join(image_folder, fn_image_eval) + '"></td>'
    html_table += '<td><img src="' + op.join(image_folder, fn_image_corr) + '"></td>'

    with open('html/lcmv_vol.html', 'w') as f:
        f.write(html_header)
        f.write(html_table)
        f.write(html_footer)
