import os.path as op

import mne
import numpy as np
import pandas as pd
from jumeg.jumeg_volume_plotting import plot_vstc_sliced_old
from tqdm import tqdm

import config
from config import dics_settings
from config import fname
from utils import set_directory

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
        df = pd.read_csv(fname.dics_results(vertex=vertex, noise=0.1), index_col=0)
        df['vertex'] = vertex
        df['noise'] = config.noise
        dfs.append(df)
    except Exception as e:
        print(e)
dics = pd.concat(dfs, ignore_index=True)
dics.to_csv('dics_new_max_ori.csv')
1/0
dics['pick_ori'].fillna('none', inplace=True)
dics['weight_norm'].fillna('none', inplace=True)
dics['ori_error'].fillna(-1, inplace=True)

cbar_range_dist = [0, dics['dist'].dropna().to_numpy().max()]
cbar_range_eval = [0, dics['eval'].dropna().to_numpy().max()]
cbar_range_ori = [0, dics['ori_error'].dropna().to_numpy().max()]

html_header = '''
    <html>
    <head>
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
        <th>real_filter</th>
        <th>use_noise_cov</th>
        <th>reduce_rank</th>
        <th>P2P distance</th>
        <th>Focality</th>
        <th>Orientation error</th>
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
                col_8: 'checklist',
                col_9: 'none',
                col_10: 'none',
                col_11: 'none',
                filters_row_index: 1,
                enable_checklist_reset_filter: false,
                alternate_rows: true,
                sticky_headers: true,
                col_types: [
                    'number', 'string', 'string',
                    'string', 'string', 'string',
                    'string', 'string', 'string',
                    'image', 'image', 'image'
                ],
                col_widths: [
                    '80px', '150px', '130px',
                    '110px', '170px', '150px',
                    '150px', '150px', '150px',
                    '210px', '210px', '210px'
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

set_directory('html/dics')

image_folder = 'dics'
image_path = op.join('html', image_folder)
set_directory(image_path)

for i, setting in enumerate(dics_settings):
    # construct query
    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, real_filter, use_noise_cov, reduce_rank = setting
    q = (f"reg=={reg:.2f} and sensor_type=='{sensor_type}' and pick_ori=='{pick_ori}' and inversion=='{inversion}' and real_filter=={real_filter} and "
         f"weight_norm=='{weight_norm}' and normalize_fwd=={normalize_fwd} and use_noise_cov=={use_noise_cov} and reduce_rank=='{reduce_rank}'")

    print(q)

    sel = dics.query(q).dropna()

    if len(sel) < 1000:
        print('Not enough voxels. Did this run fail?')
        continue

    ###############################################################################
    # Create dist stc from simulated data
    ###############################################################################

    vert_sel = sel['vertex'].to_numpy()
    data_dist_sel = sel['dist'].to_numpy()
    data_eval_sel = sel['eval'].to_numpy()
    data_ori_sel = sel['ori_error'].to_numpy()

    data_dist = np.zeros(shape=(vertno.shape[0], 1))
    data_dist[vert_sel, 0] = data_dist_sel + 0.001
    vstc_dist = mne.VolSourceEstimate(data=data_dist, vertices=vertno, tmin=0,
                                      tstep=1 / info['sfreq'], subject='sample')

    data_eval = np.zeros(shape=(vertno.shape[0], 1))
    data_eval[vert_sel, 0] = data_eval_sel + 0.001
    vstc_eval = mne.VolSourceEstimate(data=data_eval, vertices=vertno, tmin=0,
                                      tstep=1 / info['sfreq'], subject='sample')

    data_ori = np.zeros(shape=(vertno.shape[0], 1))
    data_ori[vert_sel, 0] = data_ori_sel + 0.001
    vstc_ori = mne.VolSourceEstimate(data=data_ori, vertices=vertno, tmin=0,
                                     tstep=1 / info['sfreq'], subject='sample')

    ###############################################################################
    # Plot
    ###############################################################################

    fn_image_dist = '%03d_dics_dist_ortho.png' % i
    fp_image_dist = op.join(image_path, fn_image_dist)

    plot_vstc_sliced_old(vstc_dist, vsrc, vstc_dist.tstep,
                         subjects_dir=fname.subjects_dir,
                         time=vstc_dist.tmin, cut_coords=config.cut_coords,
                         display_mode='ortho', figure=None,
                         axes=None, colorbar=True, cmap='magma_r',
                         symmetric_cbar='auto', threshold=0,
                         cbar_range=cbar_range_dist,
                         save=True, fname_save=fp_image_dist)

    fn_image_eval = '%03d_dics_eval_ortho.png' % i
    fp_image_eval = op.join(image_path, fn_image_eval)

    plot_vstc_sliced_old(vstc_eval, vsrc, vstc_eval.tstep,
                         subjects_dir=fname.subjects_dir,
                         time=vstc_eval.tmin, cut_coords=config.cut_coords,
                         display_mode='ortho', figure=None,
                         axes=None, colorbar=True, cmap='magma',
                         symmetric_cbar='auto', threshold=0,
                         cbar_range=cbar_range_eval,
                         save=True, fname_save=fp_image_eval)

    if pick_ori == 'max-power':
        fn_image_ori = '%03d_dics_ori_ortho.png' % i
        fp_image_ori = op.join(image_path, fn_image_ori)

        plot_vstc_sliced_old(vstc_ori, vsrc, vstc_ori.tstep,
                             subjects_dir=fname.subjects_dir,
                             time=vstc_ori.tmin, cut_coords=config.cut_coords,
                             display_mode='ortho', figure=None,
                             axes=None, colorbar=True, cmap='magma',
                             symmetric_cbar='auto', threshold=0,
                             cbar_range=cbar_range_ori,
                             save=True, fname_save=fp_image_ori)

    ###############################################################################
    # Plot
    ###############################################################################

    html_table += '<tr><td>' + '</td><td>'.join([str(s) for s in setting]) + '</td>'
    html_table += '<td><img src="' + op.join(image_folder, fn_image_dist) + '"></td>'
    html_table += '<td><img src="' + op.join(image_folder, fn_image_eval) + '"></td>'
    if pick_ori == 'max-power':
        html_table += '<td><img src="' + op.join(image_folder, fn_image_ori) + '"></td>'
    else:
        html_table += '<td></td>'

    with open('html/dics_vol.html', 'w') as f:
        f.write(html_header)
        f.write(html_table)
        f.write(html_footer)
