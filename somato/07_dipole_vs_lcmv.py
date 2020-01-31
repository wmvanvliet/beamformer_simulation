import os.path as op

import mne
import numpy as np
import pandas as pd
from jumeg.jumeg_volume_plotting import plot_vstc_sliced_old
from mne.beamformer import make_lcmv, apply_lcmv

from config import fname, subject_id
from config_sim import lcmv_settings
from utils_sim import make_dipole_volume, set_directory

report = mne.open_report(fname.report)

###############################################################################
# Load the data
###############################################################################

epochs = mne.read_epochs(fname.epochs)
trans = mne.transforms.read_trans(fname.trans)
fwd = mne.read_forward_solution(fname.fwd)

###############################################################################
# Sensor-level analysis for beamformer
###############################################################################

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make cov matrices
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='shrunk', rank='info')
data_cov = mne.compute_covariance(epochs, tmin=0, tmax=0.4, method='empirical', rank='info')

# Compute evokeds
tmin = 0.03
tmax = 0.05
evoked_grad = epochs_grad.average().crop(tmin, tmax)
evoked_mag = epochs_mag.average().crop(tmin, tmax)
evoked_joint = epochs_joint.average().crop(tmin, tmax)

###############################################################################
# read dipole created by 06_dipole.py
###############################################################################

dip = mne.read_dipole(fname.ecd)
# get the position of the dipole in MRI coordinates
mri_pos = mne.head_to_mri(dip.pos, mri_head_t=trans,
                          subject=subject_id, subjects_dir=fname.subjects_dir)

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
        <th>Dipole location vs. LCMV activity</th>
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
                filters_row_index: 1,
                enable_checklist_reset_filter: false,
                alternate_rows: true,
                sticky_headers: true,
                col_types: [
                    'number', 'string', 'string',
                    'string', 'string', 'string',
                    'string', 'string', 'image'
                ],
                col_widths: [
                    '80px', '150px', '130px',
                    '110px', '170px', '150px',
                    '150px', '150px', '210px'
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

###############################################################################
# Set up directories
###############################################################################

img_folder = op.join('somato', 'dip_vs_lcmv')
html_path = op.join('..', 'html')
image_path = op.join(html_path, img_folder)
set_directory(image_path)

###############################################################################
# Compute LCMV solution and plot stc at dipole location
###############################################################################

dists = []

for ii, setting in enumerate(lcmv_settings):

    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, use_noise_cov, reduce_rank = setting
    try:

        if sensor_type == 'grad':
            evoked = evoked_grad
        elif sensor_type == 'mag':
            evoked = evoked_mag
        elif sensor_type == 'joint':
            evoked = evoked_joint
        else:
            raise ValueError('Invalid sensor type: %s', sensor_type)

        filters = make_lcmv(evoked.info, fwd, data_cov, reg=reg,
                            pick_ori=pick_ori, weight_norm=weight_norm,
                            inversion=inversion, normalize_fwd=normalize_fwd,
                            noise_cov=noise_cov if use_noise_cov else None,
                            reduce_rank=reduce_rank)

        stc = apply_lcmv(evoked, filters)
        stc = abs(stc.mean())

        # Compute distance between true and estimated source
        dip_est = make_dipole_volume(stc, fwd['src'])
        dist = np.linalg.norm(dip.pos - dip_est.pos)

        fn_image = str(ii).zfill(3) + '_lcmv_' + str(setting).replace(' ', '') + '.png'
        fp_image = op.join(image_path, fn_image)

        if not op.exists(fp_image):
            cbar_range = [stc.data.min(), stc.data.max()]
            threshold = np.percentile(stc.data, 99.5)
            plot_vstc_sliced_old(stc, vsrc=fwd['src'], tstep=stc.tstep,
                                 subjects_dir=fname.subjects_dir,
                                 time=stc.tmin, cut_coords=mri_pos[0],
                                 display_mode='ortho', figure=None,
                                 axes=None, colorbar=True, cmap='magma',
                                 symmetric_cbar='auto', threshold=threshold,
                                 cbar_range=cbar_range,
                                 save=True, fname_save=fp_image)

        ###############################################################################
        # save to html
        ###############################################################################

        html_table += '<tr><td>' + '</td><td>'.join([str(s) for s in setting]) + '</td>'
        html_table += '<td><img src="' + op.join(img_folder, fn_image) + '"></td>'

        with open(op.join(html_path, 'dip_vs_lcmv_vol.html'), 'w') as f:
            f.write(html_header)
            f.write(html_table)
            f.write(html_footer)

    except Exception as e:
        print(e)
        dist = np.nan

    print(setting, dist)
    dists.append(dist)

###############################################################################
# Save everything to a pandas dataframe
###############################################################################

df = pd.DataFrame(lcmv_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'use_noise_cov', 'reduce_rank'])

df['dist'] = dists
df.to_csv(fname.dip_vs_lcmv_results)
print('OK!')
