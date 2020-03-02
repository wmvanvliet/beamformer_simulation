import os.path as op

import mne
import numpy as np
import pandas as pd
from jumeg.jumeg_volume_plotting import plot_vstc_sliced_old
from mne.beamformer import make_dics, apply_dics_csd

from config import dics_settings
from somato.config import fname, subject_id
from utils import make_dipole_volume, set_directory, evaluate_fancy_metric_volume

report = mne.open_report(fname.report)

###############################################################################
# Load the data
###############################################################################

# Read longer epochs
epochs = mne.read_epochs(fname.epochs_long).pick_types(meg=True)
trans = mne.transforms.read_trans(fname.trans)
fwd = mne.read_forward_solution(fname.fwd)

###############################################################################
# Sensor level analysis
###############################################################################

info_grad = epochs.copy().pick_types(meg='grad').info
info_mag = epochs.copy().pick_types(meg='mag').info
info_joint = epochs.copy().pick_types(meg=True).info

###############################################################################
# Compute Cross-Spectral Density matrices
###############################################################################

freqs = np.logspace(np.log10(12), np.log10(30), 9)
csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-1, tmax=1.5, decim=5, n_jobs=4)
csd_baseline = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-1, tmax=0, decim=5, n_jobs=4)
# ERS activity starts at 0.5 seconds after stimulus onset
csd_ers = mne.time_frequency.csd_morlet(epochs, freqs, tmin=0.5, tmax=1.5, decim=5, n_jobs=4)

csd = csd.mean()
csd_baseline = csd_baseline.mean()
csd_ers = csd_ers.mean()

###############################################################################
# read dipole created by 06_dipole.py
###############################################################################

dip = mne.read_dipole(fname.ecd)
# get the position of the dipole in MRI coordinates
mri_pos = mne.head_to_mri(dip.pos, mri_head_t=trans,
                          subject=subject_id, subjects_dir=fname.subjects_dir)

# get true_vert_idx
rr = fwd['src'][0]['rr']
inuse = fwd['src'][0]['inuse']
indices = np.where(fwd['src'][0]['inuse'])[0]
rr_inuse = rr[indices]
true_vert_idx = np.where(np.linalg.norm(rr_inuse - dip.pos, axis=1) ==
                         np.linalg.norm(rr_inuse - dip.pos, axis=1).min())[0][0]

###############################################################################
# HTML settings
###############################################################################

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
        <th>Dipole location vs. DICS power</th>
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
                filters_row_index: 1,
                enable_checklist_reset_filter: false,
                alternate_rows: true,
                sticky_headers: true,
                col_types: [
                    'number', 'string', 'string',
                    'string', 'string', 'string',
                    'string', 'string', 'string',
                    'image'
                ],
                col_widths: [
                    '80px', '150px', '130px',
                    '110px', '170px', '150px',
                    '150px', '150px', '150px',
                    '210px'
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

img_folder = op.join('somato', 'dip_vs_dics')
html_path = 'html'
image_path = op.join(html_path, img_folder)
set_directory(image_path)

###############################################################################
# Compute DICS solution and plot stc at dipole location
###############################################################################

dists = []
evals = []
ori_errors = []

for ii, setting in enumerate(dics_settings):

    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, real_filter, use_noise_cov, reduce_rank = setting
    try:
        if sensor_type == 'grad':
            info = info_grad
        elif sensor_type == 'mag':
            info = info_mag
        elif sensor_type == 'joint':
            info = info_joint
        else:
            raise ValueError('Invalid sensor type: %s', sensor_type)

        filters = make_dics(info, fwd, csd, reg=reg, pick_ori=pick_ori,
                            inversion=inversion, weight_norm=weight_norm,
                            noise_csd=csd_baseline if use_noise_cov else None,
                            normalize_fwd=normalize_fwd,
                            real_filter=real_filter, reduce_rank=reduce_rank)

        # Compute source power
        stc_baseline, _ = apply_dics_csd(csd_baseline, filters)
        stc_ers, _ = apply_dics_csd(csd_ers, filters)

        stc_ers_orig = stc_ers

        # Normalize with baseline power.
        stc_ers_norm_log = stc_ers / stc_baseline
        stc_ers_norm_log.data = np.log(stc_ers_norm_log.data)

        stc_ers = stc_ers_norm_log

        # Compute distance between true and estimated source
        dip_est = make_dipole_volume(stc_ers, fwd['src'])
        dist = np.linalg.norm(dip.pos - dip_est.pos)

        # Fancy evaluation metric
        # TODO: where to evaluate fancy metric? before or after normalization, i.e., stc_ers_orig or stc_ers_norm_log?
        ev = evaluate_fancy_metric_volume(stc_ers, true_vert_idx=true_vert_idx)

        if pick_ori == 'max-power':
            estimated_ori = filters['max_power_oris'][0][true_vert_idx]
            ori_error = np.rad2deg(abs(np.arccos(estimated_ori @ dip.ori[0])))
        else:
            ori_error = np.nan

        fn_image = str(ii).zfill(3) + '_dics_' + str(setting).replace(' ', '') + '.png'
        fp_image = op.join(image_path, fn_image)

        if not op.exists(fp_image):
            cbar_range = [stc_ers.data.min(), stc_ers.data.max()]
            threshold = np.percentile(stc_ers.data, 99.5)
            plot_vstc_sliced_old(stc_ers, vsrc=fwd['src'], tstep=stc_ers.tstep,
                                 subjects_dir=fname.subjects_dir,
                                 time=stc_ers.tmin, cut_coords=mri_pos[0],
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

        with open(op.join(html_path, 'dip_vs_dics_vol.html'), 'w') as f:
            f.write(html_header)
            f.write(html_table)
            f.write(html_footer)

    except Exception as e:
        print(e)
        dist = np.nan
        ev = np.nan
        ori_error = np.nan

    print(setting, dist)
    dists.append(dist)
    evals.append(ev)
    ori_errors.append(ori_error)

###############################################################################
# Save everything to a pandas dataframe
###############################################################################

df = pd.DataFrame(dics_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'real_filter',
                           'use_noise_cov', 'reduce_rank'])

df['dist'] = dists
df['eval'] = evals
df['ori_error'] = ori_errors

df.to_csv(fname.dip_vs_dics_results)
print('OK!')
