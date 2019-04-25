import mne
import numpy as np
import config
from config import vfname
import pandas as pd

from tqdm import tqdm
from itertools import product
from utils import plot_vstc_grid, set_directory


###############################################################################
# Plotting config
###############################################################################
display_mode = 'x'
coords_x = [-55, 55]
coords_z = [-55, 70]

if display_mode == 'x':
    coords = coords_x
elif display_mode == 'z':
    coords = coords_z

grid = [4, 6]
res_save = [1920, 1080]
threshold = 0.0001

title = ''

###############################################################################
# Load volume source space
###############################################################################

info = mne.io.read_info(vfname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))

fwd = mne.read_forward_solution(vfname.fwd)
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
        df = pd.read_csv(vfname.lcmv_results(noise=config.noise, vertex=vertex))
        df['vertex'] = vertex
        df['noise'] = config.noise
        dfs.append(df)
    except Exception as e:
        print(e)
lcmv = pd.concat(dfs, ignore_index=True)
lcmv['pick_ori'].fillna('none', inplace=True)
lcmv['weight_norm'].fillna('none', inplace=True)

###############################################################################
# Construct lcmv settings list
###############################################################################

regs = [0.05, 0.1, 0.5]
sensor_types = ['joint', 'grad', 'mag']
pick_oris = ['none', 'normal', 'max-power']
weight_norms = ['unit-noise-gain', 'none']
use_noise_covs = [True, False]
depths = [True, False]
settings = list(product(regs, sensor_types, pick_oris, weight_norms,
                        use_noise_covs, depths))

html_header = (
    '<html><head><link rel="stylesheet" type="text/css" href="style.css"></head><body>'
    '<table><tr>'
    '<th>reg</th>'
    '<th>sensor type</th>'
    '<th>pick_ori</th>'
    '<th>weight_norm</th>'
    '<th>use_noise_cov</th>'
    '<th>depth</th>'
    '<th>P2P distance</th>'
    '<th>Fancy metric</th>'
    '</tr>')

html_footer = '</body></table>'

html_table = ''

set_directory('html/lcmv')

for i, setting in enumerate(settings):
    # construct query
    q = ("reg==%.1f and sensor_type=='%s' and pick_ori=='%s' and "
         "weight_norm=='%s' and use_noise_cov==%s and depth==%s" % setting)

    print(q)

    sel = lcmv.query(q).dropna()

    reg, sensor_type, pick_ori, weight_norm, use_noise_cov, depth = setting

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

    fn_image = 'html/lcmv/%03d_dist_%s.png' % (i, display_mode)

    plot_vstc_grid(vstc_dist, vsrc, subjects_dir=vfname.subjects_dir,
                   time=vstc_dist.tmin, only_positive_values=True,
                   coords=coords, grid=grid, threshold=threshold,
                   display_mode=display_mode, fn_save=fn_image)

    fn_image = 'html/lcmv/%03d_eval_%s.png' % (i, display_mode)

    plot_vstc_grid(vstc_eval, vsrc, subjects_dir=vfname.subjects_dir,
                   time=vstc_eval.tmin, only_positive_values=True,
                   coords=coords, grid=grid, threshold=threshold,
                   display_mode=display_mode, fn_save=fn_image)

    ###############################################################################
    # Plot
    ###############################################################################

    html_table += '<tr><td>' + '</td><td>'.join([str(s) for s in setting]) + '</td>'
    html_table += '<td><img src="lcmv/%03d_dist_%s.png"></td>' % (i, display_mode)
    html_table += '<td><img src="lcmv/%03d_eval_%s.png"></td>' % (i, display_mode)

    with open('html/lcmv.html', 'w') as f:
        f.write(html_header)
        f.write(html_table)
        f.write(html_footer)

