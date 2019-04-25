import os.path as op
import mne
import numpy as np
from jumeg.jumeg_volmorpher import plot_vstc_sliced_grid

import config
from config import vfname
from utils import find_files

import pandas as pd

###############################################################################
# Plotting config
###############################################################################
display_mode = 'x'
coords = [-55, 55]
grid = [4, 6]
res_save = [1920, 1080]
threshold = 0

n_slices = grid[0] * grid[1]

coords_min = coords[0]
coords_max = coords[1]

step_size = (coords_max - coords_min) / float(n_slices)

cut_coords = np.arange(coords_min, coords_max, step_size) + 0.5 * step_size

title = 'name this appropriately'
fn_image = 'data/test_volume_loc_viz.png'

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

bf_type = 'lcmv'
noise = 0.1
pattern = 'volume_%s_results-noise%.1f-vertex*.csv' % (bf_type, noise)
fn_csv_list = find_files(op.join(config.target_path, 'csv'), pattern=pattern)

# TODO: create pandas data frame from csv files

for fn_csv in fn_csv_list:

    df = pd.read_csv(fn_csv, header=0)


###############################################################################
# Create stc from simulated data
###############################################################################

data = np.zeros(shape=(vertno.shape[0], 1))

vstc = mne.VolSourceEstimate(data=data, vertices=vertno, tmin=0,
                             tstep=1 / info['sfreq'], subject='sample')

###############################################################################
# Simulate a single signal dipole source as signal
###############################################################################

plot_vstc_sliced_grid(subjects_dir=vfname.subjects_dir, vstc=vstc, vsrc=vsrc,
                      title=title, time=vstc.times[0], display_mode=display_mode,
                      cut_coords=cut_coords, threshold=threshold,
                      only_positive_values=True, grid=grid,
                      res_save=res_save, fn_image=fn_image)
