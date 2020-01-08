import argparse
import os
from socket import getfqdn
from itertools import product

import numpy as np
from mne.datasets import sample, somato
from mne.datasets.brainstorm import bst_phantom_ctf

from fnames import FileNames

user = os.environ['USER']  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts
print('Running on %s@%s' % (user, host))

if user == 'rodin':
    # My laptop
    target_path = './data'
    n_jobs = 4
elif user == 'ckiefer':
    target_path = './data'
    n_jobs = 3
elif host == 'nbe-024.org.aalto.fi' and user == 'vanvlm1':
    # My workstation
    target_path = '/m/nbe/scratch/epasana/beamformer_simulation/data'
    n_jobs = 8
elif 'triton' in host and user == 'vanvlm1':
    # The big computational cluster at Aalto University
    target_path = '/scratch/nbe/epasana/beamformer_simulation/data'
    n_jobs = 1
else:
    raise RuntimeError('Please edit scripts/config.py and set the target_path '
                       'variable to point to the location where the data '
                       'should be stored and the n_jobs variable to the '
                       'number of CPU cores the analysis is allowed to use.')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Beamformer simulator')
parser.add_argument('-n', '--noise', type=float, metavar='float', default=0.1,
                    help='Amount of noise to add')
parser.add_argument('-v', '--vertex', type=int, metavar='int', default=2000,
                    help='Vertex index of the signal dipole')
args = parser.parse_args()

###############################################################################
# Settings related to the simulation
###############################################################################

trial_length = 2.0  # Length of a trial in seconds
# We have 109 seconds of empty room data
n_trials = int(109 / trial_length)  # Number of trials to simulate
signal_freq = 10  # Frequency at which to simulate the signal timecourse
signal_freq2 = 30  # Frequency at which to simulate the second signal timecourse
#n_neighbors_max = 1000  # maximum number of nearest neighbors being considered
n_neighbors_max = 1  # maximum number of nearest neighbors being considered
noise_lowpass = 40  # Low-pass frequency for generating noise timecourses
noise = args.noise  # Multiplier for the noise dipoles

# Position of the signal
vertex = args.vertex
signal_hemi = 1
n_vertices = 3765  # Number of dipoles in the source space

random = np.random.RandomState(vertex)  # Random seed for everything

###############################################################################
# Settings grid for beamformers
###############################################################################

# Compare
#   - vector vs. scalar (max-power orientation)
#   - Array-gain BF (leadfield normalization)
#   - Unit-gain BF ('vanilla' LCMV)
#   - Unit-noise-gain BF (weight normalization)
#   - pre-whitening (noise-covariance)
#   - different sensor types
#   - what changes with condition contrasting

regs = [0.05, 0.1, 0.5]
sensor_types = ['grad', 'mag', 'joint']
pick_oris = [None, 'max-power']
inversions = ['single', 'matrix']
weight_norms = ['unit-noise-gain', 'nai', None]
normalize_fwds = [True, False]
real_filters = [True, False]
use_noise_covs = [True, False]

dics_settings = list(product(
    regs, sensor_types, pick_oris, inversions, weight_norms, normalize_fwds,
    real_filters, use_noise_covs,
))

lcmv_settings = list(product(
    regs, sensor_types, pick_oris, inversions, weight_norms, normalize_fwds,
    use_noise_covs,
))

###############################################################################
# True source locations for real datasets
###############################################################################

# FIXME replace with actual value determined by Hanna Renvall
somato_true_pos = [0.042, 0.009, 0.105]  # In head space, in meters

###############################################################################
# Filenames for various things
###############################################################################
fname = FileNames()

n_noise_dipoles_vol = 150  # number of noise_dipoles in volume source space

# Filenames for various volume source space related things
fname = FileNames()
# Files from MNE-sample dataset
fname.add('data_path', sample.data_path())
fname.add('subjects_dir', '{data_path}/subjects')
fname.add('bem_folder', '{data_path}/subjects/sample/bem')
fname.add('bem', '{bem_folder}/sample-5120-bem-sol.fif')
fname.add('src', '{bem_folder}/volume-7mm-src.fif')
fname.add('sample_folder', '{data_path}/MEG/sample')
fname.add('sample_raw', '{sample_folder}/sample_audvis_raw.fif')
fname.add('ernoise', '{sample_folder}/ernoise_raw.fif')
fname.add('aseg', '{data_path}/subjects/sample/mri/aseg.mgz')
fname.add('fwd_true', '{data_path}/MEG/sample/sample_audvis-meg-vol-7-fwd.fif')
fname.add('trans_true', '{sample_folder}/sample_audvis_raw-trans.fif')

# Files from manual coregistration
fname.add('fwd_man', '{data_path}/MEG/sample/sample_coregerror-meg-vol-7-fwd.fif')
fname.add('trans_man', '{sample_folder}/sample_manual_bw-trans.fif')

# Files produced by volume simulation code
fname.add('target_path', target_path)  # Where to put everything
fname.add('fwd_discrete_true', '{data_path}/sample_audvis-meg-vol-7-discrete-fwd.fif')
fname.add('fwd_discrete_man', '{data_path}/sample_coregerror-meg-vol-7-discrete-fwd.fif')
fname.add('simulated_raw', '{target_path}/volume_simulated-raw-vertex{vertex:04d}-raw.fif')
fname.add('stc_signal', '{target_path}/volume_stc_signal-vertex{vertex:04d}-vl.stc')
fname.add('simulated_events', '{target_path}/volume_simulated-eve.fif')
fname.add('simulated_epochs', '{target_path}/volume_simulated-epochs-vertex{vertex:04d}-epo.fif')
fname.add('report', '{target_path}/volume_report-vertex{vertex:04d}.h5')
fname.add('report_html', '{target_path}/volume_report-vertex{vertex:04d}.html')
fname.add('lcmv_results', '{target_path}/lcmv_results-vertex{vertex:04d}.csv')
fname.add('lcmv_results_2s', '{target_path}/lcmv_results-2sources-vertex{vertex:04d}.csv')
fname.add('dics_results', '{target_path}/dics_results-vertex{vertex:04d}.csv')
fname.add('dics_results_2s', '{target_path}/dics_results-2sources-vertex{vertex:04d}.csv')

# Brainstorm phantom data
phantom_fname = FileNames()
phantom_fname.add('data_path', bst_phantom_ctf.data_path()) 
phantom_fname.add('raw', '{data_path}/phantom_20uA_20150603_03.ds')
phantom_fname.add('ernoise', '{data_path}/emptyroom_20150709_01.ds')

# MNE-Somato data set
fname.add('somato_path', somato.data_path())
fname.add('somato_derivatives', '{somato_path}/derivatives/sub-01')
fname.add('somato_subjects', '{somato_path}/derivatives/freesurfer/subjects')
fname.add('somato_src', '{somato_derivatives}/sub-01_task-somato_vol-src.fif')
fname.add('somato_fwd', '{somato_derivatives}/sub-01_task-somato_vol-fwd.fif')
fname.add('somato_epochs', '{somato_derivatives}/sub-01_task-somato_epo.fif')
fname.add('somato_epochs_long', '{somato_derivatives}/sub-01_task-somato_long_epo.fif')
fname.add('lcmv_somato_results', '{target_path}/lcmv_results-somato.csv')
fname.add('dics_somato_results', '{target_path}/dics_results-somato.csv')

# Set subjects_dir
os.environ['SUBJECTS_DIR'] = fname.subjects_dir
