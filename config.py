import argparse
import getpass
import os
from itertools import product
from socket import getfqdn

import numpy as np
from mne.datasets import sample
from mne.datasets.brainstorm import bst_phantom_ctf

from fnames import FileNames

user = getpass.getuser()  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts
print('Running on %s@%s' % (user, host))

if user == 'rodin':
    # My home laptop
    target_path = './data'
    n_jobs = 4
if user == 'wmvan':
    # My work laptop
    target_path = 'X:/'
    n_jobs = 8
elif user == 'ckiefer':
    target_path = '~/beamformer/data'
    n_jobs = 3
elif host == 'nbe-024.org.aalto.fi' and user == 'vanvlm1':
    # My workstation
    target_path = '/m/nbe/scratch/epasana/beamformer_simulation/data'
    n_jobs = 8
elif 'triton' in host and user == 'vanvlm1':
    # The big computational cluster at Aalto University
    target_path = '/scratch/nbe/epasana/beamformer_simulation/data'
    n_jobs = 1
elif user == 'we':
    target_path = '~/Documents/projects/beamf_sim/data'
    n_jobs = 2
elif user == '2628425':
    target_path = '/home/ECIT.QUB.AC.UK/2628425/nbe/scratch/epasana/beamformer_simulation/data'  # noqa
    n_jobs = 4
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
parser.add_argument('-s', '--subject', type=int, metavar='int', default=1,
                    help='Subject to analyse (for MEGSET analysis)')
args = parser.parse_args()

###############################################################################
# Settings related to the simulation
###############################################################################

trial_length = 2.0  # Length of a trial in seconds
tmin = -1.0
tmax = 1.0
# We have 109 seconds of empty room data
n_trials = int(109 / trial_length)  # Number of trials to simulate
signal_freq = 20  # Frequency at which to simulate the signal timecourse
signal_freq2 = 33  # Frequency at which to simulate the 2nd signal timecourse
n_neighbors_max = 1000  # maximum number of nearest neighbors being considered
# n_neighbors_max = 1  # maximum number of nearest neighbors being considered
noise_lowpass = 40  # Low-pass frequency for generating noise timecourses
noise = args.noise  # Multiplier for the noise dipoles

# Position of the signal
vertex = args.vertex
signal_hemi = 0
n_vertices = 3756  # Number of dipoles in the source space

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

if user == 'we':
    regs = [0.05, 0.1, 0.5]  # still plotting the old results
else:
    regs = [0, 0.05, 0.1]
sensor_types = ['grad', 'mag', 'joint']
pick_oris = [None, 'max-power']
inversions = ['single', 'matrix']
weight_norms = ['unit-noise-gain', None]
normalize_fwds = [True, False]
real_filters = [True, False]
use_noise_covs = [True, False]
reduce_ranks = [True, False]

dics_settings = list(product(
    regs, sensor_types, pick_oris, inversions, weight_norms, normalize_fwds,
    real_filters, use_noise_covs, reduce_ranks
))

lcmv_settings = list(product(
    regs, sensor_types, pick_oris, inversions, weight_norms, normalize_fwds,
    use_noise_covs, reduce_ranks
))

###############################################################################
# Settings related to plotting
###############################################################################

cut_coords = (-5, -30, 30)

###############################################################################
# True source locations for real datasets
###############################################################################

# FIXME replace with actual value determined by Hanna Renvall
somato_true_pos_ras = [36.9265, 7.85419, 53.4155]  # In RAS space, in mm
somato_true_pos = [0.03279403, 0.00966346, 0.10528801]  # In head space, in m
somato_true_pos = [-0.00445296, -0.0150457, 0.0552662]  # In head space, in m
somato_true_vert_idx = 5419

###############################################################################
# Colors for scatter plots
###############################################################################
# color pairs for visualization:
cols = dict(cherry='#5E2D25', orchid='#6F3746',
            magician='#6E4B6A', purple='#556486',
            sky='#277D8F', sea='#109281',
            forest='#50A164', spring='#91AA47')

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
fname.add('fwd_man', '{data_path}/MEG/sample/sample_coregerror-meg-vol-7-fwd.fif')  # noqa
fname.add('trans_man', 'sample_manual_ck-trans.fif')

# Files produced by volume simulation code
fname.add('target_path', target_path)  # Where to put everything
fname.add('fwd_discrete_true', '{data_path}/sample_audvis-meg-vol-7-discrete-fwd.fif')  # noqa
fname.add('fwd_discrete_man', '{data_path}/sample_coregerror-meg-vol-7-discrete-fwd.fif')  # noqa
fname.add('simulated_raw', '{target_path}/volume_simulated-raw-vertex{vertex:04d}-raw.fif')  # noqa
fname.add('stc_signal', '{target_path}/volume_stc_signal-vertex{vertex:04d}-vl.stc')  # noqa
fname.add('simulated_events', '{target_path}/volume_simulated-eve.fif')
fname.add('simulated_epochs', '{target_path}/volume_simulated-epochs-vertex{vertex:04d}-epo.fif')  # noqa
fname.add('report', '{target_path}/volume_report-vertex{vertex:04d}.h5')
fname.add('report_html', '{target_path}/volume_report-vertex{vertex:04d}.html')

fname.add('dics_results', '{target_path}/dics_results/dics_results-vertex{vertex:04d}-noise{noise:.1f}.csv')
fname.add('lcmv_results', '{target_path}/lcmv_results/lcmv_results-vertex{vertex:04d}-noise{noise:.1f}.csv')

# Files for parameter plots
fname.add('lcmv_params', 'lcmv.csv')  # noqa
fname.add('dics_params', 'dics.csv')  # noqa

# Brainstorm phantom data
phantom_fname = FileNames()
phantom_fname.add('data_path', bst_phantom_ctf.data_path())
phantom_fname.add('raw', '{data_path}/phantom_20uA_20150603_03.ds')
phantom_fname.add('ernoise', '{data_path}/emptyroom_20150709_01.ds')

# MEGSET results
fname.add('lcmv_megset_results', '{target_path}/lcmv_megset_results/lcmv_megset_results-subject{subject:d}.csv')  # noqa
fname.add('dics_megset_results', '{target_path}/dics_megset_results/dics_megset_results-subject{subject:d}.csv')  # noqa

# Somato results
fname.add('lcmv_somato_results', '{target_path}/lcmv_somato_results/lcmv_somato_results.csv')  # noqa
fname.add('dics_somato_results', '{target_path}/dics_somato_results/dics_somato_results.csv')  # noqa

# Set subjects_dir
os.environ['SUBJECTS_DIR'] = fname.subjects_dir
