import argparse
import os
from socket import getfqdn

import numpy as np
from mne.datasets import sample
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
parser.add_argument('-n', '--noise', type=float, metavar='float', default=1,
                    help='Amount of noise to add')
parser.add_argument('-v', '--vertex', type=int, metavar='int', default=2000,
                    help='Vertex index of the signal dipole')
args = parser.parse_args()

trial_length = 2.0  # Length of a trial in seconds
# We have 109 seconds of empty room data
n_trials = int(109 / trial_length)  # Number of trials to simulate
signal_freq = 10  # Frequency at which to simulate the signal timecourse
signal_freq2 = 30  # Frequency at which to simulate the second signal timecourse
n_neighbors_max = 1000  # maximum number of nearest neighbors being considered
noise_lowpass = 40  # Low-pass frequency for generating noise timecourses
noise = args.noise  # Multiplier for the noise dipoles

# Position of the signal
vertex = args.vertex
signal_hemi = 1

random = np.random.RandomState(vertex)  # Random seed for everything

# Filenames for various things
fname = FileNames()

# Files from MNE-sample dataset
fname.add('data_path', sample.data_path())
fname.add('subjects_dir', '{data_path}/subjects')
fname.add('bem_folder', '{data_path}/subjects/sample/bem')
fname.add('sample_folder', '{data_path}/MEG/sample')
fname.add('sample_raw', '{sample_folder}/sample_audvis_raw.fif')
fname.add('ernoise', '{sample_folder}/ernoise_raw.fif')
fname.add('bem', '{bem_folder}/sample-5120-5120-5120-bem-sol.fif')
fname.add('src', '{bem_folder}/sample-oct-6-orig-src.fif')
fname.add('fwd_true', '{sample_folder}/sample_audvis-meg-eeg-oct-6-fwd.fif')
fname.add('trans_true', '{sample_folder}/sample_audvis_raw-trans.fif')

# Files from manual coregistration
fname.add('fwd_man', '{sample_folder}/sample_coregerror-meg-eeg-oct-6-fwd.fif')
fname.add('trans_man', 'sample_manual_ck-trans.fif')
fname.add('trans_man_bw', 'sample_manual_bw-trans.fif')

# Files produced by the simulation code
fname.add('target_path', target_path)  # Where to put everything
fname.add('stc_signal', '{target_path}/stc_signal-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}')
fname.add('simulated_raw', '{target_path}/simulated-raw-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}-raw.fif')
fname.add('simulated_events', '{target_path}/simulated-eve.fif')
fname.add('simulated_epochs', '{target_path}/simulated-epochs-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}-epo.fif')
fname.add('report', '{target_path}/report-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}.h5')
fname.add('report_html', '{target_path}/report-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}.html')
fname.add('dics_results', '{target_path}/dics_results-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}.csv')
fname.add('dics_results_2s', '{target_path}/dics_results_2sources-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}.csv')
fname.add('lcmv_results', '{target_path}/lcmv_results-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}.csv')
fname.add('lcmv_results_2s', '{target_path}/lcmv_results_2sources-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}.csv')
fname.add('mne_results', '{target_path}/mne_results-noise{noise:.1f}-vertex{vertex:04d}-hemi{hemi:d}.csv')


# volume source space specific things

n_noise_dipoles_vol = 150  # number of noise_dipoles in volume source space

# Filenames for various volume source space related things
vfname = FileNames()
# Files from MNE-sample dataset
vfname.add('data_path', sample.data_path())
vfname.add('subjects_dir', '{data_path}/subjects')
vfname.add('bem_folder', '{data_path}/subjects/sample/bem')
vfname.add('bem', '{bem_folder}/sample-5120-bem-sol.fif')
vfname.add('src', '{bem_folder}/volume-7mm-src.fif')
vfname.add('sample_folder', '{data_path}/MEG/sample')
vfname.add('sample_raw', '{sample_folder}/sample_audvis_raw.fif')
vfname.add('ernoise', '{sample_folder}/ernoise_raw.fif')
vfname.add('aseg', '{data_path}/subjects/sample/mri/aseg.mgz')
vfname.add('fwd_true', '{data_path}/MEG/sample/sample_audvis-meg-vol-7-fwd.fif')
vfname.add('trans_true', '{sample_folder}/sample_audvis_raw-trans.fif')

# Files from manual coregistration
vfname.add('fwd_man', '{data_path}/MEG/sample/sample_coregerror-meg-vol-7-fwd.fif')
vfname.add('trans_man', '{sample_folder}/sample_manual_bw-trans.fif')

# Files produced by volume simulation code
vfname.add('target_path', target_path)  # Where to put everything
vfname.add('fwd_discrete_true', '{data_path}/sample_audvis-meg-vol-7-discrete-fwd.fif')
vfname.add('fwd_discrete_man', '{data_path}/sample_coregerror-meg-vol-7-discrete-fwd.fif')
vfname.add('simulated_raw', '{target_path}/volume_simulated-raw-noise{noise}-vertex{vertex:04d}-raw.fif')
vfname.add('stc_signal', '{target_path}/volume_stc_signal-noise{noise}-vertex{vertex:04d}-vl.stc')
vfname.add('simulated_events', '{target_path}/volume_simulated-eve.fif')
vfname.add('simulated_epochs', '{target_path}/volume_simulated-epochs-noise{noise}-vertex{vertex:04d}-epo.fif')
vfname.add('report', '{target_path}/volume_report-noise{noise}-vertex{vertex:04d}.h5')
vfname.add('report_html', '{target_path}/volume_report-noise{noise}-vertex{vertex:04d}.html')
vfname.add('dics_results', '{target_path}/volume_dics_results-noise{noise}-vertex{vertex:04d}.csv')
vfname.add('dics_results_2s', '{target_path}/volume_dics_results_2sources-noise{noise}-vertex{vertex:04d}.csv')
vfname.add('lcmv_results', '{target_path}/volume_lcmv_results-noise{noise}-vertex{vertex:04d}.csv')
vfname.add('lcmv_results_2s', '{target_path}/volume_lcmv_results_2sources-noise{noise}-vertex{vertex:04d}.csv')
vfname.add('mne_results', '{target_path}/volume_mne_results-noise{noise}-vertex{vertex:04d}.csv')

# Brainstorm phantom data
phantom_fname = FileNames()
phantom_fname.add('data_path', bst_phantom_ctf.data_path()) 
phantom_fname.add('raw', '{data_path}/phantom_20uA_20150603_03.ds')
phantom_fname.add('ernoise', '{data_path}/emptyroom_20150709_01.ds')

# Set subjects_dir
os.environ['SUBJECTS_DIR'] = fname.subjects_dir
