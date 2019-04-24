import os
from socket import getfqdn
from fnames import FileNames
from mne.datasets import sample
import numpy as np
import argparse

user = os.environ['USER']  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts
print('Running on %s@%s' % (user, host))

if user == 'rodin':
    # My laptop
    target_path = './data'
    n_jobs = 4
elif host == 'nbe-024.org.aalto.fi' and user == 'vanvlm1':
    # My workstation
    target_path = '/m/nbe/scratch/epasana/beamformer_simulation/data'
    n_jobs = 8
elif 'triton' in host and user == 'vanvlm1':
    # The big computational cluster at Aalto University
    target_path = '/m/nbe/scratch/epasana/beamformer_simulation/data'
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

trial_length = 2.0 # Length of a trial in seconds
# We have 109 seconds of empty room data
n_trials = int(109 / trial_length)  # Number of trials to simulate
signal_freq = 10  # Frequency at which to simulate the signal timecourse
noise_lowpass = 40  # Low-pass frequency for generating noise timecourses
noise = args.noise  # Multiplyer for the noise dipoles

# Position of the signal
vertex = args.vertex
signal_hemi = 1

random = np.random.RandomState(42)  # Random seed for everything

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
fname.add('fwd', '{sample_folder}/sample_audvis-meg-eeg-oct-6-fwd.fif')
fname.add('trans', '{sample_folder}/sample_audvis_raw-trans.fif')

# Files produced by the simulation code
fname.add('target_path', target_path)  # Where to put everything
fname.add('stc_signal', '{target_path}/stc_signal-noise{noise}-vertex{vertex:04d}')
fname.add('simulated_raw', '{target_path}/simulated-noise{noise}-vertex{vertex:04d}-raw.fif')
fname.add('simulated_events', '{target_path}/simulated-eve.fif')
fname.add('simulated_epochs', '{target_path}/simulated-epochs-noise{noise}-vertex{vertex:04d}-epo.fif')
fname.add('report', '{target_path}/report-noise{noise}-vertex{vertex:04d}.h5')
fname.add('report_html', '{target_path}/report-noise{noise}-vertex{vertex:04d}.html')
fname.add('dics_results', '{target_path}/dics_results-noise{noise}-vertex{vertex:04d}.csv')
fname.add('lcmv_results', '{target_path}/lcmv_results-noise{noise}-vertex{vertex:04d}.csv')
fname.add('mne_results', '{target_path}/mne_results-noise{noise}-vertex{vertex:04d}.csv')

# Set subjects_dir
os.environ['SUBJECTS_DIR'] = fname.subjects_dir
