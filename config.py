import os
from fnames import FileNames
from mne.datasets import sample

n_trials = 109  # Number of trials to simulate
signal_freq = 10 # Frequency at which to simulate the signal timecourse
noise_lowpass = 40  # Low-pass frequency for generating noise timecourses
SNR = 0.1  # Ratio noise to signal (not really SNR right now)

random_state = 42  # Random seed for everything

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
fname.add('target_path', '.')  # Where to put everything
fname.add('simulated_raw', '{target_path}/simulated-raw.fif')
fname.add('simulated_events', '{target_path}/simulated-eve.fif')
fname.add('simulated_epochs', '{target_path}/simulated-epochs.fif')
fname.add('report', '{target_path}/report.h5')
fname.add('report_html', '{target_path}/report.html')

# Set subjects_dir
os.environ['SUBJECTS_DIR'] = fname.subjects_dir
