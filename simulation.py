import os.path as op
import mne
import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw
from time_series import generate_signal, generate_random
from utils import add_stcs


data_path = sample.data_path()

raw_fname = op.join(data_path, 'MEG/sample/sample_audvis_raw.fif')
er_raw_fname = op.join(data_path, 'MEG/sample/ernoise_raw.fif')
bem_fname = op.join(data_path, 'subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')
src_fname = op.join(data_path, 'subjects/sample/bem/sample-oct-6-orig-src.fif')
fwd_fname = op.join(data_path, 'MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif')
trans_fname = op.join(data_path, 'MEG/sample/sample_audvis_raw-trans.fif')

info = mne.io.read_info(raw_fname)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
fwd = mne.read_forward_solution(fwd_fname)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
src = fwd['src']

times = np.arange(0, info['sfreq']) / info['sfreq']
lh_vertno = src[0]['vertno']
rh_vertno = src[1]['vertno']


###############################################################################
# Simulate a single signal dipole
###############################################################################

data = np.asarray([generate_signal(times, freq=10)])
vertices = [np.array([], dtype=np.int64), np.array([rh_vertno[0]], dtype=np.int64)]
stc_signal = mne.SourceEstimate(data=data, vertices=vertices, tmin=0,
                                tstep=1/info['sfreq'], subject='sample')


###############################################################################
# Simulate random noise dipoles
###############################################################################
n_noise_dipoles = 50
stc_noise = simulate_sparse_stc(
    src,
    n_noise_dipoles,
    times,
    data_fun=generate_random,
    random_state=42
)


###############################################################################
# Project to sensor space
###############################################################################
stc = add_stcs(stc_signal, 0.5 * stc_noise)
raw = simulate_raw(
    info,
    stc,
    trans=None,
    src=None,
    bem=None,
    forward=fwd,
    duration=1,
)

###############################################################################
# Use empty room noise as sensor noise
###############################################################################
er_raw = mne.io.read_raw_fif(data_path + '/MEG/sample/ernoise_raw.fif',
                             preload=True) 
raw_picks = mne.pick_types(raw.info, meg=True, eeg=False)
er_raw_picks = mne.pick_types(er_raw.info, meg=True, eeg=False)
raw._data[raw_picks] += er_raw._data[er_raw_picks, :len(raw.times)]


###############################################################################
# Plot it!
###############################################################################
raw.plot()
