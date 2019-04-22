import os.path as op
import mne
import numpy as np
import matplotlib.pyplot as plt

from mne.datasets import sample
from time_series import generate_signal


data_path = sample.data_path()

raw_fname = op.join(data_path, 'MEG/sample/sample_audvis_raw.fif')
bem_fname = op.join(data_path, 'subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif')
src_fname = op.join(data_path, 'subjects/sample/bem/sample-oct-6-src.fif')
fwd_fname = op.join(data_path, 'MEG/sample/sample_audvis-meg-eeg-oct-6-fwd.fif')
trans_fname = op.join(data_path, 'MEG/sample/sample_audvis_raw-trans.fif')

info = mne.io.read_info(raw_fname)
src = mne.read_source_spaces(src_fname)
fwd = mne.read_forward_solution(fwd_fname)

n_dipoles = 1
times = np.arange(0, info['sfreq']) / info['sfreq']

stc = mne.simulation.simulate_sparse_stc(src, n_dipoles, times, data_fun=generate_signal, random_state=42)


data = np.asarray([generate_signal(times, freq=10)])

possible_vertices = src[1]['vertno']

vertices = [np.array([]), np.array([possible_vertices[0]])]

#vertices = stc.vertices

stc = mne.SourceEstimate(data=data, vertices=vertices, tmin=0,
                         tstep=1/info['sfreq'], subject='sample')

raw = mne.simulation.simulate_raw(info, stc, trans=None, src=None, bem=None, forward=fwd, duration=1)
raw.plot()
