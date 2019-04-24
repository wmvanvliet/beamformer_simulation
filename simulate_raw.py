import mne
import numpy as np
from tqdm import tqdm

from mne.simulation import simulate_sparse_stc, simulate_raw
from time_series import generate_signal, generate_random
from utils import add_stcs
from matplotlib import pyplot as plt

import config
from config import fname

info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
fwd = mne.read_forward_solution(fname.fwd)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
src = fwd['src']
labels = mne.read_labels_from_annot(subject='sample', parc='aparc.a2009s')

times = np.arange(0, config.trial_length * info['sfreq']) / info['sfreq']
lh_vertno = src[0]['vertno']
rh_vertno = src[1]['vertno']
n_noise_dipoles = len(labels)


###############################################################################
# Simulate a single signal dipole source as signal
###############################################################################

signal_vertex = src[config.signal_hemi]['vertno'][config.vertex]
data = np.asarray([generate_signal(times, freq=config.signal_freq)])
vertices = [np.array([], dtype=np.int64), np.array([], dtype=np.int64)]
vertices[config.signal_hemi] = np.array([signal_vertex])
stc_signal = mne.SourceEstimate(data=data, vertices=vertices, tmin=0,
                                tstep=1 / info['sfreq'], subject='sample')
stc_signal.save(fname.stc_signal(noise=config.noise, vertex=config.vertex))


###############################################################################
# Create trials of simulated data
###############################################################################

raw_list = []
for i in range(config.n_trials):
    ###########################################################################
    # Simulate random noise dipoles
    ###########################################################################
    stc_noise = simulate_sparse_stc(
        src,
        n_noise_dipoles,
        times,
        data_fun=generate_random,
        random_state=config.random,
        labels=labels
    )

    ###########################################################################
    # Project to sensor space
    ###########################################################################
    stc = add_stcs(stc_signal, config.noise * stc_noise)
    raw = simulate_raw(
        info,
        stc,
        trans=None,
        src=None,
        bem=None,
        forward=fwd,
        duration=config.trial_length,
        cov=None,
        random_state=config.random,
    )

    raw_list.append(raw)
    print('%02d/%02d' % (i + 1, config.n_trials))

raw = mne.concatenate_raws(raw_list)

###############################################################################
# Use empty room noise as sensor noise
###############################################################################
er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True) 
raw_picks = mne.pick_types(raw.info, meg=True, eeg=False)
er_raw_picks = mne.pick_types(er_raw.info, meg=True, eeg=False)
raw._data[raw_picks] += er_raw._data[er_raw_picks, :len(raw.times)]

###############################################################################
# Save everything
###############################################################################

raw.save(fname.simulated_raw(noise=config.noise, vertex=config.vertex), overwrite=True)


###############################################################################
# Plot it!
###############################################################################
with mne.open_report(fname.report(noise=config.noise, vertex=config.vertex)) as report:
    fig = plt.figure()
    plt.plot(times, generate_signal(times, freq=10))
    plt.xlabel('Time (s)')
    report.add_figs_to_section(fig, 'Signal time course',
                               section='Sensor-level', replace=True)

    fig = raw.plot()
    report.add_figs_to_section(fig, 'Simulated raw', section='Sensor-level',
                               replace=True)
    report.save(fname.report_html(noise=config.noise, vertex=config.vertex), overwrite=True, open_browser=False)
