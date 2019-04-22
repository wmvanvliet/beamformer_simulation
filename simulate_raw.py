import os.path as op
import mne
import numpy as np
from tqdm import tqdm

from mne.datasets import sample
from mne.simulation import simulate_sparse_stc, simulate_raw
from time_series import generate_signal, generate_random
from utils import add_stcs
from matplotlib import pyplot as plt


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
# Simulate a single signal dipole for 1 sec
###############################################################################

data = np.asarray([generate_signal(times, freq=10)])
vertices = [np.array([], dtype=np.int64), np.array([rh_vertno[0]], dtype=np.int64)]
stc_signal = mne.SourceEstimate(data=data, vertices=vertices, tmin=0,
                                tstep=1/info['sfreq'], subject='sample')


###############################################################################
# Create 109 sec of simulated data
###############################################################################

raw_list = []
# 109 seconds is max length of empty room data
n_trials = 109
for i in tqdm(range(n_trials), desc='Generating trials', total=n_trials,
              unit='trials'):
    ###########################################################################
    # Simulate random noise dipoles
    ###########################################################################
    labels = mne.read_labels_from_annot(subject='sample', parc='aparc.a2009s')
    n_noise_dipoles = len(labels)
    stc_noise = simulate_sparse_stc(
        src,
        n_noise_dipoles,
        times,
        data_fun=generate_random,
        random_state=42,
        labels=labels
    )


    ###########################################################################
    # Project to sensor space
    ###########################################################################
    stc = add_stcs(stc_signal, 0.1 * stc_noise)
    raw = simulate_raw(
        info,
        stc,
        trans=None,
        src=None,
        bem=None,
        forward=fwd,
        duration=1,
    )

    raw_list.append(raw)

raw = mne.concatenate_raws(raw_list)


###############################################################################
# Use empty room noise as sensor noise
###############################################################################
er_raw = mne.io.read_raw_fif(data_path + '/MEG/sample/ernoise_raw.fif',
                             preload=True) 
raw_picks = mne.pick_types(raw.info, meg=True, eeg=False)
er_raw_picks = mne.pick_types(er_raw.info, meg=True, eeg=False)
raw._data[raw_picks] += er_raw._data[er_raw_picks, :len(raw.times)]


###############################################################################
# Save everything
###############################################################################

save_fname = 'simulated-raw.fif'
raw.save(save_fname, overwrite=True)


###############################################################################
# Plot it!
###############################################################################
with mne.open_report('report.h5') as report:
    fig = plt.figure()
    plt.plot(times, generate_signal(times, freq=10))
    plt.xlabel('Time (s)')
    report.add_figs_to_section(fig, 'Signal time course',
                               section='Sensor-level', replace=True)

    fig = raw.plot()
    report.add_figs_to_section(fig, 'Simulated raw', section='Sensor-level',
                               replace=True)
    report.save('report.html', overwrite=True, open_browser=False)
