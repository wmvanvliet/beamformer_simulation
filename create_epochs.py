import os.path as op
import mne

raw = mne.io.Raw('simulated-raw.fif', preload=True)

evt_length = 1
n_events = 109
evt_id = 1
baseline = (None, 0.3)

events = [[int(evt_length * i * raw.info['sfreq']), 0, evt_id] for i in range(n_events)]

epochs = mne.Epochs(raw=raw, events=events, event_id=evt_id, tmin=0.1, tmax=0.9, baseline=baseline, preload=True)

epo_fname = 'simulated-epo.fif'

epochs.save(epo_fname)

evoked = epochs.average()

evoked.plot_joint()


