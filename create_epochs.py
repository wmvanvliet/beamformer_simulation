import os.path as op
import mne

import config
from config import fname

raw = mne.io.Raw(fname.simulated_raw, preload=True)

evt_length = 1
n_events = config.n_trials
evt_id = 1
baseline = (None, 0.3)

events = [[int(evt_length * i * raw.info['sfreq']), 0, evt_id] for i in range(n_events)]

epochs = mne.Epochs(raw=raw, events=events, event_id=evt_id, tmin=0.1, tmax=0.9, baseline=baseline, preload=True)

epochs.save(fname.simulated_epochs, overwrite=True)

evoked = epochs.average()

with mne.open_report(fname.report) as report:
    fig = evoked.plot_joint(picks='mag', show=False)
    report.add_figs_to_section(fig, 'Simulated evoked',
                               section='Sensor-level', replace=True)
    report.save(fname.report_html, overwrite=True, open_browser=False)
