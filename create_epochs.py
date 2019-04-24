import os.path as op
import mne
import numpy as np

import config
from config import fname

# Read simulated raw
raw = mne.io.Raw(fname.simulated_raw(noise=config.noise, vertex=config.vertex), preload=True)

###############################################################################
# Create epochs
###############################################################################

events = np.hstack((
    (np.arange(config.n_trials) * config.trial_length * raw.info['sfreq'])[:, np.newaxis],
    np.zeros((config.n_trials, 1)),
    np.ones((config.n_trials, 1)),
)).astype(np.int)

epochs = mne.Epochs(raw=raw, events=events, event_id=1,
                    tmin=0.1, tmax=config.trial_length - 0.1,
                    baseline=(None, 0.3), preload=True)

epochs.save(fname.simulated_epochs(noise=config.noise, vertex=config.vertex), overwrite=True)


###############################################################################
# Save plots
###############################################################################

with mne.open_report(fname.report(noise=config.noise, vertex=config.vertex)) as report:
    fig = epochs.average().plot_joint(picks='mag', show=False)
    report.add_figs_to_section(fig, 'Simulated evoked',
                               section='Sensor-level', replace=True)
    report.save(fname.report_html(noise=config.noise, vertex=config.vertex), overwrite=True, open_browser=False)
