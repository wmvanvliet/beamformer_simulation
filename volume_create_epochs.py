import mne
import numpy as np

import config
from config import vfname

from utils import add_timestamp_next_to_xlabel

from datetime import datetime

# Read simulated raw
raw = mne.io.Raw(vfname.simulated_raw(noise=config.noise, vertex=config.vertex),
                 preload=True)

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

epochs.save(vfname.simulated_epochs(noise=config.noise, vertex=config.vertex),
            overwrite=True)


###############################################################################
# Save plots
###############################################################################

now = datetime.now()
with mne.open_report(vfname.report(noise=config.noise, vertex=config.vertex)) as report:

    fig = epochs.average().plot_joint(picks='mag', show=False)

    ax = fig.axes[0]

    add_timestamp_next_to_xlabel(fig, ax, now.strftime('%m/%d/%Y, %H:%M:%S'))

    report.add_figs_to_section(fig, 'Simulated evoked',
                               section='Sensor-level', replace=True)
    report.save(vfname.report_html(noise=config.noise, vertex=config.vertex),
                overwrite=True, open_browser=False)
