import mne

import config as config
from config import fname

# Read simulated raw
from time_series import create_epochs

raw = mne.io.Raw(fname.simulated_raw(noise=config.noise, vertex=config.vertex), preload=True)

fn_simulated_epochs = fname.simulated_epochs(noise=config.noise, vertex=config.vertex)
fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex)

create_epochs(raw, config.trial_length, config.n_trials,
              fn_simulated_epochs=fn_simulated_epochs,
              fn_report_h5=fn_report_h5)
