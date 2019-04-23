import mne
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd

import config
from config import fname

# Read in the simulated data
epochs = mne.read_epochs(fname.simulated_epochs)
fwd = mne.read_forward_solution(fname.fwd)

# Only use one sensor type (grads)
epochs.pick_types(meg='grad')

# Make CSD matrix
csd = csd_morlet(epochs, [10])

# Compute DICS beamformer
filters = make_dics(epochs.info, fwd, csd, reg=1)
stc, freqs = apply_dics_csd(csd, filters)
