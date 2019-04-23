import mne
from mne.time_frequency import csd_morlet
from mne.beamformer import make_lcmv, apply_lcmv

import config
from config import fname

# Read in the simulated data
epochs = mne.read_epochs(fname.simulated_epochs)
evoked = epochs.average()
fwd = mne.read_forward_solution(fname.fwd)

# Only use one sensor type (grads)
epochs.pick_types(meg='mag')

# Make cov matrix
cov = mne.compute_covariance(epochs)

# Compute LCMV beamformer
filters = make_lcmv(epochs.info, fwd, cov, reg=0.1)
stc = apply_lcmv(evoked, filters)

