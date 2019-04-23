import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse

import config
from config import fname

# Read in the simulated data
epochs = mne.read_epochs(fname.simulated_epochs)
evoked = epochs.average()
fwd = mne.read_forward_solution(fname.fwd)

# Make noise cov matrix
cov = mne.compute_covariance(epochs, None, 0.3)

# Compute MNE inverse
inv = make_inverse_operator(evoked.info, fwd, cov)
stc = apply_inverse(evoked, inv, 0.1, method='dSPM')

