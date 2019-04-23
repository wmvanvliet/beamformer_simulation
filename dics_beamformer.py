import mne
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd
import numpy as np
from itertools import product

import config
from config import fname
from utils import make_dipole

# Read in the simulated data
epochs = mne.read_epochs(fname.simulated_epochs)
fwd = mne.read_forward_solution(fname.fwd)
fwd = mne.convert_forward_solution(fwd, surf_ori=True)

# Only use one sensor type (grads)
epochs.pick_types(meg='grad')

# Make CSD matrix
csd = csd_morlet(epochs, [config.signal_freq])

# Compute the settings grid
regs = [0.05, 0.1, 0.5]
pick_oris = [None, 'normal', 'max-power']
inversions = ['single', 'matrix']
weight_norms = ['unit-noise-gain', 'nai', None]
normalize_fwds = [True, False]
real_filters = [True, False]
settings = list(product(regs, pick_oris, inversions, weight_norms,
                        normalize_fwds, real_filters))

# Compute DICS beamformer with all possible parameters
dists = []
for setting in settings:
    reg, pick_ori, inversion, weight_norm, normalize_fwd, real_filter = setting
    try: 
        filters = make_dics(epochs.info, fwd, csd, reg=reg, pick_ori=pick_ori,
                            inversion=inversion, weight_norm=weight_norm,
                            normalize_fwd=normalize_fwd,
                            real_filter=real_filter)
        stc, freqs = apply_dics_csd(csd, filters)

        # Compute distance between true and estimated source
        stc_signal = mne.read_source_estimate(fname.stc_signal)
        dip_true = make_dipole(stc_signal, fwd['src'])
        dip_est = make_dipole(stc, fwd['src'])
        dist = np.linalg.norm(dip_true.pos - dip_est.pos)
    except Exception as e:
        print(e)
        dist = np.nan

    dists.append(dist)
    print(setting, dist)
