import mne
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd
import numpy as np
from itertools import product
import pandas as pd

import config
from config import vfname
from utils import make_dipole_volume, evaluate_stc_volume

# Read in the simulated data
stc_signal = mne.read_source_estimate(vfname.stc_signal(noise=config.noise, vertex=config.vertex))
epochs = mne.read_epochs(vfname.simulated_epochs(noise=config.noise, vertex=config.vertex))
fwd_disc = mne.read_forward_solution(vfname.fwd_discrete)

# For pick_ori='normal', the fwd needs to be in surface orientation
# TODO: this should not be necessary since convert_forward_solution has used during fwd_disc creation
fwd_disc = mne.convert_forward_solution(fwd_disc, surf_ori=True)

# The DICS beamformer currently only uses one sensor type
epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')

# Make CSD matrix
csd = csd_morlet(epochs, [config.signal_freq])

# Compute the settings grid
regs = [0.05, 0.1, 0.5]
sensor_types = ['grad', 'mag']
pick_oris = [None, 'max-power']
inversions = ['single', 'matrix']
weight_norms = ['unit-noise-gain', 'nai', None]
normalize_fwds = [True, False]
real_filters = [True, False]
settings = list(product(regs, sensor_types, pick_oris, inversions,
                        weight_norms, normalize_fwds, real_filters))

# Compute DICS beamformer with all possible settings
dists = []
evals = []
for setting in settings:
    (reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd,
     real_filter) = setting
    try:
        if sensor_type == 'grad':
            info = epochs_grad.info
        elif sensor_type == 'mag':
            info = epochs_mag.info
        else:
            raise ValueError('Invalid sensor type: %s', sensor_type)

        filters = make_dics(info, fwd_disc, csd, reg=reg, pick_ori=pick_ori,
                            inversion=inversion, weight_norm=weight_norm,
                            normalize_fwd=normalize_fwd,
                            real_filter=real_filter)
        stc, freqs = apply_dics_csd(csd, filters)

        # Compute distance between true and estimated source
        dip_true = make_dipole_volume(stc_signal, fwd_disc['src'])
        dip_est = make_dipole_volume(stc, fwd_disc['src'])
        dist = np.linalg.norm(dip_true.pos - dip_est.pos)

        # Fancy evaluation metric
        ev = evaluate_stc_volume(stc, stc_signal)
    except Exception as e:
        print(e)
        dist = np.nan
        ev = np.nan
    print(setting, dist, ev)

    dists.append(dist)
    evals.append(ev)

# Save everything to a pandas dataframe
df = pd.DataFrame(settings, columns=['reg', 'sensor_type', 'pick_ori',
                                     'inversion', 'weight_norm',
                                     'normalize_fwd', 'real_filter'])
df['dist'] = dists
df['eval'] = evals
df.to_csv(vfname.dics_results(noise=config.noise, vertex=config.vertex))
