import os.path as op
from itertools import product
import tables
from time import sleep

import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet

import config
from config import fname
from time_series import simulate_raw_vol, create_epochs
from utils import make_dipole_volume, evaluate_fancy_metric_volume

fn_stc_signal = fname.stc_signal(vertex=config.vertex)
fn_simulated_raw = fname.simulated_raw(vertex=config.vertex)
fn_simulated_epochs = fname.simulated_epochs(vertex=config.vertex)

#fn_report_h5 = fname.report(vertex=config.vertex)
fn_report_h5 = None  # Don't produce a report

###############################################################################
# Compute the settings grid
###############################################################################

regs = [0.05, 0.1, 0.5]
sensor_types = ['grad', 'mag']
pick_oris = [None, 'max-power']
inversions = ['single', 'matrix']
weight_norms = ['unit-noise-gain', 'nai', None]
normalize_fwds = [True, False]
real_filters = [True, False]

settings = list(product(regs, sensor_types, pick_oris, inversions,
                        weight_norms, normalize_fwds, real_filters))

###############################################################################
# Simulate raw data and create epochs
###############################################################################

print('simulate data')
info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
fwd_disc_true = mne.read_forward_solution(fname.fwd_discrete_true)
fwd_disc_true = mne.pick_types_forward(fwd_disc_true, meg=True, eeg=False)
er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True)

raw, stc_signal = simulate_raw_vol(info=info, fwd_disc_true=fwd_disc_true, signal_vertex=config.vertex,
                                   signal_freq=config.signal_freq, trial_length=config.trial_length,
                                   n_trials=config.n_trials, noise_multiplier=config.noise,
                                   random_state=config.random, n_noise_dipoles=config.n_noise_dipoles_vol,
                                   er_raw=er_raw)

del info, fwd_disc_true, er_raw

epochs = create_epochs(raw, config.trial_length, config.n_trials)

###############################################################################
# Read in the manually created forward solution
###############################################################################

fwd_disc_man = mne.read_forward_solution(fname.fwd_discrete_man)

# TODO: test if this is actually necessary for a discrete volume source space
# For pick_ori='normal', the fwd needs to be in surface orientation
fwd_disc_man = mne.convert_forward_solution(fwd_disc_man, surf_ori=True)

###############################################################################
# Create epochs for for different sensors
###############################################################################

# The DICS beamformer currently only uses one sensor type
epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')

# Make CSD matrix
csd = csd_morlet(epochs, [config.signal_freq])

###############################################################################
# Compute DICS beamformer results
###############################################################################

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

        filters = make_dics(info, fwd_disc_man, csd, reg=reg, pick_ori=pick_ori,
                            inversion=inversion, weight_norm=weight_norm,
                            normalize_fwd=normalize_fwd,
                            real_filter=real_filter)

        stc, freqs = apply_dics_csd(csd, filters)

        # Compute distance between true and estimated source
        dip_true = make_dipole_volume(stc_signal, fwd_disc_man['src'])
        dip_est = make_dipole_volume(stc, fwd_disc_man['src'])
        dist = np.linalg.norm(dip_true.pos - dip_est.pos)

        # Fancy evaluation metric
        ev = evaluate_fancy_metric_volume(stc, stc_signal)
    except Exception as e:
        print(e)
        dist = np.nan
        ev = np.nan
    print(setting, dist, ev)

    dists.append(dist)
    evals.append(ev)

###############################################################################
# Save everything to a pandas dataframe
###############################################################################

df = pd.DataFrame(settings, columns=['reg', 'sensor_type', 'pick_ori',
                                     'inversion', 'weight_norm',
                                     'normalize_fwd', 'real_filter'])
df['dist'] = dists
df['eval'] = evals

for _ in range(100):
    try:
        df.to_hdf(fname.dics_results, f'vertex_{config.vertex:04d}')
        print('OK!')
        break
    except tables.exceptions.HDF5ExtError as e:
        print(f'Something went wrong? {e}')
        sleep(1)
        # Try again
else:
    raise RuntimeError('Tried to write result HDF5 file 100 times and failed.')
