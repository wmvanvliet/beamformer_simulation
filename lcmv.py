import os.path as op
from itertools import product
import tables
from time import sleep

import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_lcmv, apply_lcmv

import config
from config import fname
from time_series import simulate_raw_vol, create_epochs
from utils import make_dipole_volume, evaluate_fancy_metric_volume

fn_stc_signal = fname.stc_signal( vertex=config.vertex)
fn_simulated_raw = fname.simulated_raw( vertex=config.vertex)
fn_simulated_epochs = fname.simulated_epochs( vertex=config.vertex)

#fn_report_h5 = fname.report(vertex=config.vertex)
fn_report_h5 = None  # Don't produce a report

###############################################################################
# Compute the settings grid
###############################################################################

regs = [0.05, 0.1, 0.5]
sensor_types = ['joint', 'grad', 'mag']
pick_oris = [None, 'max-power']
weight_norms = ['unit-noise-gain', 'nai', None]
use_noise_covs = [True, False]
depths = [True, False]

settings = list(product(regs, sensor_types, pick_oris, weight_norms,
                        use_noise_covs, depths))

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

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make cov matrix
cov = mne.compute_covariance(epochs, method='shrunk')
noise_cov = mne.compute_covariance(epochs, tmin=0.7, tmax=1.3, method='shrunk')

evoked_grad = epochs_grad.average()
evoked_mag = epochs_mag.average()
evoked_joint = epochs_joint.average()

###############################################################################
# Compute LCMV beamformer results
###############################################################################

dists = []
evals = []

for setting in settings:
    reg, sensor_type, pick_ori, weight_norm, use_noise_cov, depth = setting
    try:
        if sensor_type == 'grad':
            evoked = evoked_grad
        elif sensor_type == 'mag':
            evoked = evoked_mag
        elif sensor_type == 'joint':
            evoked = evoked_joint
        else:
            raise ValueError('Invalid sensor type: %s', sensor_type)

        filters = make_lcmv(evoked.info, fwd_disc_man, cov, reg=reg,
                            pick_ori=pick_ori, weight_norm=weight_norm,
                            noise_cov=noise_cov if use_noise_cov else None,
                            depth=depth)

        stc = apply_lcmv(evoked, filters)

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
                                     'weight_norm', 'use_noise_cov', 'depth'])
df['dist'] = dists
df['eval'] = evals

for _ in range(100):
    try:
        with pd.HDFStore(fname.lcmv_results) as store:
            store[f'vertex_{config.vertex:04d}'] = df
            store.flush()
            print('OK!')
            break
    except tables.exceptions.HDF5ExtError:
        print('Something went wrong?')
        sleep(1)
        # Try again
else:
    raise RuntimeError('Tried to write result HDF5 file 100 times and failed.')
