import os.path as op
from itertools import product

import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet

import config
from config import vfname
from time_series import simulate_raw_vol, create_epochs
from utils import make_dipole_volume, evaluate_fancy_metric_volume

fn_stc_signal = vfname.stc_signal(noise=config.noise, vertex=config.vertex)
fn_simulated_raw = vfname.simulated_raw(noise=config.noise, vertex=config.vertex)
fn_simulated_epochs = vfname.simulated_epochs(noise=config.noise, vertex=config.vertex)

fn_report_h5 = vfname.report(noise=config.noise, vertex=config.vertex)

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

if op.exists(fn_stc_signal + '-lh.stc') and op.exists(fn_simulated_epochs):
    print('load stc_signal')
    stc_signal = mne.read_source_estimate(fn_stc_signal)
    print('load epochs')
    epochs = mne.read_epochs(fn_simulated_epochs)

else:
    print('simulate data')
    info = mne.io.read_info(vfname.sample_raw)
    info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
    fwd_disc_true = mne.read_forward_solution(vfname.fwd_discrete_true)
    fwd_disc_true = mne.pick_types_forward(fwd_disc_true, meg=True, eeg=False)
    er_raw = mne.io.read_raw_fif(vfname.ernoise, preload=True)

    raw, stc_signal = simulate_raw_vol(info=info, fwd_disc_true=fwd_disc_true, signal_vertex=config.vertex,
                                       signal_freq=config.signal_freq, trial_length=config.trial_length,
                                       n_trials=config.n_trials, noise_multiplier=config.noise,
                                       random_state=config.random, n_noise_dipoles=config.n_noise_dipoles_vol,
                                       er_raw=er_raw, fn_stc_signal=fn_stc_signal, fn_simulated_raw=fn_simulated_raw,
                                       fn_report_h5=fn_report_h5)

    del info, fwd_disc_true, er_raw

    epochs = create_epochs(raw, config.trial_length, config.n_trials,
                           fn_simulated_epochs=fn_simulated_epochs,
                           fn_report_h5=fn_report_h5)

###############################################################################
# Read in the manually created forward solution
###############################################################################

fwd_disc_man = mne.read_forward_solution(vfname.fwd_discrete_man)

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
# TODO: do we calculate the csd matrix for epochs_grad and epochs_mag separately?
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
df.to_csv(vfname.dics_results(noise=config.noise, vertex=config.vertex))
