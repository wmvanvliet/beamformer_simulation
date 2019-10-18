import mne
import os.path as op
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd
import numpy as np
from itertools import product
import pandas as pd

import config
from config import fname
from utils import make_dipole, evaluate_stc

from time_series import simulate_raw, create_epochs

fn_stc_signal = fname.stc_signal(noise=config.noise, vertex=config.vertex)
fn_simulated_raw = fname.simulated_raw(noise=config.noise, vertex=config.vertex)
fn_simulated_epochs = fname.simulated_epochs(noise=config.noise, vertex=config.vertex)

fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex)

###############################################################################
# Simulate raw data and create epochs
###############################################################################

# TODO:
#   Should the epochs be the same for LCMV and DICS
#   Are the dipole locations always the same or should they be randomized?

if op.exists(fn_stc_signal + '-lh.stc') and op.exists(fn_simulated_epochs):
    print('load stc_signal')
    stc_signal = mne.read_source_estimate(fn_stc_signal)
    print('load epochs')
    epochs = mne.read_epochs(fn_simulated_epochs)

else:
    print('simulate data')
    info = mne.io.read_info(fname.sample_raw)
    info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
    fwd_true = mne.read_forward_solution(fname.fwd_true)
    fwd_true = mne.pick_types_forward(fwd_true, meg=True, eeg=False)
    src_true = fwd_true['src']
    er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True)
    labels = mne.read_labels_from_annot(subject='sample', parc='aparc.a2009s')

    raw, stc_signal = simulate_raw(info, src_true, fwd_true, config.vertex, config.signal_hemi,
                                   config.signal_freq, config.trial_length, config.n_trials,
                                   config.noise, config.random, labels, er_raw, fn_stc_signal=fn_stc_signal,
                                   fn_simulated_raw=fn_simulated_raw, fn_report_h5=fn_report_h5)

    del info, fwd_true, src_true, er_raw, labels

    epochs = create_epochs(raw, config.trial_length, config.n_trials,
                           fn_simulated_epochs=fn_simulated_epochs,
                           fn_report_h5=fn_report_h5)

###############################################################################
# Compute DICS beamformer results
###############################################################################

# Read in the manually created forward solution
fwd_man = mne.read_forward_solution(fname.fwd_man)
# For pick_ori='normal', the fwd needs to be in surface orientation
fwd_man = mne.convert_forward_solution(fwd_man, surf_ori=True)

# The DICS beamformer currently only uses one sensor type
epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')

# Make CSD matrix
csd = csd_morlet(epochs, [config.signal_freq])

# Compute the settings grid
regs = [0.05, 0.1, 0.5]
sensor_types = ['grad', 'mag']
pick_oris = [None, 'normal', 'max-power']
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

        filters = make_dics(info, fwd_man, csd, reg=reg, pick_ori=pick_ori,
                            inversion=inversion, weight_norm=weight_norm,
                            normalize_fwd=normalize_fwd,
                            real_filter=real_filter)
        stc, freqs = apply_dics_csd(csd, filters)

        # Compute distance between true and estimated source
        dip_true = make_dipole(stc_signal, fwd_man['src'])
        dip_est = make_dipole(stc, fwd_man['src'])
        dist = np.linalg.norm(dip_true.pos - dip_est.pos)

        # Fancy evaluation metric
        ev = evaluate_stc(stc, stc_signal)
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
df.to_csv(fname.dics_results(noise=config.noise, vertex=config.vertex))
