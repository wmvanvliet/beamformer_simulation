import mne
import os.path as op
from mne.beamformer import make_lcmv, apply_lcmv
import numpy as np
from itertools import product
import pandas as pd

import config
from config import fname
from utils import make_dipole, evaluate_fancy_metric

from time_series import simulate_raw, create_epochs

fn_stc_signal = fname.stc_signal(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)
fn_simulated_raw = fname.simulated_raw(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)
fn_simulated_epochs = fname.simulated_epochs(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)

fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)

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
# Compute LCMV beamformer results
###############################################################################

# Read in the manually created forward solution
fwd_man = mne.read_forward_solution(fname.fwd_man)
# For pick_ori='normal', the fwd needs to be in surface orientation
fwd_man = mne.convert_forward_solution(fwd_man, surf_ori=True)

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make cov matrix
cov = mne.compute_covariance(epochs, method='empirical')
noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.3, method='empirical')

evoked_grad = epochs_grad.average()
evoked_mag = epochs_mag.average()
evoked_joint = epochs_joint.average()

# Compute the settings grid
# regs = [0.05, 0.1, 0.5]
# sensor_types = ['joint', 'grad', 'mag']
# pick_oris = [None, 'normal', 'max-power']
# weight_norms = ['unit-noise-gain', 'nai', None]
# use_noise_covs = [True, False]
# depths = [True, False]
regs = [0.05]
sensor_types = ['joint']
pick_oris = ['max-power']
weight_norms = [None]
use_noise_covs = [True]
depths = [True, False]
settings = list(product(regs, sensor_types, pick_oris, weight_norms,
                        use_noise_covs, depths))

# Compute LCMV beamformer with all possible settings
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

        filters = make_lcmv(evoked.info, fwd_man, cov, reg=reg,
                            pick_ori=pick_ori, weight_norm=weight_norm,
                            noise_cov=noise_cov if use_noise_cov else None,
                            depth=depth)

        stc = apply_lcmv(evoked, filters)

        # Compute distance between true and estimated source
        dip_true = make_dipole(stc_signal, fwd_man['src'])
        dip_est = make_dipole(stc, fwd_man['src'])
        dist = np.linalg.norm(dip_true.pos - dip_est.pos)

        # Fancy evaluation metric
        ev = evaluate_fancy_metric(stc, stc_signal)
    except Exception as e:
        print(e)
        dist = np.nan
        ev = np.nan
    print(setting, dist, ev)

    dists.append(dist)
    evals.append(ev)

# Save everything to a pandas dataframe
df = pd.DataFrame(settings, columns=['reg', 'sensor_type', 'pick_ori',
                                     'weight_norm', 'use_noise_cov', 'depth'])
df['dist'] = dists
df['eval'] = evals
df.to_csv(fname.lcmv_results(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi))
