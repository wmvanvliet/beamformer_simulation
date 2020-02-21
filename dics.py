import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet

import config
from config import fname, dics_settings
from time_series import simulate_raw, create_epochs
from utils import make_dipole_volume, evaluate_fancy_metric_volume

# Don't be verbose
mne.set_log_level(False)

fn_stc_signal = fname.stc_signal(vertex=config.vertex)
fn_simulated_raw = fname.simulated_raw(vertex=config.vertex)
fn_simulated_epochs = fname.simulated_epochs(vertex=config.vertex)

#fn_report_h5 = fname.report(vertex=config.vertex)
fn_report_h5 = None  # Don't produce a report

src = mne.read_source_spaces(fname.src)

###############################################################################
# Simulate raw data and create epochs
###############################################################################

print('simulate data')
info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
fwd_disc_true = mne.read_forward_solution(fname.fwd_discrete_true)
fwd_disc_true = mne.pick_types_forward(fwd_disc_true, meg=True, eeg=False)
er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True)

raw, stc_signal = simulate_raw(info=info, fwd_disc_true=fwd_disc_true, signal_vertex=config.vertex,
                               signal_freq=config.signal_freq, n_trials=config.n_trials,
                               noise_multiplier=config.noise, random_state=config.random,
                               n_noise_dipoles=config.n_noise_dipoles_vol, er_raw=er_raw)

true_ori = fwd_disc_true['src'][0]['nn'][config.vertex]

del info, fwd_disc_true, er_raw

epochs = create_epochs(raw)

###############################################################################
# Sensor level analysis
###############################################################################

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make CSD matrix
csd = csd_morlet(epochs, [config.signal_freq], tmin=0, tmax=1)
noise_csd = csd_morlet(epochs, [config.signal_freq], tmin=-1, tmax=0)

###############################################################################
# Compute DICS beamformer results
###############################################################################

# Read in forward solution
fwd_disc_man = mne.read_forward_solution(fname.fwd_discrete_man)

dists = []
evals = []
ori_errors = []

for setting in dics_settings:
    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, real_filter, use_noise_cov, reduce_rank = setting
    try:
        if sensor_type == 'grad':
            info = epochs_grad.info
        elif sensor_type == 'mag':
            info = epochs_mag.info
        elif sensor_type == 'joint':
            info = epochs_joint.info
        else:
            raise ValueError('Invalid sensor type: %s', sensor_type)

        filters = make_dics(info, fwd_disc_man, csd, reg=reg, pick_ori=pick_ori,
                            inversion=inversion, weight_norm=weight_norm,
                            noise_csd=noise_csd if use_noise_cov else None,
                            normalize_fwd=normalize_fwd,
                            real_filter=real_filter, reduce_rank=reduce_rank)
        stc, freqs = apply_dics_csd(csd, filters)

        # Compute distance between true and estimated source
        dip_true = make_dipole_volume(stc_signal, fwd_disc_man['src'])
        dip_est = make_dipole_volume(stc, fwd_disc_man['src'])
        dist = np.linalg.norm(dip_true.pos - dip_est.pos)

        # Fancy evaluation metric
        ev = evaluate_fancy_metric_volume(stc, stc_signal)

        if pick_ori == 'max-power':
            estimated_ori = filters['max_power_oris'][0][config.vertex]
            ori_error = np.rad2deg(abs(np.arccos(estimated_ori @ true_ori)))
        else:
            ori_error = np.nan
    except Exception as e:
        print(e)
        dist = np.nan
        ev = np.nan
        ori_error = np.nan
    print(setting, dist, ev, ori_error)

    dists.append(dist)
    evals.append(ev)
    ori_errors.append(ori_error)

###############################################################################
# Save everything to a pandas dataframe
###############################################################################

df = pd.DataFrame(dics_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'real_filter',
                           'use_noise_cov', 'reduce_rank'])
df['dist'] = dists
df['eval'] = evals
df['ori_error'] = ori_errors

df.to_csv(fname.dics_results(vertex=config.vertex, noise=config.noise))
print('OK!')
