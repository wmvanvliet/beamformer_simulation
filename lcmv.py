import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_lcmv, apply_lcmv, apply_lcmv_cov
from scipy.stats import pearsonr

import config
from config import fname, lcmv_settings
from time_series import simulate_raw, create_epochs

# Don't be verbose
mne.set_log_level(False)

fn_stc_signal = fname.stc_signal(vertex=config.vertex)
fn_simulated_raw = fname.simulated_raw(vertex=config.vertex)
fn_simulated_epochs = fname.simulated_epochs(vertex=config.vertex)

# fn_report_h5 = fname.report(vertex=config.vertex)
fn_report_h5 = None  # Don't produce a report

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

# del info, fwd_disc_true, er_raw

epochs = create_epochs(raw)

###############################################################################
# Sensor-level analysis
###############################################################################

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make cov matrices
cov = mne.compute_covariance(epochs, tmin=0, tmax=1, method='empirical')
noise_cov = mne.compute_covariance(epochs, tmin=-1, tmax=0, method='empirical')

# Compute evokeds
evoked_grad = epochs_grad.average()
evoked_mag = epochs_mag.average()
evoked_joint = epochs_joint.average()

###############################################################################
# Compute LCMV beamformer results
###############################################################################

# Read in forward solution
fwd_disc_man = mne.read_forward_solution(fname.fwd_discrete_man)

dists = []
focs = []
corrs = []
ori_errors = []
for setting in lcmv_settings:
    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, use_noise_cov, reduce_rank, project_pca = setting
    try:
        if sensor_type == 'grad':
            evoked = evoked_grad
        elif sensor_type == 'mag':
            evoked = evoked_mag
        elif sensor_type == 'joint':
            evoked = evoked_joint
        else:
            raise ValueError('Invalid sensor type: %s', sensor_type)

        if project_pca and pick_ori != 'vector':
            raise NotImplementedError('project_pca=True only makes sense when pick_ori="vector"')

        filters = make_lcmv(evoked.info, fwd_disc_man, cov, reg=reg,
                            pick_ori=pick_ori, weight_norm=weight_norm,
                            inversion=inversion,
                            depth=1. if normalize_fwd else None,
                            noise_cov=noise_cov if use_noise_cov else None,
                            reduce_rank=reduce_rank)

        stc_est = apply_lcmv(evoked, filters).crop(0.001, 1)

        if pick_ori == 'vector':
            # Combine vector time source
            if project_pca:
                stc_proj, _ = stc_est.project('pca', fwd_disc_man['src'])
            else:
                stc_proj = stc_est.magnitude()
            stc_est_power = (stc_proj ** 2).sum()
        else:
            stc_est_power = (stc_est ** 2).sum()
        peak_vertex, peak_time = stc_est_power.get_peak(vert_as_index=True, time_as_index=True)
        estimated_time_course = np.abs(stc_est.data[peak_vertex])

        # Compute distance between true and estimated source locations
        pos_est = fwd_disc_man['source_rr'][peak_vertex]
        pos_true = fwd_disc_man['source_rr'][config.vertex]
        dist = np.linalg.norm(pos_est - pos_true)

        # Ratio between estimated peak activity and all estimated activity.
        focality_score = stc_est_power.data[peak_vertex, 0] / stc_est_power.data.sum()

        # Correlation between true and reconstructed timecourse
        true_time_course = stc_signal.copy().crop(0, 1).data[0]
        corr = pearsonr(np.abs(true_time_course), estimated_time_course)[0]

        # Angle between estimated and true source orientation
        if pick_ori == 'max-power':
            estimated_ori = filters['max_power_ori'][config.vertex]
            ori_error = np.rad2deg(np.arccos(estimated_ori @ true_ori))
            if ori_error > 90:
                ori_error = 180 - ori_error
        elif pick_ori == 'vector':
            estimated_ori = stc_est.data[peak_vertex, :, peak_time]
            estimated_ori /= np.linalg.norm(estimated_ori)
            ori_error = np.rad2deg(np.arccos(estimated_ori @ true_ori))
            if ori_error > 90:
                ori_error = 180 - ori_error
        else:
            ori_error = np.nan
    except Exception as e:
        print(e)
        dist = np.nan
        focality_score = np.nan
        corr = np.nan
        ori_error = np.nan
    print(setting, dist, focality_score, corr, ori_error)

    dists.append(dist)
    focs.append(focality_score)
    corrs.append(corr)
    ori_errors.append(ori_error)

###############################################################################
# Save everything to a pandas dataframe
###############################################################################

df = pd.DataFrame(lcmv_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'use_noise_cov',
                           'reduce_rank', 'project_pca'])
df['dist'] = dists
df['focality'] = focs
df['corr'] = corrs
df['ori_error'] = ori_errors

df.to_csv(fname.lcmv_results(vertex=config.vertex, noise=config.noise))
print('OK!')
