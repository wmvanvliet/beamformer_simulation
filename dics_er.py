import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_dics, apply_dics
from mne.time_frequency import csd_morlet

import config
from config import fname, dics_settings
from time_series import simulate_raw, create_epochs

# Don't be verbose
mne.set_log_level(False)

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
#raw = mne.io.read_raw_fif('simulation-vertex3609-raw.fif')
#stc_signal = mne.read_source_estimate('simulation-vertex3609-vl.stc')

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

# Compute evokeds
evoked_grad = epochs_grad.average()
evoked_mag = epochs_mag.average()
evoked_joint = epochs_joint.average()

###############################################################################
# Compute DICS beamformer results
###############################################################################

# Read in forward solution
fwd_disc_man = mne.read_forward_solution(fname.fwd_discrete_man)

dists = []
focs = []
ori_errors = []

for setting in dics_settings:
    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, real_filter, use_noise_cov, reduce_rank = setting

    try:
        if sensor_type == 'grad':
            info = epochs_grad.info
            evoked = evoked_grad
        elif sensor_type == 'mag':
            info = epochs_mag.info
            evoked = evoked_mag
        elif sensor_type == 'joint':
            info = epochs_joint.info
            evoked = evoked_joint
        else:
            raise ValueError('Invalid sensor type: %s', sensor_type)
        # Allow using other MNE branches without this arg
        use_kwargs = dict(noise_csd=noise_csd) if use_noise_cov else dict()
        filters = make_dics(info, fwd_disc_man, csd, reg=reg,
                            pick_ori=pick_ori,
                            inversion=inversion, weight_norm=weight_norm,
                            depth=1. if normalize_fwd else None,
                            real_filter=real_filter, reduce_rank=reduce_rank,
                            **use_kwargs)
        stc_est = apply_dics(evoked, filters).crop(0.001, 1)

        stc_est_power = (stc_est ** 2).sum()
        peak_vertex, peak_time = stc_est_power.get_peak(vert_as_index=True, time_as_index=True)
        estimated_time_course = np.abs(stc_est.data[peak_vertex])

        # Compute distance between true and estimated source locations
        pos_est = fwd_disc_man['source_rr'][peak_vertex]
        pos_true = fwd_disc_man['source_rr'][config.vertex]
        dist = np.linalg.norm(pos_est - pos_true)

        # Ratio between estimated peak activity and all estimated activity.
        focality_score = stc_est_power.data[peak_vertex, 0] / stc_est_power.data.sum()

        # Angle between estimated and true source orientation
        if pick_ori == 'max-power':
            estimated_ori = filters['max_power_ori'][config.vertex]
            ori_error = np.rad2deg(np.arccos(estimated_ori @ true_ori))
            if ori_error > 90:
                ori_error = 180 - ori_error
        else:
            ori_error = np.nan
    except Exception as e:
        print(e)
        dist = np.nan
        focality_score = np.nan
        ori_error = np.nan
    print(setting, dist, focality_score, ori_error)

    dists.append(dist)
    focs.append(focality_score)
    ori_errors.append(ori_error)

###############################################################################
# Save everything to a pandas dataframe
###############################################################################

df = pd.DataFrame(dics_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'real_filter',
                           'use_noise_cov', 'reduce_rank'])
df['dist'] = dists
df['focality'] = focs
df['ori_error'] = ori_errors

df.to_csv(fname.dics_results(vertex=config.vertex, noise=config.noise))
print('OK!')
