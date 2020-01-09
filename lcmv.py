import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_lcmv, apply_lcmv
from scipy.stats import pearsonr

import config
from config import fname, lcmv_settings
from time_series import simulate_raw, create_epochs
from utils import make_dipole_volume, evaluate_fancy_metric_volume

# Don't be verbose
mne.set_log_level(False)

fn_stc_signal = fname.stc_signal( vertex=config.vertex)
fn_simulated_raw = fname.simulated_raw( vertex=config.vertex)
fn_simulated_epochs = fname.simulated_epochs( vertex=config.vertex)

#fn_report_h5 = fname.report(vertex=config.vertex)
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
                               signal_freq=config.signal_freq, trial_length=config.trial_length,
                               n_trials=config.n_trials, noise_multiplier=config.noise,
                               random_state=config.random, n_noise_dipoles=config.n_noise_dipoles_vol,
                               er_raw=er_raw)

del info, fwd_disc_true, er_raw

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
evals = []
corrs = []
for setting in lcmv_settings:
    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, use_noise_cov = setting
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
                            inversion=inversion, normalize_fwd=normalize_fwd,
                            noise_cov=noise_cov if use_noise_cov else None)

        stc = apply_lcmv(evoked, filters).crop(0.001, 1)

        # Compute distance between true and estimated source
        dip_true = make_dipole_volume(stc_signal, fwd_disc_man['src'])
        dip_est = make_dipole_volume(stc, fwd_disc_man['src'])
        dist = np.linalg.norm(dip_true.pos - dip_est.pos)

        # Fancy evaluation metric
        ev = evaluate_fancy_metric_volume(stc, stc_signal)

        # Correlation between true and reconstructed timecourse
        true_time_course = stc_signal.copy().crop(0, 1).data[0]
        peak_vertex = abs(stc).mean().get_peak(vert_as_index=True)[0]
        estimated_time_course = np.abs(stc.data[peak_vertex])
        corr = pearsonr(np.abs(true_time_course), estimated_time_course)[0]
    except Exception as e:
        print(e)
        dist = np.nan
        ev = np.nan
        corr = np.nan
    print(setting, dist, ev, corr)

    dists.append(dist)
    evals.append(ev)
    corrs.append(corr)

###############################################################################
# Save everything to a pandas dataframe
###############################################################################

df = pd.DataFrame(lcmv_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'use_noise_cov'])
df['dist'] = dists
df['eval'] = evals
df['corr'] = corrs

df.to_csv(fname.lcmv_results(vertex=config.vertex))
print('OK!')
