import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_lcmv, apply_lcmv

import config
from config import fname, lcmv_settings, somato_true_pos, somato_true_vert_idx
from utils import evaluate_fancy_metric_volume

# Don't be verbose
mne.set_log_level(False)

fn_stc_signal = fname.stc_signal( vertex=config.vertex)
fn_simulated_raw = fname.simulated_raw( vertex=config.vertex)
fn_simulated_epochs = fname.simulated_epochs( vertex=config.vertex)

#fn_report_h5 = fname.report(vertex=config.vertex)
fn_report_h5 = None  # Don't produce a report

###############################################################################
# Sensor-level analysis
###############################################################################

epochs = mne.read_epochs(fname.somato_epochs)

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make cov matrices
data_cov = mne.compute_covariance(epochs, tmin=0, tmax=0.4, method='empirical')
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='empirical')

# Compute evokeds
evoked_grad = epochs_grad.average()
evoked_mag = epochs_mag.average()
evoked_joint = epochs_joint.average()

###############################################################################
# Compute LCMV beamformer results
###############################################################################

# Read in forward solution
fwd = mne.read_forward_solution(fname.somato_fwd)

dists = []
evals = []
for setting in lcmv_settings:
    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, use_noise_cov, reduce_rank = setting
    try:
        if sensor_type == 'grad':
            evoked = evoked_grad
        elif sensor_type == 'mag':
            evoked = evoked_mag
        elif sensor_type == 'joint':
            evoked = evoked_joint
        else:
            raise ValueError('Invalid sensor type: %s', sensor_type)

        filters = make_lcmv(evoked.info, fwd, data_cov, reg=reg,
                            pick_ori=pick_ori, weight_norm=weight_norm,
                            inversion=inversion, normalize_fwd=normalize_fwd,
                            noise_cov=noise_cov if use_noise_cov else None,
                            reduce_rank=reduce_rank)

        stc = apply_lcmv(evoked, filters)

        # # Peak should be around 0.04
        # peak_time = stc.get_peak()[1]
        # if not (0.01 <= peak_time <= 0.05):
        #     print('Could not find SI peak')
        #     dist = np.nan
        #     ev = np.nan
        # else:
        #     #stc = stc.crop(0.03, 0.05).mean()

        stc = stc.crop(0.035, 0.035)

        # Compute distance between true and estimated source
        estimated_pos = fwd['src'][0]['rr'][stc.get_peak()[0]]
        dist = np.linalg.norm(somato_true_pos - estimated_pos)

        # Fancy evaluation metric
        ev = evaluate_fancy_metric_volume(stc, true_vert_idx=somato_true_vert_idx)
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

df = pd.DataFrame(lcmv_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'use_noise_cov', 'reduce_rank'])
df['dist'] = dists
df['eval'] = evals

df.to_csv(fname.lcmv_somato_results)
print('OK!')
