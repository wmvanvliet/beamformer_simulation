import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_lcmv, apply_lcmv

from config import lcmv_settings, fname
from megset.config import fname as megset_fname

subject = 7
mne.set_log_level(False)  # Shhh

###############################################################################
# Load the data
###############################################################################

epochs = mne.read_epochs(megset_fname.epochs(subject=subject))
fwd = mne.read_forward_solution(megset_fname.fwd(subject=subject))
dip = mne.read_dipole(megset_fname.ecd(subject=subject))

###############################################################################
# Sensor-level analysis for beamformer
###############################################################################

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make cov matrices
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='empirical', rank='info')
data_cov = mne.compute_covariance(epochs, tmin=0, tmax=0.4, method='empirical', rank='info')

# Compute evokeds
tmin = 0.03
tmax = 0.05
evoked_grad = epochs_grad.average().crop(tmin, tmax)
evoked_mag = epochs_mag.average().crop(tmin, tmax)
evoked_joint = epochs_joint.average().crop(tmin, tmax)

# Find the time point that corresponds to the best dipole fit
peak_time = dip[int(np.argmax(dip.gof))].times[0]

# Get true_vert_idx
rr = fwd['src'][0]['rr']
inuse = fwd['src'][0]['inuse']
indices = np.where(fwd['src'][0]['inuse'])[0]
rr_inuse = rr[indices]
true_vert_idx = np.where(np.linalg.norm(rr_inuse - dip.pos, axis=1) ==
                         np.linalg.norm(rr_inuse - dip.pos, axis=1).min())[0][0]

###############################################################################
# Compute LCMV solution and plot stc at dipole location
###############################################################################

dists = []
focs = []
ori_errors = []

for ii, setting in enumerate(lcmv_settings):
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
        stc = abs(stc.copy().crop(peak_time, peak_time))
        stc_power = (stc ** 2).sqrt()
        peak_vertex, _ = stc.get_peak(vert_as_index=True)

        # Compute distance between true and estimated source locations
        pos = fwd['source_rr'][peak_vertex]
        dist = np.linalg.norm(dip.pos - pos)

        # Ratio between estimated peak activity and all estimated activity.
        focality_score = stc_power.data[peak_vertex, 0] / stc_power.data.sum()

        if pick_ori == 'max-power':
            estimated_ori = filters['max_power_ori'][peak_vertex]
            ori_error = np.rad2deg(np.arccos(estimated_ori @ dip.ori[0]))
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

df = pd.DataFrame(lcmv_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'use_noise_cov', 'reduce_rank'])

df['dist'] = dists
df['focs'] = focs
df['ori_error'] = ori_errors

df.to_csv(fname.lcmv_megset_results(subject=subject))
print('OK!')
