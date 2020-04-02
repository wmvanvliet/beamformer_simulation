import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_dics, apply_dics_csd

from config import dics_settings, fname, args
from megset.config import fname as megset_fname
from megset.config import freq_range

subject = args.subject
print(f'Running analsis for subject {subject}')

mne.set_log_level(False)  # Shhh

###############################################################################
# Load the data
###############################################################################

epochs = mne.read_epochs(megset_fname.epochs_long(subject=subject))
fwd = mne.read_forward_solution(megset_fname.fwd(subject=subject))
dip = mne.read_dipole(megset_fname.ecd(subject=subject))

###############################################################################
# Sensor-level analysis for beamformer
###############################################################################

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make csd matrices
freqs = np.arange(*freq_range[subject])
csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-0.8, tmax=1.0, decim=5)
csd_baseline = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-0.8, tmax=0, decim=5)
# ERS activity starts at 0.5 seconds after stimulus onset
csd_ers = mne.time_frequency.csd_morlet(epochs, freqs, tmin=0.2, tmax=1.0, decim=5)

csd = csd.mean()
csd_baseline = csd_baseline.mean()
csd_ers = csd_ers.mean()

###############################################################################
# Compute dics solution and plot stc at dipole location
###############################################################################

dists = []
focs = []
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

        info_eq, fwd_eq, csd_eq = mne.channels.equalize_channels([info, fwd, csd])
        filters = make_dics(info_eq, fwd_eq, csd_eq, reg=reg, pick_ori=pick_ori,
                            inversion=inversion, weight_norm=weight_norm,
                            noise_csd=csd_baseline if use_noise_cov else None,
                            normalize_fwd=normalize_fwd,
                            real_filter=real_filter, reduce_rank=reduce_rank)

        # Compute source power
        stc_baseline, _ = apply_dics_csd(csd_baseline, filters)
        stc_power, _ = apply_dics_csd(csd_ers, filters)

        # Normalize with baseline power.
        stc_power /= stc_baseline
        stc_power.data = np.log(stc_power.data)

        peak_vertex, _ = stc_power.get_peak(vert_as_index=True)

        # Compute distance between true and estimated source locations
        pos = fwd['source_rr'][peak_vertex]
        dist = np.linalg.norm(dip.pos - pos)

        # Ratio between estimated peak activity and all estimated activity.
        focality_score = stc_power.data[peak_vertex, 0] / stc_power.data.sum()

        if pick_ori == 'max-power':
            estimated_ori = filters['max_power_oris'][0][peak_vertex]
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

df = pd.DataFrame(dics_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'real_filter',
                           'use_noise_cov', 'reduce_rank'])
df['dist'] = dists
df['focality'] = focs
df['ori_error'] = ori_errors

df.to_csv(fname.dics_megset_results(subject=subject))
print('OK!')
