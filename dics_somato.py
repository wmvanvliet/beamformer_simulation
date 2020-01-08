import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet

import config
from config import fname, dics_settings, somato_true_pos
from utils import make_dipole_volume, evaluate_fancy_metric_volume

# Don't be verbose
mne.set_log_level(False)

###############################################################################
# Sensor level analysis
###############################################################################

epochs = mne.read_epochs(fname.somato_epochs_long)
freqs = np.logspace(np.log10(12), np.log10(30), 9)

# Compute Cross-Spectral Density matrices
csd = csd_morlet(epochs, freqs, tmin=-1, tmax=1.5, decim=5)
noise_csd = csd_morlet(epochs, freqs, tmin=-1, tmax=0, decim=5)
# ERS activity starts at 0.5 seconds after stimulus onset
csd_ers = csd_morlet(epochs, freqs, tmin=0.5, tmax=1.5, decim=5)

csd = csd.mean()
noise_csd = noise_csd.mean()
csd_ers = csd_ers.mean()

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

###############################################################################
# Compute DICS beamformer results
###############################################################################

fwd = mne.read_forward_solution(fname.somato_fwd)

dists = []
evals = []

for setting in dics_settings:
    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, real_filter, use_noise_cov = setting
    try:
        if sensor_type == 'grad':
            info = epochs_grad.info
        elif sensor_type == 'mag':
            info = epochs_mag.info
        elif sensor_type == 'joint':
            info = epochs_joint.info
        else:
            raise ValueError('Invalid sensor type: %s', sensor_type)

        filters = make_dics(info, fwd, csd, reg=reg, pick_ori=pick_ori,
                            inversion=inversion, weight_norm=weight_norm,
                            noise_csd=noise_csd if use_noise_cov else None,
                            normalize_fwd=normalize_fwd,
                            real_filter=real_filter)

        # Compute source power
        stc_baseline, _ = apply_dics_csd(noise_csd, filters)
        stc_ers, _ = apply_dics_csd(csd_ers, filters)

        # Normalize with baseline power.
        stc_ers /= stc_baseline
        stc_ers.data = np.log(stc_ers.data)

        # Compute distance between true and estimated source
        dip_est = make_dipole_volume(stc_ers, fwd['src'])
        dist = np.linalg.norm(somato_true_pos - dip_est.pos)

        # Fancy evaluation metric
        #ev = evaluate_fancy_metric_volume(stc, stc_signal)
        ev = np.nan
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

df = pd.DataFrame(dics_settings,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'real_filter',
                           'use_noise_cov'])
df['dist'] = dists
df['eval'] = evals

df.to_csv(fname.dics_somato_results)
print('OK!')
