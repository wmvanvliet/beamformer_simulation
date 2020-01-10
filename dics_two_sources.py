import mne
import numpy as np
import pandas as pd
import warnings
from mne.beamformer import make_dics, apply_dics_csd
from mne.time_frequency import csd_morlet

import config
from config import fname, dics_settings
from spatial_resolution import get_nearest_neighbors, correlation
from time_series import simulate_raw, add_source_to_raw, create_epochs

# Don't be verbose
mne.set_log_level(False)

#fn_report_h5 = fname.report(vertex=config.vertex)
fn_report_h5 = None  # Don't make reports.


###############################################################################
# Simulate raw data
###############################################################################

print('simulate data')
info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
fwd_disc_true = mne.read_forward_solution(fname.fwd_discrete_true)
fwd_disc_true = mne.pick_types_forward(fwd_disc_true, meg=True, eeg=False)
er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True)

raw, stc_signal = simulate_raw(info=info, fwd_disc_true=fwd_disc_true,
                               signal_vertex=config.vertex,
                               signal_freq=config.signal_freq,
                               trial_length=config.trial_length,
                               n_trials=config.n_trials, noise_multiplier=0,
                               random_state=config.random, n_noise_dipoles=0,
                               er_raw=er_raw)

del info, er_raw


# Read in forward solution
fwd_disc_man = mne.read_forward_solution(fname.fwd_discrete_man)

###############################################################################
# Get nearest neighbors
###############################################################################

nearest_neighbors, distances = get_nearest_neighbors(config.vertex, signal_hemi=0, src=fwd_disc_true['src'])

corrs = []

n_settings = len(dics_settings)
do_break = np.zeros(shape=n_settings, dtype=bool)


for i, (nb_vertex, nb_dist) in enumerate(np.column_stack((nearest_neighbors, distances))[:config.n_neighbors_max]):
    print(f'Processing neighbour {i}/{config.n_neighbors_max}', flush=True)

    # after column_stack nb_vertex is float
    nb_vertex = int(nb_vertex)

    ###############################################################################
    # Simulate second dipole
    ###############################################################################

    raw2, stc_signal2 = add_source_to_raw(raw, fwd_disc_true=fwd_disc_true,
                                          signal_vertex=nb_vertex, signal_freq=config.signal_freq2,
                                          trial_length=config.trial_length, n_trials=config.n_trials,
                                          source_type='chirp')

    ###############################################################################
    # Create epochs
    ###############################################################################

    title = 'Simulated evoked for two signal vertices'
    epochs = create_epochs(raw2, title=title, fn_simulated_epochs=None, fn_report_h5=fn_report_h5)

    epochs_grad = epochs.copy().pick_types(meg='grad')
    epochs_mag = epochs.copy().pick_types(meg='mag')
    epochs_joint = epochs.copy().pick_types(meg=True)

    # Make CSDs
    csd = csd_morlet(epochs, [config.signal_freq, config.signal_freq2], tmin=0, tmax=1)
    noise_csd = csd_morlet(epochs, [config.signal_freq, config.signal_freq2], tmin=-1, tmax=0)

    ###############################################################################
    # Compute DICS beamformer results
    ###############################################################################

    for idx_setting, setting in enumerate(dics_settings):
        if do_break[idx_setting]:
            print(setting, '(skip)')
            continue

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

            filters = make_dics(info, fwd_disc_man, csd, reg=reg, pick_ori=pick_ori,
                                inversion=inversion, weight_norm=weight_norm,
                                noise_csd=noise_csd if use_noise_cov else None,
                                normalize_fwd=normalize_fwd,
                                real_filter=real_filter)

            stc, freqs = apply_dics_csd(csd, filters)

            ratio1 = stc.data[config.vertex, 0] / stc.data[config.vertex, 1]
            ratio2 = stc.data[nb_vertex, 1] / stc.data[nb_vertex, 0]
            ratio = ratio1 * ratio2
            corrs.append(list(setting) + [nb_vertex, nb_dist, ratio])

            print(setting, nb_dist, ratio)

            if ratio > 4:
                do_break[idx_setting] = True
                break

        except Exception as e:
            print(e)
            corrs.append(list(setting) + [nb_vertex, nb_dist, np.nan])

    if do_break.all():
        # for all settings the shared variance between neighbors is less than 1/sqrt(2)
        # no need to compute correlation for neighbors further away
        break
else:
    warnings.warn('Reached max number of sources, but still some parameter combinations have large correlations.')
    

###############################################################################
# Save everything to a pandas dataframe
###############################################################################

df = pd.DataFrame(corrs,
                  columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                           'weight_norm', 'normalize_fwd', 'real_filter', 'use_noise_cov',
                           'nb_vertex', 'nb_dist', 'corr'])
df.to_csv(fname.dics_results_2s(vertex=config.vertex))
print('OK!')
