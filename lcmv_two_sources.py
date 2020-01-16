import warnings

import mne
import numpy as np
import pandas as pd
from mne.beamformer import make_lcmv, apply_lcmv
from mne.forward.forward import _restrict_forward_to_src_sel

import config
from config import fname, lcmv_settings
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

n_settings = len(lcmv_settings)
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
                                          source_type='random')

    ###############################################################################
    # Create epochs
    ###############################################################################

    title = 'Simulated evoked for two signal vertices'
    epochs = create_epochs(raw2, title=title, fn_simulated_epochs=None, fn_report_h5=fn_report_h5)

    epochs_grad = epochs.copy().pick_types(meg='grad')
    epochs_mag = epochs.copy().pick_types(meg='mag')
    epochs_joint = epochs.copy().pick_types(meg=True)

    # Make cov matrix
    data_cov = mne.compute_covariance(epochs, tmin=0, tmax=None, method='empirical')
    noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0, method='empirical')

    evoked_grad = epochs_grad.average()
    evoked_mag = epochs_mag.average()
    evoked_joint = epochs_joint.average()

    ###############################################################################
    # Compute LCMV beamformer results
    ###############################################################################

    # Speed things up by restricting the forward solution to only the two
    # relevant source points.
    src_sel = np.sort(np.array([config.vertex, nb_vertex]))
    fwd = _restrict_forward_to_src_sel(fwd_disc_man, src_sel)

    for idx_setting, setting in enumerate(lcmv_settings):
        if do_break[idx_setting]:
            print(setting, '(skip)')
            continue

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
                                normalize_fwd=normalize_fwd, inversion=inversion,
                                noise_cov=noise_cov if use_noise_cov else None,
                                reduce_rank=reduce_rank)
            stc = apply_lcmv(evoked, filters).crop(0.001, 1)

            vert1_idx = np.searchsorted(src_sel, config.vertex)
            vert2_idx = np.searchsorted(src_sel, nb_vertex)
            corr = correlation(stc, signal_vertex1=vert1_idx,
                               signal_vertex2=vert2_idx, signal_hemi=0)
            corrs.append(list(setting) + [nb_vertex, nb_dist, corr])

            print(setting, nb_dist, corr)

            if corr < 2 ** -0.5:
                do_break[idx_setting] = True

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
                           'weight_norm', 'normalize_fwd', 'use_noise_cov',
                           'reduce_rank', 'nb_vertex', 'nb_dist', 'corr'])
df.to_csv(fname.lcmv_results_2s(vertex=config.vertex))
print('OK!')
