from itertools import product

import mne
import numpy as np
import pandas as pd
from time import sleep
import warnings

import config
from config import fname, vertex
from spatial_resolution import get_nearest_neighbors, compute_lcmv_beamformer_results_two_sources
from time_series import simulate_raw_vol_two_sources, create_epochs

# Don't be verbose
mne.set_log_level(False)

#fn_report_h5 = fname.report(vertex=config.vertex)
fn_report_h5 = None  # Don't make reports.

###############################################################################
# Compute the settings grid
###############################################################################

# Compare
#   - vector vs. scalar (max-power orientation)
#   - Array-gain BF (leadfield normalization)
#   - Unit-gain BF ('vanilla' LCMV)
#   - Unit-noise-gain BF (weight normalization)
#   - pre-whitening (noise-covariance)
#   - different sensor types
#   - what changes with condition contrasting

regs = [0.05, 0.1, 0.5]
sensor_types = ['joint', 'grad', 'mag']
pick_oris = [None, 'max-power']
weight_norms = ['unit-noise-gain', 'nai', None]
use_noise_covs = [True, False]
depths = [True, False]
settings = list(product(regs, sensor_types, pick_oris, weight_norms,
                        use_noise_covs, depths))

###############################################################################
# Load data
###############################################################################

print('simulate data')
info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
fwd_disc_true = mne.read_forward_solution(fname.fwd_discrete_true)
fwd_disc_true = mne.pick_types_forward(fwd_disc_true, meg=True, eeg=False)
er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True)

# Read in the manually created discrete forward solution
fwd_disc_man = mne.read_forward_solution(fname.fwd_discrete_man)
# TODO: test if this is actually necessary for a discrete volume source space
# For pick_ori='normal', the fwd needs to be in surface orientation
fwd_disc_man = mne.convert_forward_solution(fwd_disc_man, surf_ori=True)

###############################################################################
# Get nearest neighbors
###############################################################################

nearest_neighbors, distances = get_nearest_neighbors(config.vertex, signal_hemi=0, src=fwd_disc_true['src'])

corrs = []

n_settings = len(settings)
do_break = np.zeros(shape=n_settings, dtype=bool)

for nb_vertex, nb_dist in np.column_stack((nearest_neighbors, distances))[:config.n_neighbors_max]:

    # after column_stack nb_vertex is float
    nb_vertex = int(nb_vertex)

    ###############################################################################
    # Simulate raw data
    ###############################################################################

    raw, _, _ = simulate_raw_vol_two_sources(info=info, fwd_disc_true=fwd_disc_true, signal_vertex1=config.vertex,
                                             signal_freq1=config.signal_freq, signal_vertex2=nb_vertex,
                                             signal_freq2=config.signal_freq2, trial_length=config.trial_length,
                                             n_trials=config.n_trials, noise_multiplier=config.noise,
                                             random_state=config.random, n_noise_dipoles=config.n_noise_dipoles_vol,
                                             er_raw=er_raw)

    ###############################################################################
    # Create epochs
    ###############################################################################

    title = 'Simulated evoked for two signal vertices'
    epochs = create_epochs(raw, config.trial_length, config.n_trials, title=title,
                           fn_simulated_epochs=None, fn_report_h5=fn_report_h5)

    epochs_grad = epochs.copy().pick_types(meg='grad')
    epochs_mag = epochs.copy().pick_types(meg='mag')
    epochs_joint = epochs.copy().pick_types(meg=True)

    # Make cov matrix
    cov = mne.compute_covariance(epochs, method='empirical')
    noise_cov = mne.compute_covariance(epochs, tmin=0.7, tmax=1.3, method='empirical')

    evoked_grad = epochs_grad.average()
    evoked_mag = epochs_mag.average()
    evoked_joint = epochs_joint.average()

    ###############################################################################
    # Compute LCMV beamformer results
    ###############################################################################

    for idx_setting, setting in enumerate(settings):
        reg, sensor_type, pick_ori, weight_norm, use_noise_cov, depth = setting
        try:
            if sensor_type == 'grad':
                evoked = evoked_grad
            elif sensor_type == 'mag':
                evoked = evoked_mag
            elif sensor_type == 'joint':
                evoked = evoked_joint
            else:
                raise ValueError('Invalid sensor type: %s', sensor_type)

            corr = compute_lcmv_beamformer_results_two_sources(setting, evoked, cov, noise_cov, fwd_disc_man,
                                                               signal_vertex1=config.vertex, signal_vertex2=nb_vertex,
                                                               signal_hemi=0)

            corrs.append(list(setting) + [nb_vertex, nb_dist, corr])

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

df = pd.DataFrame(corrs, columns=['reg', 'sensor_type', 'pick_ori', 'weight_norm', 'use_noise_cov', 'depth',
                                  'nb_vertex', 'nb_dist', 'corr'])
df.to_csv(fname.lcmv_results_2s(vertex=config.vertex))
print('OK!')
