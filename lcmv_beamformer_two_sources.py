import mne
import numpy as np
from itertools import product

import config
from config import fname
from utils import add_stcs

from time_series import simulate_raw_two_sources, create_epochs
from spatial_resolution import compute_lcmv_beamformer_results_two_sources, get_nearest_neighbors

# TODO: maybe create a second separate report or make sure that nothing gets overwritten
fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)
fn_report_h5 = None

###############################################################################
# Compute the settings grid
###############################################################################

regs = [0.05, 0.1, 0.5]
sensor_types = ['joint', 'grad', 'mag']
pick_oris = [None, 'normal', 'max-power']
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
fwd_true = mne.read_forward_solution(fname.fwd_true)
fwd_true = mne.pick_types_forward(fwd_true, meg=True, eeg=False)
src_true = fwd_true['src']
er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True)
labels = mne.read_labels_from_annot(subject='sample', parc='aparc.a2009s')

# Read in the manually created forward solution
fwd_man = mne.read_forward_solution(fname.fwd_man)
# For pick_ori='normal', the fwd needs to be in surface orientation
fwd_man = mne.convert_forward_solution(fwd_man, surf_ori=True)

###############################################################################
# Get nearest neighbors
###############################################################################

nearest_neighbors, distances = get_nearest_neighbors(config.vertex, config.signal_hemi, src_true)

corr_list = []

for nb_vertex, nb_dist in np.column_stack((nearest_neighbors, distances))[:config.n_neighbors_max]:

    # after column_stack nb_vertex is float
    nb_vertex = int(nb_vertex)

    ###############################################################################
    # Simulate raw data
    ###############################################################################

    raw, stc_signal1, stc_signal2 = simulate_raw_two_sources(info, src=src_true, fwd=fwd_true,
                                                             signal_vertex1=config.vertex,
                                                             signal_hemi1=config.signal_hemi,
                                                             signal_freq1=config.signal_freq,
                                                             signal_vertex2=nb_vertex,
                                                             signal_hemi2=config.signal_hemi,
                                                             signal_freq2=config.signal_freq2,
                                                             trial_length=config.trial_length,
                                                             n_trials=config.n_trials,
                                                             noise_multiplier=config.noise,
                                                             random_state=config.random,
                                                             labels=labels, er_raw=er_raw,
                                                             fn_stc_signal1=None, fn_stc_signal2=None,
                                                             fn_simulated_raw=None, fn_report_h5=fn_report_h5)

    # TODO: delete if not needed
    stc_signal = add_stcs(stc_signal1, stc_signal2)

    ###############################################################################
    # Create epochs
    ###############################################################################

    # TODO: make sure nothing important is overwritten in report
    epochs = create_epochs(raw, config.trial_length, config.n_trials,
                           fn_simulated_epochs=None, fn_report_h5=fn_report_h5)

    epochs_grad = epochs.copy().pick_types(meg='grad')
    epochs_mag = epochs.copy().pick_types(meg='mag')
    epochs_joint = epochs.copy().pick_types(meg=True)

    # Make cov matrix
    # TODO: would it not be better to compute cov
    cov = mne.compute_covariance(epochs, method='empirical')
    noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.3, method='empirical')

    evoked_grad = epochs_grad.average()
    evoked_mag = epochs_mag.average()
    evoked_joint = epochs_joint.average()

    ###############################################################################
    # Compute LCMV beamformer results
    ###############################################################################

    # TODO: better organization of results
    corrs = []
    for setting in settings:
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

            corr = compute_lcmv_beamformer_results_two_sources(setting, evoked, cov, noise_cov, fwd_man,
                                                               signal_vertex1=config.vertex, signal_vertex2=nb_vertex,
                                                               signal_hemi=config.signal_hemi)
            corrs.append([setting, corr])

    corr_list.append([nb_dist, corrs])

# TODO: save results in report
