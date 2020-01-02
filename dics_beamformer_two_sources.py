from itertools import product

import mne
import numpy as np
import pandas as pd
from mne.time_frequency import csd_morlet

import config
from config import fname
from spatial_resolution import compute_dics_beamformer_results_two_sources
from spatial_resolution import get_nearest_neighbors
from time_series import simulate_raw_two_sources, create_epochs

fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)

###############################################################################
# Compute the settings grid
###############################################################################

regs = [0.05, 0.1, 0.5]
sensor_types = ['grad', 'mag']
pick_oris = [None, 'normal', 'max-power']
inversions = ['single', 'matrix']
weight_norms = ['unit-noise-gain', 'nai', None]
normalize_fwds = [True, False]
real_filters = [True, False]

settings = list(product(regs, sensor_types, pick_oris, inversions,
                        weight_norms, normalize_fwds, real_filters))

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

corrs = []

n_settings = len(settings)
do_break = np.zeros(shape=n_settings, dtype=bool)

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

    ###############################################################################
    # Create epochs
    ###############################################################################

    title = 'Simulated evoked for two signal vertices'
    epochs = create_epochs(raw, config.trial_length, config.n_trials, title=title,
                           fn_simulated_epochs=None, fn_report_h5=fn_report_h5)

    # The DICS beamformer currently only uses one sensor type
    epochs_grad = epochs.copy().pick_types(meg='grad')
    epochs_mag = epochs.copy().pick_types(meg='mag')

    # Make CSD matrix
    csd = csd_morlet(epochs, [config.signal_freq])

    ###############################################################################
    # Compute DICS beamformer results with all possible settings
    ###############################################################################

    for idx_setting, setting in enumerate(settings):
        (reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd,
         real_filter) = setting
        try:
            if sensor_type == 'grad':
                epo_info = epochs_grad.info
            elif sensor_type == 'mag':
                epo_info = epochs_mag.info
            else:
                raise ValueError('Invalid sensor type: %s', sensor_type)

            corr = compute_dics_beamformer_results_two_sources(setting, epo_info, csd, fwd_man,
                                                               signal_vertex1=config.vertex,
                                                               signal_vertex2=nb_vertex,
                                                               signal_hemi=config.signal_hemi)

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

###############################################################################
# Save everything to a pandas dataframe
###############################################################################

df = pd.DataFrame(corrs, columns=['reg', 'sensor_type', 'pick_ori', 'inversion',
                                  'weight_norm', 'normalize_fwd', 'real_filter',
                                  'nb_vertex', 'nb_dist', 'corr'])

df.to_csv(fname.dics_results_2s(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi))
