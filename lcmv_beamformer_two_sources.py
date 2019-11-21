import mne
import numpy as np

import config
from config import fname
from utils import add_stcs

from time_series import simulate_raw_two_sources, create_epochs
from spatial_resolution import compute_lcmv_beamformer_results_two_sources, get_nearest_neighbors

# TODO: maybe create a second separate report or make sure that nothing gets overwritten
fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)

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

correlation = []

for nb_vertex, nb_dist in np.column_stack(nearest_neighbors, distances):

    ###############################################################################
    # Simulate raw data and create epochs
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

    stc_signal = add_stcs(stc_signal1, stc_signal2)

    # TODO: make sure nothing important is overwritten in report
    epochs = create_epochs(raw, config.trial_length, config.n_trials,
                           fn_simulated_epochs=None, fn_report_h5=fn_report_h5)

    ###############################################################################
    # Compute LCMV beamformer results
    ###############################################################################

    corrs = compute_lcmv_beamformer_results_two_sources(epochs, fwd_man, signal_vertex1=config.vertex,
                                                        signal_vertex2=nb_vertex, signal_hemi=config.signal_hemi)

    correlation.append([nb_dist, corrs.mean()])

# TODO: save results in report
