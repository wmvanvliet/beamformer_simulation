import mne
from mne.beamformer import make_lcmv, apply_lcmv
import numpy as np
from itertools import product
import pandas as pd

import config
from config import fname
from utils import make_dipole, evaluate_fancy_metric

from time_series import simulate_raw_two_sources, create_epochs

fn_stc_signal = fname.stc_signal(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)
fn_simulated_raw = fname.simulated_raw(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)
fn_simulated_epochs = fname.simulated_epochs(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)

fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi)

###############################################################################
# Simulate raw data and create epochs
###############################################################################

print('simulate data')
info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
fwd_true = mne.read_forward_solution(fname.fwd_true)
fwd_true = mne.pick_types_forward(fwd_true, meg=True, eeg=False)
src_true = fwd_true['src']
er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True)
labels = mne.read_labels_from_annot(subject='sample', parc='aparc.a2009s')

connectivity = mne.spatial_src_connectivity(src_true)
connectivity_full = connectivity.toarray()

signal_vertex2 = None
signal_hemi2 = config.signal_hemi
signal_freq2 = config.signal_freq2


def get_nearest_neighbors(signal_vertex, signal_hemi, src):
    """
    Returns a list with the nearest neighbors sorted by distances
    and the distances.

    Parameters:
    -----------
    signal_vertex : int
        The vertex where the signal dipole is placed.
    signal_hemi : 0 or 1
        The signal vertex is in the left (0) or right (1) hemisphere.
    src : mne.SourceSpaces
        The source space.

    Returns:
    --------
    nearest_neighbors : np.array of shape (n_neighbors, )
        The list of neighbors starting with the nearest and
        ending with the vertex the furthest apart.
    distances : np.array of shape (n_neighbors, )
        The distances corresponding to the neighbors.
    """

    rr = src[signal_hemi]['rr'][np.where(src[signal_hemi]['inuse'])]

    rr_dist = np.linalg.norm(rr - rr[signal_vertex], axis=1)

    nearest_neighbors = np.argsort(rr_dist)
    distances = rr_dist[nearest_neighbors]

    return nearest_neighbors[1:], distances[1:]


raw, stc_signal1, stc_signal2 = simulate_raw_two_sources(info, src=src_true, fwd=fwd_true,
                                                         signal_vertex1=config.vertex, signal_hemi1=config.signal_hemi,
                                                         signal_freq1=config.signal_freq, signal_vertex2=signal_vertex2,
                                                         signal_hemi2=signal_hemi2, signal_freq2=signal_freq2,
                                                         trial_length=config.trial_length, n_trials=config.n_trials,
                                                         noise_multiplier=config.noise, random_state=config.random,
                                                         labels=labels, er_raw=er_raw, fn_stc_signal1=None,
                                                         fn_stc_signal2=None, fn_simulated_raw=None,
                                                         fn_report_h5=fn_report_h5)

del info, fwd_true, src_true, er_raw, labels

epochs = create_epochs(raw, config.trial_length, config.n_trials,
                       fn_simulated_epochs=fn_simulated_epochs,
                       fn_report_h5=fn_report_h5)

###############################################################################
# Compute LCMV beamformer results
###############################################################################

# Read in the manually created forward solution
fwd_man = mne.read_forward_solution(fname.fwd_man)
# For pick_ori='normal', the fwd needs to be in surface orientation
fwd_man = mne.convert_forward_solution(fwd_man, surf_ori=True)

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make cov matrix
cov = mne.compute_covariance(epochs, method='empirical')
noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0.3, method='empirical')

evoked_grad = epochs_grad.average()
evoked_mag = epochs_mag.average()
evoked_joint = epochs_joint.average()

# Compute the settings grid
# regs = [0.05, 0.1, 0.5]
# sensor_types = ['joint', 'grad', 'mag']
# pick_oris = [None, 'normal', 'max-power']
# weight_norms = ['unit-noise-gain', 'nai', None]
# use_noise_covs = [True, False]
# depths = [True, False]
regs = [0.05]
sensor_types = ['joint']
pick_oris = ['max-power']
weight_norms = [None]
use_noise_covs = [True]
depths = [True, False]
settings = list(product(regs, sensor_types, pick_oris, weight_norms,
                        use_noise_covs, depths))

# Compute LCMV beamformer with all possible settings
dists = []
evals = []
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

        filters = make_lcmv(evoked.info, fwd_man, cov, reg=reg,
                            pick_ori=pick_ori, weight_norm=weight_norm,
                            noise_cov=noise_cov if use_noise_cov else None,
                            depth=depth)

        stc = apply_lcmv(evoked, filters)

        # Compute distance between true and estimated source
        dip_true = make_dipole(stc_signal, fwd_man['src'])
        dip_est = make_dipole(stc, fwd_man['src'])
        dist = np.linalg.norm(dip_true.pos - dip_est.pos)

        # Fancy evaluation metric
        ev = evaluate_fancy_metric(stc, stc_signal)
    except Exception as e:
        print(e)
        dist = np.nan
        ev = np.nan
    print(setting, dist, ev)

    dists.append(dist)
    evals.append(ev)

# Save everything to a pandas dataframe
df = pd.DataFrame(settings, columns=['reg', 'sensor_type', 'pick_ori',
                                     'weight_norm', 'use_noise_cov', 'depth'])
df['dist'] = dists
df['eval'] = evals
df.to_csv(fname.lcmv_results(noise=config.noise, vertex=config.vertex, hemi=config.signal_hemi))
