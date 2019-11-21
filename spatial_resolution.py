import mne
import numpy as np
from itertools import product
from mne.beamformer import make_lcmv, apply_lcmv

from scipy.stats import pearsonr


def correlation(stc, signal_vertex1, signal_vertex2, signal_hemi):

    vertex1 = signal_vertex1 if signal_hemi == 0 else signal_vertex1 + stc.vertices[0].shape[0]
    vertex2 = signal_vertex2 if signal_hemi == 0 else signal_vertex2 + stc.vertices[0].shape[0]

    return pearsonr(stc[vertex1], stc[vertex2])[0]


def compute_lcmv_beamformer_results_two_sources(epochs, fwd_man, signal_vertex1, signal_vertex2, signal_hemi):

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

            filters = make_lcmv(evoked.info, fwd_man, cov, reg=reg,
                                pick_ori=pick_ori, weight_norm=weight_norm,
                                noise_cov=noise_cov if use_noise_cov else None,
                                depth=depth)

            stc = apply_lcmv(evoked, filters)

            corr = correlation(stc, signal_vertex1, signal_vertex2, signal_hemi)

        except Exception as e:
            print(e)
            corr = np.nan

        print(setting, corr)

        corrs.append(corr)

    return np.array(corrs)


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