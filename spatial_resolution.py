import mne
import numpy as np
from itertools import product
from mne.beamformer import make_lcmv, apply_lcmv

from scipy.stats import pearsonr


def correlation(stc, signal_vertex1, signal_vertex2, signal_hemi):

    vertex1 = signal_vertex1 if signal_hemi == 0 else signal_vertex1 + stc.vertices[0].shape[0]
    vertex2 = signal_vertex2 if signal_hemi == 0 else signal_vertex2 + stc.vertices[0].shape[0]

    return pearsonr(stc.data[vertex1], stc.data[vertex2])[0]


def compute_lcmv_beamformer_results_two_sources(setting, evoked, cov, noise_cov, fwd_man,
                                                signal_vertex1, signal_vertex2, signal_hemi):

    reg, sensor_type, pick_ori, weight_norm, use_noise_cov, depth = setting
    try:
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

    return corr


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