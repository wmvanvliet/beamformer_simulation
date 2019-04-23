import mne
import numpy as np

import config

def add_stcs(stc1, stc2):
    """Adds two SourceEstimates together, allowing for different vertices."""
    vertices = [np.union1d(stc1.vertices[0], stc2.vertices[0]),
                np.union1d(stc1.vertices[1], stc2.vertices[1])]
    assert stc1.data.shape[1] == stc2.data.shape[1]
    assert stc1.tmin == stc2.tmin
    assert stc1.tstep == stc2.tstep

    data = np.zeros((len(vertices[0]) + len(vertices[1]), stc1.data.shape[1]))
    i = 0

    # Left hemisphere
    for vert in vertices[0]:
        if vert in stc1.vertices[0]:
            data[[i]] += stc1.lh_data[stc1.vertices[0] == vert]
        if vert in stc2.vertices[0]:
            data[[i]] += stc2.lh_data[stc2.vertices[0] == vert]
        i += 1

    # Right hemisphere
    for vert in vertices[1]:
        if vert in stc1.vertices[1]:
            data[[i]] += stc1.rh_data[stc1.vertices[1] == vert]
        if vert in stc2.vertices[1]:
            data[[i]] += stc2.rh_data[stc2.vertices[1] == vert]
        i += 1

    return mne.SourceEstimate(data, vertices, tmin=stc1.tmin, tstep=stc1.tstep)


def plot_estimation(stc_est, stc_signal, initial_time=1.5):
    """Plots the source estimate, along with the true signal location"""
    brain = stc_est.plot(hemi='both', subject='sample', initial_time=initial_time)
    hemi = ['lh', 'rh'][config.signal_hemi]
    vert = stc_signal.vertices[config.signal_hemi][0]
    brain.add_foci([vert], coords_as_verts=True, hemi=hemi)
    return brain
