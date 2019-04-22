import mne
import numpy as np

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

    # Right hemisphere
    for vert in vertices[1]:
        if vert in stc1.vertices[1]:
            data[[i]] += stc1.rh_data[stc1.vertices[1] == vert]
        if vert in stc2.vertices[1]:
            data[[i]] += stc2.rh_data[stc2.vertices[1] == vert]

    return mne.SourceEstimate(data, vertices, tmin=stc1.tmin, tstep=stc1.tstep)
