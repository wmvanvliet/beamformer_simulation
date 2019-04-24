import mne
import numpy as np
from scipy.spatial import distance
import surfer

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


def add_volume_stcs(stc1, stc2):
    """Adds two SourceEstimates together, allowing for different vertices."""
    vertices = np.union1d(stc1.vertices, stc2.vertices)

    assert stc1.data.shape[1] == stc2.data.shape[1]
    assert stc1.tmin == stc2.tmin
    assert stc1.tstep == stc2.tstep

    data = np.zeros((len(vertices), stc1.data.shape[1]))
    for i, vert in enumerate(vertices):
        if vert in stc1.vertices:
            data[[i]] += stc1.data[stc1.vertices == vert]
        if vert in stc2.vertices:
            data[[i]] += stc2.data[stc2.vertices == vert]

    return mne.VolSourceEstimate(data, vertices, tmin=stc1.tmin, tstep=stc1.tstep)


def plot_estimation(stc_est, stc_signal, initial_time=1.5, surface='inflated'):
    """Plots the source estimate, along with the true signal location"""
    brain = stc_est.plot(hemi='both', subject='sample', initial_time=initial_time, surface=surface)
    hemi = ['lh', 'rh'][config.signal_hemi]
    vert = stc_signal.vertices[config.signal_hemi][0]
    brain.add_foci([vert], coords_as_verts=True, hemi=hemi)
    return brain


def compute_distances(src):
    """Computes vertex to vertex distance matrix, given a src"""
    rr = np.vstack((src[0]['rr'][src[0]['inuse'].astype(np.bool)],
                    src[1]['rr'][src[1]['inuse'].astype(np.bool)]))
    return distance.squareform(distance.pdist(rr))


def plot_distance(stc_est, stc_signal, D, surface='inflated'):
    """Plots the distance to the peak estimated signal, along with the true signal location"""
    peak = stc_est.get_peak(vert_as_index=True)[0]
    peak_hemi = 0 if peak < len(stc_est.vertices[0]) else 1 
    true_hemi = config.signal_hemi

    est_vert = np.hstack(stc_est.vertices)[peak]
    true_vert = stc_signal.vertices[true_hemi][0]

    brain = surfer.Brain('sample', hemi='both', surf=surface)
    brain.add_data(D[peak, :len(stc_est.vertices[0])], vertices=stc_est.vertices[0],
                   hemi='lh', transparent=True)  
    brain.add_data(D[peak, len(stc_est.vertices[0]):], vertices=stc_est.vertices[1],
                   hemi='rh', transparent=True)  
    brain.add_foci([est_vert], coords_as_verts=True, hemi='lh' if peak_hemi == 0 else 'rh', color='red')
    brain.add_foci([true_vert], coords_as_verts=True, hemi='lh' if true_hemi == 0 else 'rh', color='green')
    return brain


def make_dipole(stc, src):
    """Find the peak in a distrubuted source estimate and make a dipole out of it"""
    stc = abs(stc).mean()
    max_idx = stc.get_peak(vert_as_index=True)[0]
    max_vertno = stc.get_peak()[0]
    max_hemi = int(max_idx < len(stc.vertices[0]))

    pos = src[max_hemi]['rr'][max_vertno]
    dip = mne.Dipole(stc.times, pos, stc.data[max_idx], [1., 0., 0.], 1)
    return dip


def evaluate_stc(stc_est, stc_signal):
    # Find the estimated source distribution at peak activity
    peak_time = stc_est.get_peak(time_as_index=True)[1]
    estimate = abs(stc_est).data[:, peak_time]

    # Normalize the estimated source distribution to sum to 1
    estimate /= estimate.sum()

    # Measure the estimated activity left at the true signal location
    true_hemi = config.signal_hemi
    true_vert = stc_signal.vertices[true_hemi][0]
    true_vert_idx = np.hstack(stc_est.vertices) == true_vert
    return estimate[true_vert_idx][0]


def add_timestamp_next_to_xlabel(fig, ax, text):
    """
    Add text to the right of the label of the x-axis
    in the same style.

    Parameters
    ----------
    fig : matplotlib.pyplot.figure
        The figure containing the axis given by ax.
    ax : matplotlib.axes.Axes
        Axis containing the x-axis label.
    text : str
        Text to add to the figure.

    Returns
    -------
    None
    """

    xlbl = ax.xaxis.get_label()

    # draw figure using renderer because axis position only fixed after drawing
    fig.canvas.draw()

    transform = xlbl.get_transform()
    font_properties = xlbl.get_font_properties()
    position = xlbl.get_position()
    ha = xlbl.get_horizontalalignment()
    va = xlbl.get_verticalalignment()

    txt = ax.text(0., 0, text)

    txt.set_transform(transform)
    txt.set_position((position[0] * 1.7, position[1]))
    txt.set_font_properties(font_properties)
    txt.set_horizontalalignment(ha)
    txt.set_verticalalignment(va)