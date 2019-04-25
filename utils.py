import os
import fnmatch
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
    """Find the peak in a distributed source estimate and make a dipole out of it"""
    stc = abs(stc).mean()
    max_idx = stc.get_peak(vert_as_index=True)[0]
    max_vertno = stc.get_peak()[0]
    max_hemi = int(max_idx < len(stc.vertices[0]))

    pos = src[max_hemi]['rr'][max_vertno]
    dip = mne.Dipole(stc.times, pos, stc.data[max_idx], [1., 0., 0.], 1)
    return dip


def make_dipole_volume(stc, src):
    """Find the peak in a distributed source estimate and make a dipole out of it
    for volume source space data"""
    stc = abs(stc).mean()
    max_idx = stc.get_peak(vert_as_index=True)[0]
    max_vertno = stc.get_peak()[0]

    pos = src[0]['rr'][max_vertno]
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


def evaluate_stc_volume(stc_est, stc_signal):
    # Find the estimated source distribution at peak activity
    peak_time = stc_est.get_peak(time_as_index=True)[1]
    estimate = abs(stc_est).data[:, peak_time]

    # Normalize the estimated source distribution to sum to 1
    estimate /= estimate.sum()

    # Measure the estimated activity left at the true signal location
    true_vert = stc_signal.vertices[0]
    true_vert_idx = np.hstack(stc_est.vertices) == true_vert
    return estimate[true_vert_idx][0]


def add_text_next_to_xlabel(fig, ax, text):
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


def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = np.random.uniform(0,np.pi*2)
    costheta = np.random.uniform(-1,1)

    theta = np.arccos(costheta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return x, y, z


def make_vstc_sequence(vstc, vsrc, subjects_dir, tmin, tmax, percentile, only_positive_values=False,
                       sfreq=None, coords=[-60, 55], grid=[4, 6], cmap='gist_ncar', threshold='min',
                       display_mode='x', res_save=[1920, 1080], fname_save='plt.png'):

    """
    Plot the activity for each time slice and create a movie from the images.

    Parameters:
    -----------
    vstc : mne.SourceEstimate
        Volume source time courses.
    vsrc : mne.SourceSpaces
        Volume source space for the subject.
    subjects_dir : str
        Path to the subject directory.
    tmin : float
        Lower value for cropping the volume source time courses.
    tmax : float
        Upper value for corpping the volume source time courses.
    percentile : float
        Percentile value used to calculate the threshold value.
        Determine part of the file names.
    only_positive_values : bool
        Constrain the plots to only positive values, e.g., if there
        are no negative values as in the case of MFT inverse solutions.
    sfreq : float
        Sampling frequency to be used for the movie creation.
    coords : 2-tuple
        Minimum and maximum value between which the slices are to be made.
    grid : 2-tuple
        Determines the number of rows and columns.
    cmap : str
        matplotlib color map
    threshold : float or 'min'
        Vertices with actvitity below the treshold are omitted in the plot. If
        'min' the minimum activity is set as the threshold.
    display_mode : 'x', 'y', 'z', or 'ortho'
        Specifies the display mode, i.e., direction of the slices, for the images.
    res_save : 2-tuple
        Resolution for the saved images.
    fname_save : str

    Returns:
    --------
    None
    """

    vstc = vstc.copy().crop(tmin, tmax)

    if sfreq is not None:
        vstc.resample(sfreq)

    tstep = vstc.tstep

    n_slices = grid[0] * grid[1]

    coords_min = coords[0]
    coords_max = coords[1]

    step_size = (coords_max - coords_min) / float(n_slices)

    cut_coords_z = np.arange(coords_min, coords_max, step_size) + 0.5 * step_size

    for idx, time in enumerate(vstc.times):

        frmt = fname_save.split('.')[-1]
        fn_image = fname_save.split('.png')[0] + '_' + display_mode + '_' + str(int(10 * percentile)).zfill(3)
        fn_image = fn_image + '_' + str(idx).zfill(3) + '.' + frmt

        title = "%.3f s" % time

        from jumeg.jumeg_volmorpher import plot_vstc_sliced_grid
        plot_vstc_sliced_grid(subjects_dir=subjects_dir, vstc=vstc, vsrc=vsrc,
                              tstep=tstep, title=title, time=time, display_mode=display_mode,
                              cut_coords=cut_coords_z, cmap=cmap, threshold=threshold,
                              only_positive_values=only_positive_values, grid=grid,
                              res_save=res_save, fn_image=fn_image)


def find_files(rootdir='.', pattern='*'):
    """
    Looks for all files in the root directory matching the file
    name pattern.

    Parameters:
    -----------
    rootdir : str
        Path to the directory to be searched.
    pattern : str
        File name pattern to be looked for.

    Returns:
    --------
    files : list
        List of file names matching the pattern.
    """

    files = []
    for root, dirnames, filenames in os.walk(rootdir):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    files = sorted(files)

    return files