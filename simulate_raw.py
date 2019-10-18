import mne
import numpy as np

from mne.simulation import simulate_sparse_stc
from mne.simulation import simulate_raw as simulate_raw_mne
from time_series import generate_signal, generate_random
from utils import add_stcs
from matplotlib import pyplot as plt

import config
from config import fname

info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
fwd = mne.read_forward_solution(fname.fwd_true)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
src = fwd['src']
labels = mne.read_labels_from_annot(subject='sample', parc='aparc.a2009s')

er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True)
fn_simulated_raw = fname.simulated_raw(noise=config.noise, vertex=config.vertex)
fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex)


def simulate_raw(info, src, fwd, signal_vertex, signal_hemi, signal_freq,
                 trial_length, n_trials, noise_multiplier, random_state, labels,
                 er_raw, fn_simulated_raw=None, fn_report_h5=None):
    """
    Simulate raw time courses for a single dipole with frequency
    given by signal_freq. In each label a noise dipole is placed.
    
    Parameters:
    -----------
    info : instance of Info | instance of Raw
        The channel information to use for simulation.
    src : instance of mne.SourceSpaces
        The source space for which the raw instance is computed.
    forward : instance of mne.Forward
        The forward operator to use.
    signal_vertex : int
        The vertex where signal dipole is placed.
    signal_hemi : 0 or 1
        The signal dipole is placed in the left (0) or right (1) hemisphere.
    signal_freq : float
        The frequency of the signal.
    trial_length : float
        Length of a single trial in samples.
    n_trials : int
        Number of trials to create.
    noise_multiplier : float
        Multiplier for the noise dipoles. For noise_multiplier equal to one
        the signal and noise dipoles have the same magnitude.
    random_state : None | int | instance of RandomState
        If random_state is an int, it will be used as a seed for RandomState.
        If None, the seed will be obtained from the operating system (see
        RandomState for details). Default is None.
    labels : None | list of Label
        The labels. The default is None, otherwise its size must be n_dipoles.
    er_raw : instance of Raw
        Empty room measurement to be used as sensor noise.
    fn_simulated_raw : None | string
        Path where the raw file is to be saved. If None the file is not saved.
    fn_report_h5 : None | string
        Path where the .h5 file for the report is to be saved.

    Returns:
    --------
    raw : instance of Raw
        Simulated raw file.
    """

    n_noise_dipoles = len(labels)
    times = np.arange(0, trial_length * info['sfreq']) / info['sfreq']

    ###############################################################################
    # Simulate a single signal dipole source as signal
    ###############################################################################

    signal_vertex = src[signal_hemi]['vertno'][signal_vertex]
    data = np.asarray([generate_signal(times, freq=signal_freq)])
    vertices = [np.asarray([], dtype=np.int64), np.array([], dtype=np.int64)]
    vertices[signal_hemi] = np.array([signal_vertex])
    stc_signal = mne.SourceEstimate(data=data, vertices=vertices, tmin=0,
                                    tstep=1 / info['sfreq'], subject='sample')
    stc_signal.save(fname.stc_signal(noise=noise_multiplier, vertex=signal_vertex))

    ###############################################################################
    # Create trials of simulated data
    ###############################################################################

    raw_list = []
    for i in range(n_trials):
        # Simulate random noise dipoles
        stc_noise = simulate_sparse_stc(src, n_noise_dipoles, times,
                                        data_fun=generate_random,
                                        random_state=random_state,
                                        labels=labels)

        # Project to sensor space
        stc = add_stcs(stc_signal, noise_multiplier * stc_noise)

        raw = simulate_raw_mne(info, stc, trans=None, src=None,
                               bem=None, forward=fwd, cov=None,
                               duration=trial_length)

        raw_list.append(raw)
        print('%02d/%02d' % (i + 1, n_trials))

    raw = mne.concatenate_raws(raw_list)

    # Use empty room noise as sensor noise
    raw_picks = mne.pick_types(raw.info, meg=True, eeg=False)
    er_raw_picks = mne.pick_types(er_raw.info, meg=True, eeg=False)
    raw._data[raw_picks] += er_raw._data[er_raw_picks, :len(raw.times)]

    ###############################################################################
    # Save everything
    ###############################################################################

    if fn_simulated_raw is not None:
        raw.save(fn_simulated_raw, overwrite=True)

    # Plot the simulated raw data in the report
    if fn_report_h5 is not None:
        fn_report_html = fn_report_h5.rsplit('.h5')[0] + '.html'
        with mne.open_report(fn_report_h5) as report:
            fig = plt.figure()
            plt.plot(times, generate_signal(times, freq=10))
            plt.xlabel('Time (s)')
            report.add_figs_to_section(fig, 'Signal time course',
                                       section='Sensor-level', replace=True)

            fig = raw.plot()
            report.add_figs_to_section(fig, 'Simulated raw', section='Sensor-level',
                                       replace=True)
            report.save(fn_report_html, overwrite=True, open_browser=False)

    return raw


simulate_raw(info, src, fwd, config.vertex, config.signal_hemi,
             config.signal_freq, config.trial_length, config.n_trials,
             config.noise, config.random, labels, er_raw,
             fn_simulated_raw=fn_simulated_raw, fn_report_h5=fn_report_h5)