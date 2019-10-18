import mne
import os.path as op
import numpy as np
from matplotlib import pyplot as plt
from mne.simulation import simulate_sparse_stc, simulate_raw as simulate_raw_mne
from scipy.signal import butter, filtfilt

import config
from utils import add_stcs, set_directory


def generate_signal(times, freq=10., n_trial=2, phase_lock=False):
    """Simulate a time series.

    Parameters:
    -----------
    times : np.array
        Time points.
    freq : float
        Frequency of oscillations in Hz.
    n_trial : int
        Number of trials, defaults to 1.
    """
    signal = np.zeros_like(times)

    for trial in range(n_trial):
        envelope = np.exp(50. * -(times - 0.5 - trial) ** 2.)
        if phase_lock is False:
            phase = config.random.rand() * 2 * np.pi
            signal += np.cos(phase + freq * 2 * np.pi * times) * envelope
        else:
            signal += np.cos(freq * 2 * np.pi * times) * envelope
    return signal * 1e-7


def generate_random(times, lowpass=40):
    """Simulate a random time course.
    Starts out with Guassian noise and lowpass filters that.

    Parameters
    ----------
    times : np.array
        Time points.

    Returns
    -------
    signal : np.array, shape (len(times),)
        The random signal
    """
    n_samples = len(times)
    padding = n_samples // 2
    sample_rate = 1 / np.median(np.diff(times))
    signal = config.random.randn(padding + n_samples + padding)
    signal = filtfilt(*butter(4, lowpass / (sample_rate / 2)), signal)
    signal = signal[padding:-padding]
    signal /= np.amax(signal)
    signal *= 1e-7
    return signal


def simulate_raw(info, src, fwd, signal_vertex, signal_hemi, signal_freq,
                 trial_length, n_trials, noise_multiplier, random_state, labels,
                 er_raw, fn_stc_signal=None, fn_simulated_raw=None, fn_report_h5=None):
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
    fn_stc_signal : None | string
        Path where the signal source time courses are to be saved. If None the file is not saved.
    fn_simulated_raw : None | string
        Path where the raw data is to be saved. If None the file is not saved.
    fn_report_h5 : None | string
        Path where the .h5 file for the report is to be saved.

    Returns:
    --------
    raw : instance of Raw
        Simulated raw file.
    stc_signal : instance of SourceEstimate
        Source time courses of the signal.
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
    if fn_stc_signal is not None:
        set_directory(op.dirname(fn_stc_signal))
        stc_signal.save(fn_stc_signal)

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
                               bem=None, forward=fwd, cov=None)

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
        set_directory(op.dirname(fn_simulated_raw))
        raw.save(fn_simulated_raw, overwrite=True)

    # Plot the simulated raw data in the report
    if fn_report_h5 is not None:
        set_directory(op.dirname(fn_report_h5))
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

    return raw, stc_signal


def create_epochs(raw, trial_length, n_trials, fn_simulated_epochs=None, fn_report_h5=None):
    """
    Create epochs based on the raw object with the baseline
    going from 0 to 0.3 s.

    Parameters:
    -----------
    raw : instance of Raw
        Simulated raw file.
    trial_length : float
        Length of a single trial in samples.
    n_trials : int
        Number of trials to create.
    fn_simulated_raw : None | string
        Path where the epochs file is to be saved. If None the file is not saved.
    fn_report_h5 : None | string
        Path where the .h5 file for the report is to be saved.

    Returns:
    --------
    epochs : instance of Epochs
        Epochs created from the simulated raw.
    """

    events = np.hstack((
        (np.arange(n_trials) * trial_length * raw.info['sfreq'])[:, np.newaxis],
        np.zeros((n_trials, 1)),
        np.ones((n_trials, 1)),
    )).astype(np.int)

    # TODO: why is tmin=0.1? Why not -0.1 and why tmax = length - 0.1?
    #   Is it to avoid to boundary between trials?
    epochs = mne.Epochs(raw=raw, events=events, event_id=1,
                        tmin=0.1, tmax=trial_length - 0.1,
                        baseline=(None, 0.3), preload=True)

    ###############################################################################
    # Save everything
    ###############################################################################

    if fn_simulated_epochs is not None:
        set_directory(op.dirname(fn_simulated_epochs))
        epochs.save(fn_simulated_epochs, overwrite=True)

    # Plot the simulated epochs in the report
    if fn_report_h5 is not None:
        set_directory(op.dirname(fn_report_h5))
        fn_report_html = fn_report_h5.rsplit('.h5')[0] + '.html'

        with mne.open_report(fn_report_h5) as report:

            fig = epochs.average().plot_joint(picks='mag', show=False)
            report.add_figs_to_section(fig, 'Simulated evoked',
                                       section='Sensor-level', replace=True)
            report.save(fn_report_html, overwrite=True, open_browser=False)

    return epochs
