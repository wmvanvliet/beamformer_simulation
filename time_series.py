import os.path as op
from datetime import datetime

import mne
import numpy as np
from mne.simulation import simulate_sparse_stc, simulate_raw as simulate_raw_mne
from scipy.signal import butter, filtfilt

import config
from utils import add_text_next_to_xlabel
from utils import add_volume_stcs, set_directory


def generate_signal(times, freq=10., phase=0):
    """Simulate a time series.

    Parameters:
    -----------
    times : np.array
        Time points.
    freq : float
        Frequency of oscillations in Hz.
    phase : float
        Phase of the oscillations. From 0 to 2 * pi.
    """
    signal = np.zeros_like(times)

    for chirp in range(2):
        envelope = np.exp(100. * -(times - 0.25 - 0.5 * chirp) ** 2.)
        signal += np.cos(phase + freq * 2 * np.pi * times) * envelope
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


def create_epochs(raw, title='Simulated evoked',
                  fn_simulated_epochs=None, fn_report_h5=None):
    """
    Create epochs based on the raw object with the baseline
    going from config.tmin to config.tmax

    Parameters:
    -----------
    raw : instance of Raw
        Simulated raw file.
    fn_simulated_raw : None | string
        Path where the epochs file is to be saved. If None the file is not saved.
    fn_report_h5 : None | string
        Path where the .h5 file for the report is to be saved.

    Returns:
    --------
    epochs : instance of Epochs
        Epochs created from the simulated raw.
    """
    sfreq = raw.info['sfreq']
    trial_length = int((config.tmax - config.tmin) * sfreq)
    events = np.hstack((
        (np.arange(config.n_trials) * trial_length)[:, np.newaxis] - int(config.tmin * sfreq),
        np.zeros((config.n_trials, 1)),
        np.ones((config.n_trials, 1)),
    )).astype(np.int)

    #  Use tmin=0.1 and tmax=trial_length - 0.1 to avoid edge artifacts
    epochs = mne.Epochs(raw=raw, events=events, event_id=1,
                        tmin=config.tmin + 1 / sfreq, tmax=config.tmax - 1 / sfreq,
                        baseline=(None, 0), preload=True)

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
            report.add_figs_to_section(fig, title, section='Sensor-level', replace=True)
            report.save(fn_report_html, overwrite=True, open_browser=False)

    return epochs


def simulate_raw(info, fwd_disc_true, signal_vertex, signal_freq,
                 n_trials, noise_multiplier, random_state, n_noise_dipoles,
                 er_raw, fn_stc_signal=None, fn_simulated_raw=None,
                 fn_report_h5=None):
    """
    Simulate raw time courses for two dipoles with frequencies
    given by signal_freq1 and signal_freq2. Noise dipoles are
    placed randomly in the whole cortex.

    Parameters:
    -----------
    info : instance of Info | instance of Raw
        The channel information to use for simulation.
    fwd_disc_true : instance of mne.Forward
        The forward operator for the discrete source space created with
        the true transformation file.
    signal_vertex : int
        The vertex where signal dipole is placed.
    signal_freq : float
        The frequency of the signal.
    n_trials : int
        Number of trials to create.
    noise_multiplier : float
        Multiplier for the noise dipoles. For noise_multiplier equal to one
        the signal and noise dipoles have the same magnitude.
    random_state : None | int | instance of RandomState
        If random_state is an int, it will be used as a seed for RandomState.
        If None, the seed will be obtained from the operating system (see
        RandomState for details). Default is None.
    n_noise_dipoles : int
        The number of noise dipoles to place within the volume.
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

    sfreq = info['sfreq']
    trial_length = int((config.tmax - config.tmin) * sfreq)
    times = np.arange(trial_length) / sfreq + config.tmin

    ###############################################################################
    # Simulate a single signal dipole source as signal
    ###############################################################################

    # TODO: I think a discrete source space was used because mne.simulate_raw did not take volume source spaces -> test
    src = fwd_disc_true['src']
    signal_vert = src[0]['vertno'][signal_vertex]
    data = np.asarray([generate_signal(times, freq=signal_freq)])
    vertices = np.array([signal_vert])
    stc_signal = mne.VolSourceEstimate(data=data, vertices=[vertices], tmin=times[0],
                                       tstep=np.diff(times[:2])[0], subject='sample')
    if fn_stc_signal is not None:
        set_directory(op.dirname(fn_stc_signal))
        stc_signal.save(fn_stc_signal)

    ###############################################################################
    # Create trials of simulated data
    ###############################################################################

    # select n_noise_dipoles entries from rr and their corresponding entries from nn
    raw_list = []

    for i in range(n_trials):
        # Simulate random noise dipoles
        stc_noise = simulate_sparse_stc(src, n_noise_dipoles, times,
                                        data_fun=generate_random,
                                        random_state=random_state,
                                        labels=None)

        # Project to sensor space
        stc = add_volume_stcs(stc_signal, noise_multiplier * stc_noise)

        raw = simulate_raw_mne(info, stc, trans=None, src=None,
                               bem=None, forward=fwd_disc_true)

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
        from matplotlib import pyplot as plt
        set_directory(op.dirname(fn_report_h5))
        fn_report_html = fn_report_h5.rsplit('.h5')[0] + '.html'

        now = datetime.now()
        with mne.open_report(fn_report_h5) as report:
            fig = plt.figure()
            plt.plot(times, generate_signal(times, freq=10))
            plt.xlabel('Time (s)')

            ax = fig.axes[0]
            add_text_next_to_xlabel(fig, ax, now.strftime('%m/%d/%Y, %H:%M:%S'))

            report.add_figs_to_section(fig, now.strftime('Signal time course'),
                                       section='Sensor-level', replace=True)

            fig = raw.plot()

            # axis 1 contains the xlabel
            ax = fig.axes[1]
            add_text_next_to_xlabel(fig, ax, now.strftime('%m/%d/%Y, %H:%M:%S'))

            report.add_figs_to_section(fig, now.strftime('Simulated raw'),
                                       section='Sensor-level', replace=True)
            report.save(fn_report_html, overwrite=True, open_browser=False)

    raw._annotations = mne.annotations.Annotations([], [], [])
    return raw, stc_signal


def add_source_to_raw(raw, fwd_disc_true, signal_vertex, signal_freq,
                      trial_length, n_trials, source_type):
    """
    Add a new simulated dipole source to an existing raw. Operates on a copy of
    the raw.

    Parameters:
    -----------
    raw : instance of Raw
        The raw data to add a new source to.
    fwd_disc_true : instance of mne.Forward
        The forward operator for the discrete source space created with
        the true transformation file.
    signal_vertex : int
        The vertex where signal dipole is placed.
    signal_freq : float
        The frequency of the signal.
    trial_length : float
        Length of a single trial in samples.
    n_trials : int
        Number of trials to create.
    source_type : 'chirp' | 'random'
        Type of source signal to add.

    Returns:
    --------
    raw : instance of Raw
        The summation of the original raw and the simulated source.
    stc_signal : instance of SourceEstimate
        Source time courses of the new signal.
    """
    sfreq = raw.info['sfreq']
    trial_length = int((config.tmax - config.tmin) * sfreq)
    times = np.arange(trial_length) / sfreq + config.tmin

    src = fwd_disc_true['src']
    signal_vert = src[0]['vertno'][signal_vertex]
    data = np.zeros(len(times))
    signal_part = times >= 0

    if source_type == 'chirp':
        data[signal_part] += generate_signal(times[signal_part], signal_freq, phase=0.5)
    elif source_type == 'random':
        data[signal_part] += generate_random(times[signal_part])

    vertices = np.array([signal_vert])
    stc_signal = mne.VolSourceEstimate(data=data[np.newaxis, :], vertices=vertices, tmin=times[0],
                                       tstep=np.diff(times[:2])[0], subject='sample')
    raw_signal = simulate_raw_mne(raw.info, stc_signal, trans=None, src=None,
                                  bem=None, forward=fwd_disc_true)
    raw_signal = mne.concatenate_raws([raw_signal.copy() for _ in range(n_trials)])

    raw = raw.copy()
    raw_picks = mne.pick_types(raw.info, meg=True, eeg=False)
    raw._data[raw_picks] += raw_signal._data[raw_picks]

    return raw, stc_signal
