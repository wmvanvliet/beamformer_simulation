import numpy as np
from mne.utils import check_random_state
from scipy.signal import butter, filtfilt

import config

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
