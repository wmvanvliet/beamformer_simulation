import numpy as np


def generate_signal(times, freq, n_trial=1, phase_lock=False):
    """Simulate a time series.

    Parameters:
    -----------
    times : np.array
        time vector
    freq : float
        frequency of oscillations
    n_trial : int
        number of trials, defaults to 1.
    """
    signal = np.zeros_like(times)

    for trial in range(n_trial):
        envelope = np.exp(50. * -(times - 0.5 - trial) ** 2.)
        if phase_lock is False:
            phase = np.random.rand() * 2 * np.pi
            signal += np.cos(phase + freq * 2 * np.pi * times) * envelope
        else:
            signal += np.cos(freq * 2 * np.pi * times) * envelope
    return signal * 1e-12
