import mne
import numpy as np

import config as config
from config import fname

# Read simulated raw
raw = mne.io.Raw(fname.simulated_raw(noise=config.noise, vertex=config.vertex), preload=True)

fn_simulated_epochs = fname.simulated_epochs(noise=config.noise, vertex=config.vertex)
fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex)


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
        epochs.save(fn_simulated_epochs, overwrite=True)

    # Plot the simulated epochs in the report
    if fn_report_h5 is not None:
        fn_report_html = fn_report_h5.rsplit('.h5')[0] + '.html'

        with mne.open_report(fn_report_h5) as report:

            fig = epochs.average().plot_joint(picks='mag', show=False)
            report.add_figs_to_section(fig, 'Simulated evoked',
                                       section='Sensor-level', replace=True)
            report.save(fn_report_html, overwrite=True, open_browser=False)

    return epochs


create_epochs(raw, config.trial_length, config.n_trials,
              fn_simulated_epochs=fn_simulated_epochs,
              fn_report_h5=fn_report_h5)
