import mne
import argparse
import numpy as np
from config import fname, events_id, subjects_with_extra_stim_artifacts, stim_artifact_sensor

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

report = mne.open_report(fname.report(subject=subject))

raw = mne.io.read_raw_fif(fname.raw_filt(subject=subject))
ica = mne.preprocessing.read_ica(fname.ica(subject=subject))

# Create short epochs for evoked analysis
events = mne.find_events(raw, shortest_event=0.01)
epochs = mne.Epochs(raw, events, events_id, tmin=-0.2, tmax=0.5, reject=None, baseline=(-0.2, 0), preload=True)
report.add_figs_to_section(epochs.average().plot_joint(times=[0.035, 0.1]), ['Evokeds without cleaning (grads)', 'Evokeds without cleaning (mags)'], 'Sensor level', replace=True)

# Apply ICA to remove EOG and ECG artifacts
epochs = ica.apply(epochs)

# Do a first pass for fixing the stim artifact
mne.preprocessing.fix_stim_artifact(epochs)
epochs.save(fname.epochs(subject=subject), overwrite=True)

# For some subjects, there are more stim artifacts that we need to remove using ICA
if subject in subjects_with_extra_stim_artifacts:
    ica2 = mne.preprocessing.ICA(0.9).fit(epochs.copy().crop(-0.1, 0.1))
    stim_scores = ica2.score_sources(epochs, stim_artifact_sensor[subject])
    ica2.exclude = mne.preprocessing.find_outliers(stim_scores, threshold=5)
    report.add_figs_to_section(ica2.plot_scores(stim_scores), [f'Correlation with {stim_artifact_sensor[subject]}'], 'ICA', replace=True)
    report.add_figs_to_section(ica2.plot_overlay(epochs.average()), ['Signal removed by second ICA'], 'ICA', replace=True)
    epochs = ica2.apply(epochs)

report.add_figs_to_section(epochs.average().plot_joint(times=[0.035, 0.1]), ['Evokeds after cleaning (grads)', 'Evokeds after cleaning (mags)'], 'Sensor level', replace=True)

# Create longer epochs for rhythmic analysis
epochs_long = mne.Epochs(raw, events, events_id, tmin=-1.5, tmax=2, reject=None, baseline=None, preload=True)
epochs_long = ica.apply(epochs_long)
mne.preprocessing.fix_stim_artifact(epochs_long)
if subject in subjects_with_extra_stim_artifacts:
    epochs_long = ica2.apply(epochs_long)
epochs_long.save(fname.epochs_long(subject=subject), overwrite=True)

# Visualize spectral content of the longer epochs
freqs = np.logspace(np.log10(5), np.log10(40), 20)
epochs_tfr = mne.time_frequency.tfr_morlet(epochs_long, freqs, n_cycles=7, return_itc=False, n_jobs=4)
fig = epochs_tfr.plot_topo(baseline=(-0.8, 0), tmin=-0.8, tmax=1.0, mode='logratio')
fig.set_size_inches((12, 12))
report.add_figs_to_section(fig, 'Time-frequency decomposition', 'Sensor level', replace=True)

report.save(fname.report(subject=subject), overwrite=True, open_browser=False)
report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
