import mne
import argparse
import numpy as np
from config import fname

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

report = mne.open_report(fname.report(subject=subject))

# Fit ICA to the continuous data
raw_detrended = mne.io.read_raw_fif(fname.raw_detrend(subject=subject))
ica = mne.preprocessing.ICA(n_components=100).fit(raw_detrended)

# Get ICA components that capture eye blinks and heart beats
eog_epochs = mne.preprocessing.create_eog_epochs(raw_detrended)
_, eog_scores = ica.find_bads_eog(eog_epochs)
eog_bads = list(np.flatnonzero(abs(eog_scores) > 0.2))
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_detrended)
ecg_bads, ecg_scores = ica.find_bads_ecg(ecg_epochs)
ica.exclude = eog_bads + ecg_bads
print(eog_bads)
print(ecg_bads)

if len(eog_bads) > 0:
    report.add_figs_to_section(ica.plot_scores(eog_scores), 'Correlation between ICA components and EOG channel', 'ICA', replace=True)
    report.add_figs_to_section(ica.plot_properties(eog_epochs, picks=eog_bads), ['Properties of EOG component %02d' % e for e in eog_bads], 'ICA', replace=True)
if len(ecg_bads) > 0:
    report.add_figs_to_section(ica.plot_scores(ecg_scores), 'Correlation between ICA components and ECG channel', 'ICA', replace=True)
    report.add_figs_to_section(ica.plot_properties(ecg_epochs, picks=ecg_bads), ['Properties of ECG component %02d' % e for e in ecg_bads], 'ICA', replace=True)
report.add_figs_to_section(ica.plot_overlay(eog_epochs.average()), 'EOG signal removed by ICA', 'ICA', replace=True)
report.add_figs_to_section(ica.plot_overlay(ecg_epochs.average()), 'ECG signal removed by ICA', 'ICA', replace=True)

ica.save(fname.ica(subject=subject))

report.save(fname.report(subject=subject), overwrite=True, open_browser=False)
report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
