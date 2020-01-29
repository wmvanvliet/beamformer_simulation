import mne
import mne_bids
import numpy as np

from config import fname

report = mne.open_report(fname.report)

# Load raw data (tSSS already applied)
raw = mne_bids.read_raw_bids(fname.raw, fname.bids_root)
raw.load_data()
report.add_figs_to_section(raw.plot_psd(), 'PSD of unfiltered raw', 'Raw', replace=True)

raw = raw.notch_filter([50, 100, 150, 200, 250])
report.add_figs_to_section(raw.plot_psd(), 'PSD of notch filtered raw', 'Raw', replace=True)

# Fit ICA to the continuous data
raw_detrended = raw.copy().filter(1, None)
ica = mne.preprocessing.ICA(n_components=0.99).fit(raw_detrended)
ica.save(fname.ica)

# Get ICA components that capture eye blinks
eog_epochs = mne.preprocessing.create_eog_epochs(raw_detrended)
_, eog_scores = ica.find_bads_eog(raw_detrended)
ica.exclude = np.flatnonzero(abs(eog_scores) > 0.2)
report.add_figs_to_section(ica.plot_scores(eog_scores), 'Correlation between ICA components and EOG channel', 'ICA',
                           replace=True)
report.add_figs_to_section(ica.plot_properties(eog_epochs, picks=ica.exclude),
                           ['Properties of component %02d' % e for e in ica.exclude], 'ICA', replace=True)
report.add_figs_to_section(ica.plot_overlay(eog_epochs.average()), 'Signal removed by ICA', 'ICA', replace=True)

# Create short epochs for evoked analysis
epochs = mne.Epochs(raw, *mne.events_from_annotations(raw), tmin=-0.2, tmax=0.5, reject=None, baseline=(-0.2, 0),
                    preload=True)
epochs_clean = ica.apply(epochs)
epochs_clean.save(fname.epochs, overwrite=True)
evoked = epochs_clean.average()
evoked.save(fname.evoked)
report.add_figs_to_section(epochs.average().plot_joint(times=[0.035, 0.1]),
                           ['Evokeds without ICA (grads)', 'Evokeds without ICA (mags)'], 'Sensor level', replace=True)
report.add_figs_to_section(epochs_clean.average().plot_joint(times=[0.035, 0.1]),
                           ['Evokeds after ICA (grads)', 'Evokeds after ICA (mags)'], 'Sensor level', replace=True)

# Create longer epochs for rhythmic analysis
epochs_long = mne.Epochs(raw, *mne.events_from_annotations(raw), tmin=-1.5, tmax=2, reject=None, baseline=None,
                         preload=True)
epochs_long = ica.apply(epochs_long)
epochs_long.save(fname.epochs_long, overwrite=True)

# Visualize spectral content of the longer repochs
freqs = np.logspace(np.log10(5), np.log10(40), 20)
epochs_tfr = mne.time_frequency.tfr_morlet(epochs_long, freqs, n_cycles=7, return_itc=False, n_jobs=4)
fig = epochs_tfr.plot_topo(baseline=(-1, 0), mode='logratio')
fig.set_size_inches((12, 12))
report.add_figs_to_section(fig, 'Time-frequency decomposition', 'Spectrum', replace=True)
report.add_figs_to_section(epochs_tfr.plot(picks=['MEG 1143'], baseline=(-1, 0), mode='logratio'), 'Time-frequency decomposition for MEG 1143', 'Spectrum', replace=True)
report.add_figs_to_section(epochs_tfr.plot(picks=['MEG 2033'], baseline=(-1, 0), mode='logratio'), 'Time-frequency decomposition for MEG 2033', 'Spectrum', replace=True)

report.save(fname.report, overwrite=True, open_browser=False)
report.save(fname.report_html, overwrite=True, open_browser=False)
