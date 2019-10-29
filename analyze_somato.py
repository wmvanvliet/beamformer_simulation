import mne
import mne_bids
import numpy as np
from mayavi import mlab

report = mne.Report()

# Paths
bids_root = mne.datasets.somato.data_path()
raw_fname = bids_root + '/sub-01/meg/sub-01_task-somato_meg.fif'
fwd_fname = bids_root + '/derivatives/sub-01/sub-01_task-somato-fwd.fif'
subjects_dir = bids_root + '/derivatives/freesurfer/subjects'
subject_id = '01'

# Load raw data (tSSS already applied)
raw = mne_bids.read_raw_bids(raw_fname, bids_root)
raw.load_data()
report.add_figs_to_section(raw.plot_psd(), 'PSD of unfiltered raw', 'Raw', replace=True)

# Filter the data
raw_filtered = raw.copy().filter(1, 40)
report.add_figs_to_section(raw_filtered.plot_psd(), 'PSD of bandpass filtered raw', 'Bandpass filter', replace=True)

# Fit ICA to the continuous data
ica = mne.preprocessing.ICA(n_components=0.999).fit(raw_filtered)

# Filter out eye blinks
eog_epochs = mne.preprocessing.create_eog_epochs(raw)
ica.exclude, eog_scores = ica.find_bads_eog(eog_epochs)
report.add_figs_to_section(ica.plot_scores(eog_scores), 'Correlation between ICA components and EOG channel', 'ICA', replace=True)
report.add_figs_to_section(ica.plot_properties(eog_epochs, picks=ica.exclude), ['Properties of component %02d' % e for e in ica.exclude], 'ICA', replace=True)
report.add_figs_to_section(ica.plot_overlay(eog_epochs.average()), 'Signal removed by ICA', 'ICA', replace=True)

# Create epochs
epochs = mne.Epochs(raw_filtered, *mne.events_from_annotations(raw), tmin=-1, tmax=2.5, reject=None, baseline=None, preload=True)

# Visualize spectral content of the signal
freqs = np.logspace(np.log10(5), np.log10(40), 20)
epochs_tfr = mne.time_frequency.tfr_morlet(epochs, freqs, n_cycles=7, return_itc=False, n_jobs=4)
report.add_figs_to_section(epochs_tfr.plot_topo(baseline=(-1, 0), mode='logratio'), 'Time-frequency decomposition', 'Spectrum')
report.add_figs_to_section(epochs_tfr.plot(picks=['MEG 1143'], baseline=(-1, 0), mode='logratio'), 'Time-frequency decomposition for MEG 1143', 'Spectrum')
report.add_figs_to_section(epochs_tfr.plot(picks=['MEG 2033'], baseline=(-1, 0), mode='logratio'), 'Time-frequency decomposition for MEG 2033', 'Spectrum')

# Create CSD matrices
csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-1, tmax=1.5, decim=5, n_jobs=4)
csd_baseline = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-1, tmax=0, decim=5, n_jobs=4)
csd_ers = mne.time_frequency.csd_morlet(epochs, freqs, tmin=0.5, tmax=1.5, decim=5, n_jobs=4)
csd_erd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=0, tmax=1.0, decim=5, n_jobs=4)

# Compute DICS beamformer at all frequencies
fwd = mne.read_forward_solution(fwd_fname)
info = mne.pick_info(epochs.info, mne.pick_types(epochs.info, meg='grad'))  # Boo! Only grads :(
filters = mne.beamformer.make_dics(info, fwd, csd, pick_ori='max-power')

# Compute source power
stc_baseline, _ = mne.beamformer.apply_dics_csd(csd_baseline, filters)
stc_ers, _ = mne.beamformer.apply_dics_csd(csd_ers, filters)
stc_erd, _ = mne.beamformer.apply_dics_csd(csd_erd, filters)
stc_baseline.subject = subject_id
stc_ers.subject = subject_id
stc_erd.subject = subject_id

# Normalize with baseline power.
stc_ers /= stc_baseline
stc_ers.data = np.log(stc_ers.data)
stc_erd /= stc_baseline
stc_erd.data = np.log(stc_erd.data)

screenshots = []
brain = stc_ers.plot(subject_id, hemi='both', subjects_dir=subjects_dir, views='par')
for i, freq in enumerate(freqs):
    brain.set_time(i)
    screenshots.append(mlab.screenshot(antialiased=True))
report.add_slider_to_section(screenshots, ['%.2f Hz' % freq for freq in freqs], 'DICS', 'ERS Power', replace=True)

screenshots = []
brain = stc_erd.plot(subject_id, hemi='both', subjects_dir=subjects_dir, views='par')
for i, freq in enumerate(freqs):
    brain.set_time(i)
    screenshots.append(mlab.screenshot(antialiased=True))
report.add_slider_to_section(screenshots, ['%.2f Hz' % freq for freq in freqs], 'DICS', 'ERD Power', replace=True)

report.save('somato.html', overwrite=True)
