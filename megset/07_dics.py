import mne
import argparse
import numpy as np

from config import fname, n_jobs, freq_range, reg

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

# Create longer epochs
epochs = mne.read_epochs(fname.epochs_long(subject=subject))
epochs.apply_baseline((-0.8, 1.0))

# Compute Cross-Spectral Density matrices
#freqs = np.arange(7, 15)
freqs = np.arange(*freq_range[subject])
csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-0.8, tmax=1.0, n_jobs=n_jobs, decim=5)
csd_baseline = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-0.8, tmax=0, n_jobs=n_jobs, decim=5)
# ERS activity starts at 0.5 seconds after stimulus onset
csd_ers = mne.time_frequency.csd_morlet(epochs, freqs, tmin=0.2, tmax=1.0, n_jobs=n_jobs, decim=5)

csd = csd.mean()
csd_baseline = csd_baseline.mean()
csd_ers = csd_ers.mean()

# Compute DICS beamformer to localize ERS
fwd = mne.read_forward_solution(fname.fwd(subject=subject))

info, fwd, csd = mne.channels.equalize_channels([epochs.info, fwd, csd])
inv = mne.beamformer.make_dics(info, fwd, csd, reduce_rank=True,
                               pick_ori='max-power', inversion='matrix',
                               reg=reg[subject]['dics'])

# Compute source power
stc_baseline, _ = mne.beamformer.apply_dics_csd(csd_baseline, inv)
stc_ers, _ = mne.beamformer.apply_dics_csd(csd_ers, inv)
stc_baseline.subject = fname.subject_id(subject=subject)
stc_ers.subject = fname.subject_id(subject=subject)

# Normalize with baseline power.
stc_ers /= stc_baseline
stc_ers.data = np.log(stc_ers.data)
stc_ers.save(fname.stc_dics(subject=subject))
stc_ers.save_as_volume(fname.nii_dics(subject=subject), src=fwd['src'])

fig = stc_ers.plot(subject=fname.subject_id(subject=subject), subjects_dir=fname.subjects_dir, src=fwd['src'])
                   

with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(fig, 'DICS Source estimate', 'Source level', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
