import mne
import argparse
from config import fname

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

epochs = mne.read_epochs(fname.epochs(subject=subject))
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='empirical')
data_cov = mne.compute_covariance(epochs, tmin=0, tmax=0.4, method='empirical')
fwd = mne.read_forward_solution(fname.fwd(subject=subject))
inv = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, weight_norm='unit-noise-gain')
#inv = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=0.05, noise_cov=noise_cov, weight_norm=None, pick_ori=None)
stc = mne.beamformer.apply_lcmv(epochs.average(), inv)
stc.save(fname.stc_lcmv(subject=subject))

stc_peak = abs(stc.copy().crop(0.035, 0.035))
stc_peak.save_as_volume(fname.nii_lcmv(subject=subject), src=fwd['src'])

fig = stc_peak.plot(subject=fname.subject_id(subject=subject), subjects_dir=fname.subjects_dir, src=fwd['src'],
                    clim=dict(kind='percent', lims=[99.8, 99.9, 100]))
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(fig, 'LCMV Source estimate at 40ms', 'Source level', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
