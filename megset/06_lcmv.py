import mne
import argparse
import numpy as np

from config import fname, reg

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
inv = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov,
                               reg=reg[subject]['lcmv'], noise_cov=noise_cov,
                               weight_norm='unit-noise-gain',
                               pick_ori='max-power')
stc = mne.beamformer.apply_lcmv(epochs.average(), inv)
stc.save(fname.stc_lcmv(subject=subject))

# Find the time point that corresponds to the best dipole fit
dip = mne.read_dipole(fname.ecd(subject=subject))
peak_time = dip[int(np.argmax(dip.gof))].times[0]

stc_peak = abs(stc.copy().crop(peak_time, peak_time))
stc_peak.save(fname.stc_lcmv(subject=subject))
stc_peak.save_as_volume(fname.nii_lcmv(subject=subject), src=fwd['src'])

stc_power = (stc ** 2).sqrt()
fig = stc_power.plot(subject=fname.subject_id(subject=subject), subjects_dir=fname.subjects_dir, src=fwd['src'],
                     clim=dict(kind='percent', lims=[99.9, 99.95, 100]), initial_time=peak_time)
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(fig, f'LCMV Source estimate at {peak_time * 1000}ms', 'Source level', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
