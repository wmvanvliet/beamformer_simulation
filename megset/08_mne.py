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

epochs = mne.read_epochs(fname.epochs(subject=subject))
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='shrunk')
fwd = mne.read_forward_solution(fname.fwd(subject=subject))
inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov)
stc = mne.minimum_norm.apply_inverse(epochs.average(), inv)
stc.save(fname.stc_mne(subject=subject))

# Find the time point that corresponds to the best dipole fit
dip = mne.read_dipole(fname.ecd(subject=subject))
peak_time = dip[int(np.argmax(dip.gof))].times[0]

stc_peak = abs(stc.copy().crop(peak_time, peak_time).mean())
stc_peak.save(fname.stc_mne(subject=subject))
stc_peak.save_as_volume(fname.nii_mne(subject=subject), src=fwd['src'])

fig = stc.plot(initial_time=0.04, subject=fname.subject_id(subject=subject), subjects_dir=fname.subjects_dir, src=fwd['src'],
               clim=dict(kind='percent', lims=[99.9, 99.95, 100]))
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(fig, f'MNE Source estimate at {peak_time}', 'Source level', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
