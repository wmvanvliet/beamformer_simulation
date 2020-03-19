import os.path as op
import argparse

import matplotlib.pyplot as plt
import mne
from nilearn.plotting import plot_anat
import numpy as np

from config import fname, n_jobs

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

report = mne.open_report(fname.report(subject=subject))

epochs = mne.read_epochs(fname.epochs(subject=subject))
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='shrunk', rank='info')
bem = mne.read_bem_solution(fname.bem(subject=subject))
trans = mne.transforms.read_trans(fname.trans(subject=subject))

# Find the slope of the onset
evoked = epochs.average().crop(0.03, 0.05)
_, mag_peak = evoked.get_peak('mag')
_, grad_peak = evoked.get_peak('grad')
peak_time = (mag_peak + grad_peak) / 2
evoked = epochs.average().crop(peak_time - 0.005, peak_time)
print(evoked)

dip, res = mne.fit_dipole(evoked, noise_cov, bem, trans, n_jobs=n_jobs, verbose=True)
dip.save(fname.ecd(subject=subject), overwrite=True)

# Plot the result in 3D brain with the MRI image using Nilearn
mri_pos = mne.head_to_mri(dip.pos, mri_head_t=trans,
                          subject=fname.subject_id(subject=subject), subjects_dir=fname.subjects_dir)
t1_fname = op.join(fname.subjects_dir, fname.subject_id(subject=subject), 'mri', 'T1.mgz')
fig = plt.figure()
plot_anat(t1_fname, cut_coords=mri_pos[np.argmax(dip.gof)], title='Dipole loc.', figure=fig)
report.add_figs_to_section(fig, 'ECD source location', 'Source level', replace=True)

report.save(fname.report(subject=subject), overwrite=True, open_browser=False)
report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
