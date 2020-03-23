import os.path as op

import numpy as np
import matplotlib.pyplot as plt
import mne
from nilearn.plotting import plot_anat

from config import fname, subject_id, n_jobs

report = mne.open_report(fname.report)

epochs = mne.read_epochs(fname.epochs)
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='shrunk', rank='info')
bem = mne.read_bem_solution(fname.bem)
trans = mne.transforms.read_trans(fname.trans)

# Find the slope of the onset
evoked = epochs.average().crop(0.03, 0.05)
_, mag_peak = evoked.get_peak('mag')
_, grad_peak = evoked.get_peak('grad')
peak_time = (mag_peak + grad_peak) / 2
evoked = epochs.average().crop(peak_time - 0.005, peak_time + 0.005)
print(evoked)

dip, res = mne.fit_dipole(evoked, noise_cov, bem, trans, n_jobs=n_jobs, verbose=True)
dip = dip[int(np.argmax(dip.gof))]
dip.save(fname.ecd, overwrite=True)

# Plot the result in 3D brain with the MRI image using Nilearn
mri_pos = mne.head_to_mri(dip.pos, mri_head_t=trans,
                          subject=subject_id, subjects_dir=fname.subjects_dir)
t1_fname = op.join(fname.subjects_dir, subject_id, 'mri', 'T1.mgz')
fig = plt.figure()
plot_anat(t1_fname, cut_coords=mri_pos[0], title='Dipole loc.', figure=fig)
report.add_figs_to_section(fig, 'ECD source location', 'Source level', replace=True)

report.save(fname.report, overwrite=True, open_browser=False)
report.save(fname.report_html, overwrite=True, open_browser=False)
