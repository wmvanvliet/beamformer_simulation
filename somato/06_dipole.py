import mne
import os.path as op
from nilearn.plotting import plot_anat
import matplotlib.pyplot as plt

from config import fname, subject_id

report = mne.open_report(fname.report)

epochs = mne.read_epochs(fname.epochs)
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='shrunk')
bem = mne.read_bem_solution(fname.bem)
trans = mne.transforms.read_trans(fname.trans)

evoked = epochs.average().crop(0.037, 0.037)
dip, res = mne.fit_dipole(evoked, noise_cov, bem, trans, n_jobs=1, verbose=True)
dip.save(fname.ecd)


# Plot the result in 3D brain with the MRI image using Nilearn
mri_pos = mne.head_to_mri(dip.pos, mri_head_t=trans,
                          subject=subject_id, subjects_dir=fname.subjects_dir)
t1_fname = op.join(fname.subjects_dir, subject_id, 'mri', 'T1.mgz')
fig = plt.figure()
plot_anat(t1_fname, cut_coords=mri_pos[0], title='Dipole loc.', figure=fig)
report.add_figs_to_section(fig, 'ECD source location', 'Source level', replace=True)

report.save(fname.report, overwrite=True, open_browser=False)
report.save(fname.report_html, overwrite=True, open_browser=False)
