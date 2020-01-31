import mne
from config import fname, subject_id

epochs = mne.read_epochs(fname.epochs)
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='shrunk', rank='info')
fwd = mne.read_forward_solution(fname.fwd)
inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov)
stc = mne.minimum_norm.apply_inverse(epochs.average(), inv)
stc.subject = subject_id
stc.save(fname.stc_mne)

stc_peak = abs(stc.copy().crop(0.030, 0.050).mean())
stc_peak.save_as_volume(fname.nii_mne, src=fwd['src'])

fig = stc.plot(initial_time=0.04, subject=subject_id, subjects_dir=fname.subjects_dir, src=fwd['src'],
               clim=dict(kind='percent', lims=[99.9, 99.95, 100]))
with mne.open_report(fname.report) as report:
    report.add_figs_to_section(fig, 'MNE Source estimate at 40ms', 'Source level', replace=True)
    report.save(fname.report_html, overwrite=True, open_browser=False)
