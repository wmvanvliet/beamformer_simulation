import mne
import mne_bids
from mayavi import mlab

from config import fname, subject_id, n_jobs

info = mne.io.read_info(fname.raw)

# From T1-weighted MRI to forward solution
trans = mne_bids.get_head_mri_trans(fname.raw, fname.bids_root)
bem = mne.make_bem_model(subject_id, ico=4, subjects_dir=fname.subjects_dir)
bem_sol = mne.make_bem_solution(bem)
mne.write_bem_solution(fname.bem, bem_sol)

# create surface source space & forward solution
src_surf = mne.setup_source_space(subject=subject_id, subjects_dir=fname.subjects_dir, n_jobs=n_jobs)
fwd_surf = mne.make_forward_solution(info=info, trans=trans, src=src_surf, bem=bem_sol)

# create volume source space & forward solution
src = mne.setup_volume_source_space(subject=subject_id, pos=7., mri=fname.mri,
                                    bem=bem_sol, subjects_dir=fname.subjects_dir)
fwd = mne.make_forward_solution(info=info, trans=trans, src=src, bem=bem_sol)

# Save things
trans.save(fname.trans)
src_surf.save(fname.src_surf, overwrite=True)
mne.write_forward_solution(fname.fwd_surf, fwd_surf, overwrite=True)

src.save(fname.src, overwrite=True)
mne.write_forward_solution(fname.fwd, fwd, overwrite=True)

# Visualize surface source space and MEG sensors
fig = mne.viz.plot_alignment(info=info, trans=trans, subject=subject_id,
                             subjects_dir=fname.subjects_dir, meg='sensors',
                             src=src_surf, bem=bem_sol)
mlab.view(138, 73, 0.6, [0.02, 0.01, 0.03])
with mne.open_report(fname.report) as report:
    report.add_figs_to_section(fig, 'Surface source space and MEG sensors', 'Source level', replace=True)
    report.save(fname.report_html, overwrite=True, open_browser=False)

# Visualize volume source space and MEG sensors
fig = mne.viz.plot_alignment(info=info, trans=trans, subject=subject_id,
                             subjects_dir=fname.subjects_dir, meg='sensors',
                             src=src, bem=bem_sol)
mlab.view(138, 73, 0.6, [0.02, 0.01, 0.03])
with mne.open_report(fname.report) as report:
    report.add_figs_to_section(fig, 'Volume source space and MEG sensors', 'Source level', replace=True)
    report.save(fname.report_html, overwrite=True, open_browser=False)
