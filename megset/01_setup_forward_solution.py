"""
From T1-weighted MRI to forward solution
"""
import mne
import argparse
from mayavi import mlab

from config import fname

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

info = mne.io.read_info(fname.raw(subject=subject, run=1))

bem = mne.make_bem_model(fname.subject_id(subject=subject), ico=4, subjects_dir=fname.subjects_dir,
                         conductivity=[0.3, 0.006, 0.3])
bem_sol = mne.make_bem_solution(bem)
mne.write_bem_solution(fname.bem(subject=subject), bem_sol)
src = mne.setup_volume_source_space(subject=fname.subject_id(subject=subject), bem=bem_sol, subjects_dir=fname.subjects_dir)
fwd = mne.make_forward_solution(info=info, trans=fname.trans(subject=subject), src=src, bem=bem_sol, eeg=True)

# Save things
src.save(fname.src(subject=subject), overwrite=True)
mne.write_forward_solution(fname.fwd(subject=subject), fwd, overwrite=True)

# Visualize source space and MEG sensors
fig = mne.viz.plot_alignment(info=info, trans=fname.trans(subject=subject), subject=fname.subject_id(subject=subject),
                             subjects_dir=fname.subjects_dir, meg='sensors',
                             src=src, bem=bem_sol)
mlab.view(138, 73, 0.6, [0.02, 0.01, 0.03])
with mne.open_report(fname.report(subject=subject)) as report:
    report.add_figs_to_section(fig, 'Source space and MEG sensors', 'Source level', replace=True)
    report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
