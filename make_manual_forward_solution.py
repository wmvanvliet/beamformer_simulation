from config import fname, vfname
import mne

# use same settings as in https://github.com/mne-tools/mne-scripts/tree/master/sample-data

info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
trans_man = mne.read_trans(fname.trans_man)

# create forward solution for surface source space
src = mne.read_source_spaces(fname.src)
bem = mne.read_bem_solution(fname.bem)

fwd_man = mne.make_forward_solution(info, trans=trans_man, src=src, bem=bem,
                                    meg=True, eeg=False, mindist=5.0, n_jobs=2)
mne.write_forward_solution(fname.fwd_man, fwd_man)

# create forward solution for volume source space
vsrc = mne.read_forward_solution(vfname.fwd_true)['src']
vbem = mne.read_bem_solution(vfname.bem)

vfwd_man = mne.make_forward_solution(info, trans=trans_man, src=vsrc, bem=vbem,
                                     meg=True, eeg=False)
mne.write_forward_solution(vfname.fwd_man, vfwd_man)

