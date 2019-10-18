from config import fname, vfname
import mne
import numpy as np

###############################################################################
# Create forward solutions based on manually created trans file
###############################################################################

# use same settings as in https://github.com/mne-tools/mne-scripts/tree/master/sample-data

info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
trans_man_head_to_mri = mne.read_trans(fname.trans_man)
trans_true_head_to_mri = mne.read_trans(fname.trans_true)

# create forward solution for surface source space
src_true_mri = mne.read_source_spaces(fname.src)
bem = mne.read_bem_solution(fname.bem)

fwd_man = mne.make_forward_solution(info, trans=trans_man_head_to_mri, src=src_true_mri,
                                    bem=bem, meg=True, eeg=False, mindist=5.0)
mne.write_forward_solution(fname.fwd_man, fwd_man, overwrite=True)

# create forward solution for volume source space
vsrc_true_mri = mne.read_source_spaces(vfname.src)
vbem = mne.read_bem_solution(vfname.bem)

vfwd_man = mne.make_forward_solution(info, trans=trans_man_head_to_mri, src=vsrc_true_mri,
                                     bem=vbem, meg=True, eeg=False)
mne.write_forward_solution(vfname.fwd_man, vfwd_man, overwrite=True)

###############################################################################
# Check coregistration error
###############################################################################

trans_true_mri_to_head = mne.transforms._get_trans(trans_true_head_to_mri,
                                                   fro='mri', to='head')[0]
for hemi in range(2):

    rr_true_mri = src_true_mri[hemi]['rr'][src_true_mri[hemi]['vertno']]
    # Transform the source space from mri to head space with true trans file
    rr_true_mri_to_head = mne.transforms.apply_trans(trans_true_mri_to_head, rr_true_mri)
    # Transform the source space from mri space back to head space with inverse manual trans file
    rr_true_mri_to_head_to_mri = mne.transforms.apply_trans(trans_man_head_to_mri, rr_true_mri_to_head)

    distances = np.linalg.norm(rr_true_mri_to_head_to_mri - rr_true_mri, axis=1)

    print(distances.mean())

vrr_true_mri = vsrc_true_mri[0]['rr'][vsrc_true_mri[0]['vertno']]
# Transform the source space from mri to head space with true trans file
vrr_true_mri_to_head = mne.transforms.apply_trans(trans_true_mri_to_head, vrr_true_mri)
# Transform the source space from mri space back to head space with inverse manual trans file
vrr_true_mri_to_head_to_mri = mne.transforms.apply_trans(trans_man_head_to_mri, vrr_true_mri_to_head)

distances = np.linalg.norm(vrr_true_mri_to_head_to_mri - vrr_true_mri, axis=1)

print(distances.mean())

