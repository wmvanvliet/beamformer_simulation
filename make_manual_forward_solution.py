import mne
import numpy as np

from config import fname
from utils import make_discrete_forward_solutions

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

fwd_true = mne.make_forward_solution(info, trans=trans_true_head_to_mri, src=src_true_mri,
                                     bem=bem, meg=True, eeg=False)

fwd_man = mne.make_forward_solution(info, trans=trans_man_head_to_mri, src=src_true_mri,
                                    bem=bem, meg=True, eeg=False)
mne.write_forward_solution(fname.fwd_man, fwd_man, overwrite=True)

###############################################################################
# Construct forward solutions for discrete source spaces
###############################################################################

rr = src_true_mri[0]['rr']
# use only vertices inuse to construct vrr
rr = rr[src_true_mri[0]['inuse'] == 1]

fwd_disc_true, fwd_disc_man = make_discrete_forward_solutions(info, rr, bem, trans_true_head_to_mri,
                                                              trans_man_head_to_mri, fname.subjects_dir,
                                                              fname.fwd_discrete_true, fname.fwd_discrete_man)

###############################################################################
# Check coregistration error
###############################################################################

# surface source space coregistration error

trans_true_mri_to_head = mne.transforms._get_trans(trans_true_head_to_mri,
                                                   fro='mri', to='head')[0]

# Volume source space coregistration error

rr_true_mri = src_true_mri[0]['rr'][src_true_mri[0]['vertno']]
# Transform the source space from mri to head space with true trans file
rr_true_mri_to_head = mne.transforms.apply_trans(trans_true_mri_to_head, rr_true_mri)
# Transform the source space from head space back to mri space with inverse manual trans file
rr_true_mri_to_head_to_mri = mne.transforms.apply_trans(trans_man_head_to_mri, rr_true_mri_to_head)

distances = np.linalg.norm(rr_true_mri_to_head_to_mri - rr_true_mri, axis=1)
print('Volume: avg. distance %.4f, n_vertno %d' % (distances.mean(), len(rr_true_mri)))

src_disc_true = fwd_disc_true['src']
rr_disc_true_head = src_disc_true[0]['rr']
# true source space in head coordinates transformed to mri coordinates using manually created trans file
rr_disc_true_head_to_mri_man = mne.transforms.apply_trans(trans_man_head_to_mri, rr_disc_true_head)

src_disc_man = fwd_disc_man['src']
rr_disc_man_head = src_disc_man[0]['rr']
# manually created source space in head coordinates transformed to mri coordinates using manually created trans file
rr_disc_man_head_to_mri_man = mne.transforms.apply_trans(trans_man_head_to_mri, rr_disc_man_head)
distances_disc = np.linalg.norm(rr_disc_true_head_to_mri_man - rr_disc_man_head_to_mri_man, axis=1)
print('Discrete volume: avg. distance %.4f, n_vertno %d' % (distances_disc.mean(), len(rr_disc_true_head)))
