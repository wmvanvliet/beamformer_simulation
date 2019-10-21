from config import fname, vfname
import mne
import numpy as np
from utils import random_three_vector

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

vfwd_true = mne.make_forward_solution(info, trans=trans_true_head_to_mri, src=vsrc_true_mri,
                                      bem=vbem, meg=True, eeg=False)

vfwd_man = mne.make_forward_solution(info, trans=trans_man_head_to_mri, src=vsrc_true_mri,
                                     bem=vbem, meg=True, eeg=False)
mne.write_forward_solution(vfname.fwd_man, vfwd_man, overwrite=True)

###############################################################################
# Construct forward solutions for discrete source spaces
###############################################################################

rr = vsrc_true_mri[0]['rr']


def make_discrete_fwd_solution(info, vsrc_disc, vbem, trans, fn_fwd_disc):
    """
    Create a forward solution based on the discrete volume
    source space and the trans file provided.

    Parameters:
    -----------
    info : instance of mne.Info | str
        If str, then it should be a filename to a Raw, Epochs, or Evoked
        file with measurement information. If dict, should be an info
        dict (such as one from Raw, Epochs, or Evoked).
    vsrc_disc : instance of SourceSpaces
        The discrete source space for which a forward solution is to be created.
    vbem : dict | str
        Filename of the volume BEM (e.g., "sample-5120-bem-sol.fif") to
    trans : str
        The head<->MRI transform.
    fn_vfwd_disc : str
        File name to save the forward solution to. It should end with -fwd.fif
        or -fwd.fif.gz.

    Returns:
    --------
    fwd_disc : instace of mne.Forward
        The discrete forward solution.
    """

    fwd_disc = mne.make_forward_solution(info, trans=trans, src=vsrc_disc,
                                         bem=vbem, meg=True, eeg=False)

    fwd_disc = mne.convert_forward_solution(fwd_disc, surf_ori=True,
                                            force_fixed=True)

    mne.write_forward_solution(fn_fwd_disc, fwd_disc, overwrite=False)

    return fwd_disc


def make_discrete_forward_solutions(info, rr, vbem, trans_true, trans_man,
                                    fn_fwd_disc_true, fn_fwd_disc_man):
    """
    Create a discrete source space based on the rr coordinates and
    make one forward solution for the true trans file and one for
    the manually created trans file.

    Parameters:
    -----------
    info : instance of mne.Info | str
        If str, then it should be a filename to a Raw, Epochs, or Evoked
        file with measurement information. If dict, should be an info
        dict (such as one from Raw, Epochs, or Evoked).
    rr : np.array of shape (n_vertices, 3)
        The coordinates of the volume source space.
    vbem : dict | str
        Filename of the volume BEM (e.g., "sample-5120-bem-sol.fif") to
        use, or a loaded sphere model (dict).
    trans_true : str
        The true head<->MRI transform.
    trans_man : str
        The manually created head<->MRI transform.
    fn_fwd_disc_true : str
        Path where the forward solution corresponding to the true
        transformation is to be saved.
    fn_fwd_disc_man : str
        Path where the forward solution corresponding to the manually
        created transformation is to be saved.

    Returns:
    --------
    fwd_disc_true : instance of mne.Forward
        The discrete forward solution created with the true trans file.
    fwd_disc_man : instance of mne.Forward
        The discrete forward solution created with the manual trans file.
    """
    
    ###########################################################################
    # Construct source space normals as random tangential vectors
    ###########################################################################
    
    com = rr.mean(axis=0)  # center of mass
    
    # get vectors pointing from center of mass to voxels
    radial = rr - com
    rnd_vectors = np.array([random_three_vector() for i in range(rr.shape[0])])
    tangential = np.cross(radial, rnd_vectors)
    # normalize to unit length
    nn = (tangential.T * (1. / np.linalg.norm(tangential, axis=1))).T
    
    pos = {'rr': rr, 'nn': nn}

    ###########################################################################
    # make discrete source space
    ###########################################################################

    # setup_volume_source_space sets coordinate frame to MRI
    vsrc_disc_mri = mne.setup_volume_source_space(subject='sample', pos=pos,
                                                  mri=None, bem=vbem)

    fwd_disc_true = make_discrete_fwd_solution(info, vsrc_disc_mri, vbem,
                                               trans_true, fn_fwd_disc_true)

    fwd_disc_man = make_discrete_fwd_solution(info, vsrc_disc_mri, vbem,
                                              trans_man, fn_fwd_disc_man)

    return fwd_disc_true, fwd_disc_man


vfwd_disc_true, vfwd_disc_man = make_discrete_forward_solutions(info, rr, vbem, trans_true_head_to_mri,
                                                                trans_man_head_to_mri, vfname.fwd_discrete_true,
                                                                vfname.fwd_discrete_man)

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

