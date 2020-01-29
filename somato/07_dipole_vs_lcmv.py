import mne
import numpy as np
from jumeg.jumeg_volume_plotting import plot_vstc_sliced_old
from mne.beamformer import make_lcmv, apply_lcmv

from config import fname, subject_id
from config_bf import lcmv_settings
from utils import make_dipole_volume

report = mne.open_report(fname.report)

###############################################################################
# Load the data
###############################################################################

epochs = mne.read_epochs(fname.epochs)
bem = mne.read_bem_solution(fname.bem)
trans = mne.transforms.read_trans(fname.trans)
fwd = mne.read_forward_solution(fname.fwd)

###############################################################################
# Sensor-level analysis for beamformer
###############################################################################

epochs_grad = epochs.copy().pick_types(meg='grad')
epochs_mag = epochs.copy().pick_types(meg='mag')
epochs_joint = epochs.copy().pick_types(meg=True)

# Make cov matrices
cov = mne.compute_covariance(epochs, tmin=0, tmax=None, method='empirical')
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='empirical')

# Compute evokeds
evoked_grad = epochs_grad.average().crop(0.037, 0.037)
evoked_mag = epochs_mag.average().crop(0.037, 0.037)
evoked_joint = epochs_joint.average().crop(0.037, 0.037)

###############################################################################
# Dipole fit
###############################################################################

evoked = epochs.average().crop(0.037, 0.037)
noise_cov_shrunk = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method='shrunk')

dip, res = mne.fit_dipole(evoked, noise_cov_shrunk, bem, trans, n_jobs=1, verbose=True)
dip.save(fname.ecd)

# get the position of the dipole in MRI coordinates
mri_pos = mne.head_to_mri(dip.pos, mri_head_t=trans,
                          subject=subject_id, subjects_dir=fname.subjects_dir)

dists = []
evals = []
corrs = []
ori_errors = []
for setting in lcmv_settings:
    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, use_noise_cov, reduce_rank = setting

    if sensor_type == 'grad':
        evoked = evoked_grad
    elif sensor_type == 'mag':
        evoked = evoked_mag
    elif sensor_type == 'joint':
        evoked = evoked_joint
    else:
        raise ValueError('Invalid sensor type: %s', sensor_type)

    filters = make_lcmv(evoked.info, fwd, cov, reg=reg,
                        pick_ori=pick_ori, weight_norm=weight_norm,
                        inversion=inversion, normalize_fwd=normalize_fwd,
                        noise_cov=noise_cov if use_noise_cov else None,
                        reduce_rank=reduce_rank)

    stc = apply_lcmv(evoked, filters)

    # Compute distance between true and estimated source
    dip_est = make_dipole_volume(stc, fwd['src'])
    dist = np.linalg.norm(dip.pos - dip_est.pos)

    # fn_image = 'dipole_vs_lcmv_dist_ortho.png'
    # fp_image = op.join(image_path, fn_image_dist)
    fp_image = None
    cbar_range = [stc.data.min(), stc.data.max()]

    plot_vstc_sliced_old(stc, vsrc=fwd['src'], tstep=stc.tstep,
                         subjects_dir=fname.subjects_dir,
                         time=stc.tmin, cut_coords=mri_pos,
                         display_mode='ortho', figure=None,
                         axes=None, colorbar=True, cmap='magma_r',
                         symmetric_cbar='auto', threshold=0,
                         cbar_range=cbar_range,
                         save=False, fname_save=fp_image)

###############################################################################
# Save everything
###############################################################################

# TODO
