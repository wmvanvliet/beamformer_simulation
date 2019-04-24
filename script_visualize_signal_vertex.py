import mne
import numpy as np
from jumeg.jumeg_volmorpher import plot_vstc_sliced_old

from config import fname, vfname

info = mne.io.read_info(vfname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))

fwd = mne.read_forward_solution(vfname.fwd)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)

src_mri_fname = fname.src
src_mri = mne.read_source_spaces(src_mri_fname)

src_head = fwd['src']
rr_head = src_head[0]['rr']

# there is only one volume source space
vertno_head = src_head[0]['vertno']

###############################################################################
# Simulate a single signal dipole source as signal
###############################################################################

com_head = rr_head[vertno_head].mean()
dist_head = np.linalg.norm(rr_head[vertno_head] - com_head, axis=1)
idx_max_head = np.argmax(dist_head)

signal_vertex = vertno_head[idx_max_head]

data_head = np.zeros(shape=(vertno_head.shape[0], 1))
data_head[idx_max_head] = 1

stc_signal_head = mne.VolSourceEstimate(data=data_head, vertices=vertno_head, tmin=0,
                                   tstep=1 / info['sfreq'], subject='sample')

src_head[0]['subject_his_id'] = 'sample'
# TODO: src might have to be in mri coordinates -> check
plot_vstc_sliced_old(stc_signal_head, src_head, stc_signal_head.tstep,
                     subjects_dir='/Users/ckiefer/mne_data/MNE-sample-data/subjects',
                     time=0.5, cut_coords=None,
                     display_mode='ortho', figure=None, axes=None, colorbar=False,
                     cmap='gist_ncar', symmetric_cbar=False, threshold=0,
                     save=True, fname_save='test.png')
