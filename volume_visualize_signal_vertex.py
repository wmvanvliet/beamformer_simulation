import mne
import numpy as np
from jumeg.jumeg_volume_plotting import plot_vstc_sliced_old

from config import vfname

info = mne.io.read_info(vfname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))

fwd = mne.read_forward_solution(vfname.fwd_discrete_man)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)

src = fwd['src']

rr = src[0]['rr']
vertno = src[0]['vertno']

rr_sel = rr[vertno]

###############################################################################
# Simulate a single signal dipole source as signal
###############################################################################

# center of mass
com = rr_sel.mean(axis=0)
dist = np.linalg.norm(rr_sel - com, axis=1)
idx_max = np.argmax(dist)
idx_min = np.argmin(dist)

print("vertex with min distance from center of mass %d" % idx_min)
print("vertex with max distance from center of mass %d" % idx_max)

data = np.zeros(shape=(rr_sel.shape[0], 1))
data[idx_max] = 1

stc_signal = mne.VolSourceEstimate(data=data, vertices=vertno, tmin=0,
                                   tstep=1 / info['sfreq'], subject='sample')

# needs to be set for plot_vstc_sliced_old to work
if src[0]['subject_his_id'] is None:
    src[0]['subject_his_id'] = 'sample'

plot_vstc_sliced_old(stc_signal, src, stc_signal.tstep,
                     subjects_dir=vfname.subjects_dir,
                     time=stc_signal.times[0], cut_coords=None,
                     display_mode='ortho', figure=None,
                     axes=None, colorbar=False,
                     cmap='gist_ncar', symmetric_cbar=False,
                     threshold=0, save=True, fname_save='test.png')
