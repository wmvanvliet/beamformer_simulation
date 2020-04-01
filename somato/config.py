import mne

from fnames import FileNames

fname = FileNames()
fname.add('bids_root', mne.datasets.somato.data_path())
fname.add('prefix', 'sub-01_task-somato')
fname.add('derivatives', '{bids_root}/derivatives/sub-01')
fname.add('subjects_dir', '{bids_root}/derivatives/freesurfer/subjects')
fname.add('raw', '{bids_root}/sub-01/meg/sub-01_task-somato_meg.fif')
fname.add('trans', '{derivatives}/sub-01_task-somato_trans.fif')
fname.add('bem', '{derivatives}/sub-01_task-somato_vol-bem-sol.fif')
fname.add('src', '{derivatives}/sub-01_task-somato_vol-src.fif')
fname.add('src_surf', '{derivatives}/sub-01_task-somato-src.fif')
fname.add('fwd', '{derivatives}/sub-01_task-somato_vol-fwd.fif')
fname.add('fwd_surf', '{derivatives}/sub-01_task-somato-fwd.fif')
fname.add('ica', '{derivatives}/sub-01_task-somato_ica.fif')
fname.add('epochs', '{derivatives}/sub-01_task-somato_epo.fif')
fname.add('epochs_long', '{derivatives}/sub-01_task-somato_long_epo.fif')
fname.add('evoked', '{derivatives}/sub-01_task-somato_ave.fif')
fname.add('stc_mne', '{derivatives}/sub-01_task-somato_mne')
fname.add('stc_lcmv', '{derivatives}/sub-01_task-somato_lcmv')
fname.add('stc_dics', '{derivatives}/sub-01_task-somato_dics')
fname.add('ecd', '{derivatives}/sub-01_task-somato_ecd.dip')
fname.add('nii_mne', '{derivatives}/sub-01_task-somato_mne.nii.gz')
fname.add('nii_lcmv', '{derivatives}/sub-01_task-somato_lcmv.nii.gz')
fname.add('nii_dics', '{derivatives}/sub-01_task-somato_dics.nii.gz')
fname.add('mri', '{bids_root}/derivatives/freesurfer/subjects/01/mri/orig.mgz')
fname.add('report', 'somato.h5')
fname.add('report_html', 'somato.html')
fname.add('lcmv_somato_results', 'lcmv_somato_results.csv')
fname.add('dics_somato_results', 'dics_somato_results.csv')

subject_id = '01'
n_jobs = 4
