from fnames import FileNames

subjects = [1, 2, 4, 5, 6, 7]
bad_subjects = [3]

events_id = {
    # 'visRL': 1,
    # 'visR': 2,
    # 'visL': 4,
    # 'ssR': 8,
    'ssL': 16,
    # 'ssRL': 24,
    # 'audR': 32,
    # 'audL': 64
}

bads = {
    1: ['MEG2233', 'EEG001', 'EEG035', 'EEG015'],
    2: ['MEG1041', 'EEG001', 'EEG035', 'EEG002'],
    3: ['MEG2233', 'MEG0741', 'EEG045', 'EEG035', 'EEG001', 'EEG027'],
    4: ['MEG1842', 'MEG2113', 'MEG2342', 'MEG2233', 'MEG1942', 'MEG1922', 'EEG045', 'EEG001', 'EEG035'],
    5: ['MEG2233', 'MEG0811', 'MEG2342', 'MEG0812', 'MEG0813', 'MEG0722', 'MEG0632', 'MEG0913', 'MEG0912', 'EEG001', 'EEG035', 'EEG045'],
    6: ['EEG001', 'EEG035'],
    7: ['MEG0213', 'MEG2233', 'MEG2212', 'MEG2231', 'EEG001', 'EEG035', 'EEG045'],
}


fname = FileNames()
fname.add('megset_dir', '/m/nbe/project/megset')
fname.add('target_dir', '/m/nbe/scratch/epasana/beamformer_simulation/megset/sub{subject:02d}')
fname.add('subjects_dir', '/m/nbe/scratch/epasana/beamformer_simulation/megset/mri')
fname.add('subject_id', 'k{subject:d}_T1')
fname.add('raw', '{target_dir}/sub{subject:02d}-raw.fif')
fname.add('raw_filt', '{target_dir}/sub{subject:02d}-filtered-raw.fif')
fname.add('raw_detrend', '{target_dir}/sub{subject:02d}-detrended-raw.fif')
fname.add('annotations', '{target_dir}/sub{subject:02d}-annotations.txt')
fname.add('trans', '{target_dir}/sub{subject:02d}-trans.fif')
fname.add('bem', '{target_dir}/sub{subject:02d}-vol-bem-sol.fif')
fname.add('src', '{target_dir}/sub{subject:02d}-vol-src.fif')
fname.add('fwd', '{target_dir}/sub{subject:02d}-vol-fwd.fif')
fname.add('ica', '{target_dir}/sub{subject:02d}-ica.fif')
fname.add('epochs', '{target_dir}/sub{subject:02d}-epo.fif')
fname.add('epochs_long', '{target_dir}/sub{subject:02d}-long_epo.fif')
fname.add('evoked', '{target_dir}/sub{subject:02d}-ave.fif')
fname.add('stc_mne', '{target_dir}/sub{subject:02d}_mne')
fname.add('stc_lcmv', '{target_dir}/sub{subject:02d}_lcmv')
fname.add('stc_dics', '{target_dir}/sub{subject:02d}_dics')
fname.add('nii_mne', '{target_dir}/sub{subject:02d}_mne.nii.gz')
fname.add('nii_lcmv', '{target_dir}/sub{subject:02d}_lcmv.nii.gz')
fname.add('nii_dics', '{target_dir}/sub{subject:02d}_dics.nii.gz')
fname.add('ecd', '{target_dir}/sub{subject:02d}_ecd.dip')
fname.add('report', '{target_dir}/sub{subject:02d}_report.h5')
fname.add('report_html', '{target_dir}/sub{subject:02d}_report.html')
