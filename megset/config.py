from fnames import FileNames
import getpass
from socket import getfqdn

user = getpass.getuser()  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts
print('Running on %s@%s' % (user, host))

if user == 'wmvan':
    # My work laptop
    target_path = 'X:/'
    n_jobs = 6
elif host == 'nbe-024.org.aalto.fi' and user == 'vanvlm1':
    # My workstation
    target_path = '/m/nbe/scratch/epasana/beamformer_simulation'
    n_jobs = 4
elif 'triton' in host and user == 'vanvlm1':
    # The big computational cluster at Aalto University
    target_path = '/scratch/nbe/epasana/beamformer_simulation'
    n_jobs = 1
else:
    raise RuntimeError('Please edit scripts/config.py and set the target_path '
                       'variable to point to the location where the data '
                       'should be stored and the n_jobs variable to the '
                       'number of CPU cores the analysis is allowed to use.')

subjects = [1, 2, 4, 5, 6, 7]
bad_subjects = [3]  # For subject 3, we don't have T1 MRI

# The events to create epochs for. In this study, we only care about
# stimulation to the left medial nerve.
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

# Bad channels for each subject (manually flagged by Marijn van Vliet)
# For some subjects, the signal is really "drifty", but I decided not to flag
# these sensors, as there would be too many.
bads = {
    1: ['MEG2233', 'EEG001', 'EEG035', 'EEG015'],
    2: ['MEG1041', 'EEG001', 'EEG035', 'EEG002'],
    3: ['MEG2233', 'MEG0741', 'EEG045', 'EEG035', 'EEG001', 'EEG027'],
    4: ['MEG1842', 'MEG2113', 'MEG2342', 'MEG2233', 'MEG1942', 'MEG1922', 'EEG045', 'EEG001', 'EEG035'],
    5: ['MEG2233', 'MEG0811', 'MEG2342', 'MEG0812', 'MEG0813', 'MEG0722', 'MEG0632', 'MEG0913', 'MEG0912', 'EEG001', 'EEG035', 'EEG045'],
    6: ['EEG001', 'EEG035'],
    7: ['MEG0213', 'MEG2233', 'MEG2212', 'MEG2231', 'EEG001', 'EEG035', 'EEG045'],
}

# For these subjects, we need an extra ICA pass to get rid of all stimulation
# artifacts. Marijn van Vliet manually picked MEG sensors that showed a lot of
# the artifacts. This is the channel the ICA components will be compared
# against to detect components that capture the artifact.
subjects_with_extra_stim_artifacts = [4, 7]
stim_artifact_sensor = {
    4: 'MEG2631',
    7: 'MEG1721',
}

# Frequency range used in the DICS beamformer. The optimal frequencies where we
# find ERD/ERS effects varies a little between subjects.
freq_range = {
    1: (7, 15),
    2: (7, 11),
    4: (7, 15),
    5: (7, 15),
    6: (7, 15),
    7: (7, 15),
}

# Amount of regularization needed for the beamformers. Varies between subjects.
reg = {
    1: dict(lcmv=0.05, dics=0.05),
    2: dict(lcmv=0.05, dics=1.10),  # Crazy DICS regularization needed
    4: dict(lcmv=0.05, dics=0.05),
    5: dict(lcmv=0.05, dics=0.05),
    6: dict(lcmv=0.05, dics=0.05),
    7: dict(lcmv=0.05, dics=0.05),
}

# All filenames consumed and produced in this study
fname = FileNames()
fname.add('target_path', target_path)
fname.add('target_dir', '{target_path}/megset/sub{subject:02d}')
fname.add('subjects_dir', '{target_path}/megset/mri')
fname.add('subject_id', 'k{subject:d}_T1')
fname.add('raw', '{target_dir}/sub{subject:02d}-raw.fif')
fname.add('raw_tsss', '{target_dir}/sub{subject:02d}-tsss-raw.fif')
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

# Maxfilter related files
fname.add('mf_database', '/work/modules/Ubuntu/14.04/amd64/t314/neuromag/3.4.1/databases')
fname.add('mf_cal', '{mf_database}/sss/sss_cal.dat')
fname.add('mf_ct', '{mf_database}/ctc/ct_sparse.fif')
