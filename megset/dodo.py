from config import fname, subjects

DOIT_CONFIG = dict(
    verbosity=2,
    sort='definition',
)

def task_forward():
    """Step 1: setup the forward solution"""
    for subject in subjects:
        yield dict(
            name=str(subject),
            file_dep=[fname.raw(subject=subject), fname.trans(subject=subject), '01_setup_forward_solution.py'],
            targets=[fname.src(subject=subject), fname.fwd(subject=subject)],
            actions=[f'ipython 01_setup_forward_solution.py {subject:d}'],
        )

def task_filter():
    """Step 2: frequency filtering"""
    for subject in subjects:
        yield dict(
            name=str(subject),
            file_dep=[fname.raw(subject=subject), '02_filter.py'],
            targets=[fname.raw_filt(subject=subject), fname.raw_detrend(subject=subject)],
            actions=[f'ipython 02_filter.py {subject:d}'],
        )

def task_ica():
    """Step 3: ICA"""
    for subject in subjects:
        yield dict(
            name=str(subject),
            file_dep=[fname.raw_detrend(subject=subject), '03_ica.py'],
            targets=[fname.ica(subject=subject)],
            actions=[f'ipython 03_ica.py {subject:d}'],
        )

def task_epochs():
    """Step 4: Construct epochs"""
    for subject in subjects:
        yield dict(
            name=str(subject),
            file_dep=[fname.raw_filt(subject=subject), fname.ica(subject=subject), '04_epochs.py'],
            targets=[fname.epochs(subject=subject), fname.epochs_long(subject=subject)],
            actions=[f'ipython 04_epochs.py {subject:d}'],
        )

def task_dipole():
    """Step 5: Dipole source estimate (golden standard)"""
    for subject in subjects:
        yield dict(
            name=str(subject),
            file_dep=[fname.epochs(subject=subject), fname.fwd(subject=subject), '05_dipole.py'],
            targets=[fname.ecd(subject=subject)],
            actions=[f'ipython 05_dipole.py {subject:d}'],
        )

def task_lcmv():
    """Step 6: LCMV source estimate"""
    for subject in subjects:
        yield dict(
            name=str(subject),
            file_dep=[fname.epochs(subject=subject), fname.fwd(subject=subject), fname.ecd(subject=subject), '06_lcmv.py'],
            targets=[fname.stc_lcmv(subject=subject), fname.nii_lcmv(subject=subject)],
            actions=[f'ipython 06_lcmv.py {subject:d}'],
        )

def task_dics():
    """Step 7: DICS source estimate"""
    for subject in subjects:
        yield dict(
            name=str(subject),
            file_dep=[fname.epochs_long(subject=subject), fname.fwd(subject=subject), fname.ecd(subject=subject), '07_dics.py'],
            targets=[fname.stc_dics(subject=subject), fname.nii_dics(subject=subject)],
            actions=[f'ipython 07_dics.py {subject:d}'],
        )

def task_mne():
    """Step 8: MNE source estimate"""
    for subject in subjects:
        yield dict(
            name=str(subject),
            file_dep=[fname.epochs(subject=subject), fname.fwd(subject=subject), fname.ecd(subject=subject), '08_mne.py'],
            targets=[fname.stc_mne(subject=subject), fname.nii_mne(subject=subject)],
            actions=[f'ipython 08_mne.py {subject:d}'],
        )
