from config import fname

DOIT_CONFIG = dict(
    verbosity=2,
    sort='definition',
)

def task_forward():
    """Step 1: setup the forward solution"""
    return dict(
        file_dep=[fname.raw, '01_setup_forward_solution.py'],
        targets=[fname.trans, fname.src, fname.fwd],
        actions=['ipython 01_setup_forward_solution.py'],
    )

def task_preprocessing():
    """Step 2: perform preprocessing: ICA and epochs"""
    return dict(
        file_dep=[fname.raw, '02_preprocessing.py'],
        targets=[fname.epochs, fname.ica],
        actions=['ipython 02_preprocessing.py'],
    )

def task_mne():
    """Step 3: MNE source estimate"""
    return dict(
        file_dep=[fname.epochs, fname.fwd, '03_mne.py'],
        targets=[fname.stc_mne, fname.nii_mne],
        actions=['ipython 03_mne.py'],
    )

def task_lcmv():
    """Step 4: LCMV source estimate"""
    return dict(
        file_dep=[fname.epochs, fname.fwd, '04_lcmv.py'],
        targets=[fname.stc_lcmv, fname.nii_lcmv],
        actions=['ipython 04_lcmv.py'],
    )

def task_dics():
    """Step 5: DICS source estimate"""
    return dict(
        file_dep=[fname.epochs, fname.fwd, '05_dics.py'],
        targets=[fname.stc_dics, fname.nii_dics],
        actions=['ipython 05_dics.py'],
    )
