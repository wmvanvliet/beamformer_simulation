import mne

from time_series import simulate_raw

import config
from config import fname

info = mne.io.read_info(fname.sample_raw)
info = mne.pick_info(info, mne.pick_types(info, meg=True, eeg=False))
fwd = mne.read_forward_solution(fname.fwd_true)
fwd = mne.pick_types_forward(fwd, meg=True, eeg=False)
src = fwd['src']
labels = mne.read_labels_from_annot(subject='sample', parc='aparc.a2009s')

er_raw = mne.io.read_raw_fif(fname.ernoise, preload=True)
fn_stc_signal = fname.stc_signal(noise=config.noise, vertex=config.vertex)
fn_simulated_raw = fname.simulated_raw(noise=config.noise, vertex=config.vertex)
fn_report_h5 = fname.report(noise=config.noise, vertex=config.vertex)

simulate_raw(info, src, fwd, config.vertex, config.signal_hemi,
             config.signal_freq, config.trial_length, config.n_trials,
             config.noise, config.random, labels, er_raw,
             fn_stc_signal=fn_stc_signal, fn_simulated_raw=fn_simulated_raw,
             fn_report_h5=fn_report_h5)