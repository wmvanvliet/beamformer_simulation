import mne
import argparse
from config import fname, bads

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

report = mne.open_report(fname.report(subject=subject))

raw = mne.io.read_raw_fif(fname.raw(subject=subject), preload=True)
raw.info['bads'] = bads[subject]
raw.pick_types(meg=True, stim=True, eog=True)
raw.set_annotations(mne.read_annotations(fname.annotations(subject=subject)))
report.add_figs_to_section(raw.plot_psd(tmax=600), 'PSD of unfiltered raw', 'Raw', replace=True)

raw = mne.chpi.filter_chpi(raw, include_line=True)
report.add_figs_to_section(raw.plot_psd(tmax=600), 'PSD of notch filtered raw', 'Raw', replace=True)
raw.save(fname.raw_filt(subject=subject), overwrite=True)

# Make version for ICA use
raw = raw.filter(1, None)
raw = raw.resample(128)
raw.save(fname.raw_detrend(subject=subject), overwrite=True)
report.save(fname.report(subject=subject), overwrite=True, open_browser=False)
report.save(fname.report_html(subject=subject), overwrite=True, open_browser=False)
