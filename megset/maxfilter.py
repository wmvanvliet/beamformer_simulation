import argparse
import mne
from config import fname

# Handle command line arguments
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('subject', metavar='sub###', type=int, help='The subject to process')
args = parser.parse_args()
subject = args.subject
print('Processing subject:', subject)

raw = mne.io.read_raw_fif(fname.raw(subject=subject))
raw_tsss = mne.preprocessing.maxwell_filter(raw, st_duration=60,
                                            calibration=fname.mf_cal,
                                            cross_talk=fname.mf_ct)
raw.tsss.save(fname.raw_tsss(subject=subject))
