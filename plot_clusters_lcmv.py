import numpy as np
import pandas as pd

import config
from plotting_functions import get_plotting_specs, scatter_plot

###############################################################################
# Settings: what to plot

beamf_type = 'lcmv'  # can be lcmv or dics

# plot_type can be "corr" for correlation, "foc" for focality or "ori" for
# orientation error
plot_type = 'foc'

###############################################################################
# Read in the data

if beamf_type == 'lcmv':
    settings = config.lcmv_settings
    settings_columns = ['reg', 'sensor_type', 'pick_ori', 'inversion',
                        'weight_norm', 'normalize_fwd', 'use_noise_cov',
                        'reduce_rank', 'noise']
    data_fname = config.fname.lcmv_params(noise=config.noise)
elif beamf_type == 'dics':
    settings = config.dics_settings
    settings_columns = ['reg', 'sensor_type', 'pick_ori', 'inversion',
                        'weight_norm', 'normalize_fwd', 'real_filter',
                        'use_noise_cov', 'reduce_rank', 'noise']
    data_fname = config.fname.dics_params(noise=config.noise)
else:
    raise ValueError('Unknown beamformer type %s.' % beamf_type)

data = pd.read_csv(data_fname, index_col=0)
data['weight_norm'] = data['weight_norm'].fillna('none')
data['pick_ori'] = data['pick_ori'].fillna('none')
data['dist'] *= 1000  # Measure distance in mm

# Average across the various performance scores
data = data.groupby(settings_columns).agg('mean').reset_index()
del data['vertex']  # No longer needed

assert len(data) == len(settings)

# Exchange the -1 with NaN for the orientation error case
if plot_type == 'ori':
    data.loc[(data['ori_error'] == -1), data.columns[-1]] = np.nan

# plot setttings
title, kwargs = get_plotting_specs(beamf_type, plot_type)

###############################################################################
# WEIGHT NORMALIZATION

options = ['weight_norm=="unit-noise-gain" and normalize_fwd==False and \
           inversion=="matrix" and reduce_rank==False',
           'weight_norm=="unit-noise-gain" and normalize_fwd==False and \
           inversion=="matrix" and reduce_rank==True',
           'weight_norm=="unit-noise-gain" and normalize_fwd==False and \
           inversion=="single"',
           'weight_norm=="unit-noise-gain" and normalize_fwd==False and \
           inversion=="matrix" and reduce_rank==False and \
           pick_ori=="none"']

labels = ['matrix and no rank red', 'matrix and rank red', 'single',
          'matrix and no rank red and vector']
colors = [config.cols['forest'], config.cols['magician'],
          config.cols['orchid'],
          config.cols['sea']]
full_title = ('Weight Normalization')

scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# LEAD FIELD NORMALIZATION

options = ['weight_norm=="none" and normalize_fwd==True and \
           inversion=="single"',
           'weight_norm=="none" and normalize_fwd==True and \
           inversion=="matrix" and reduce_rank==True and \
           pick_ori=="max-power"',
           'weight_norm=="none" and normalize_fwd==True and \
           inversion=="matrix" and reduce_rank==True and pick_ori=="none"',
           'weight_norm=="none" and normalize_fwd==True and \
           inversion=="matrix" and reduce_rank==False']
labels = ['single', 'matrix and reduce rank and scalar',
          'matrix and red rank and vec',
          'matrix and no rank red']
colors = [config.cols['orchid'], config.cols['magician'],
          config.cols['forest'], config.cols['sky'], config.cols['cherry']]
full_title = 'Lead field normalization'

scatter_plot(data, options, colors, labels, full_title, **kwargs)
