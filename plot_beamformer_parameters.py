import numpy as np
import pandas as pd

import config
from plotting_functions import get_plotting_specs, scatter_plot

###############################################################################
# Settings: what to plot

beamf_type = 'lcmv'  # can be lcmv or dics

# plot_type can be "corr" for correlation, "foc" for focality or "ori" for
# orientation error
plot_type = 'ori'

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
# Plot the different NORMALIZATIONS contrasted with each other

options = ['weight_norm=="unit-noise-gain" and normalize_fwd==False',
           'weight_norm=="none" and normalize_fwd==True',
           'weight_norm=="none" and normalize_fwd==False']
labels = ['Weight normalization', 'Lead field normalization',
          'No normalization']
colors = [config.cols['orchid'], config.cols['sky'], config.cols['spring']]
full_title = (title % 'Normalization')

scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# Plot VECTOR vs SCALAR beamformer considering normalization

if plot_type != 'ori':
    options = ['pick_ori=="none"', 'pick_ori=="max-power"']
    labels = ['Vector beamformers', 'Scalar beamformers']
    colors = [config.cols['cherry'], config.cols['sky']]
    full_title = (title % 'Scalar and vector beamformers')

    scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# Plot the impact of WHITENING

options = ['use_noise_cov==True', 'use_noise_cov==False']
labels = ['Whitening', 'No whitening']
colors = [config.cols['orchid'], config.cols['sea']]
full_title = (title % 'Whitening')

scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# Plot different sensor types

options = ['sensor_type=="grad"', 'sensor_type=="mag"', 'sensor_type=="joint"']
labels = ['Gradiometers', 'Magnetometers', 'Joint grad. and mag.']
colors = [config.cols['cherry'], config.cols['purple'], config.cols['forest']]
full_title = (title % 'Sensor types')

scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# Plot the impact of INVERSION SCHEMES

options = ['inversion=="single"', 'inversion=="matrix"']
labels = ['Single inversion', 'Matrix inversion']
colors = [config.cols['magician'], config.cols['forest']]
full_title = (title % 'Inversion method')

scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# highlight the impact of RANK REDUCTION

options = ['reduce_rank==True', 'reduce_rank==False']
labels = ['Lead field rank reduction', 'No rank reduction']
colors = [config.cols['purple'], config.cols['spring']]
full_title = (title % 'Lead field rank reduction')

scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# For DICS: plot FILTER TYPE

if beamf_type == 'dics':
    options = ['real_filter==True', 'real_filter==False']
    labels = ['Real filter', 'Complex filter']
    colors = [config.cols['magician'], config.cols['spring']]
    full_title = (title % 'Filter type')

    scatter_plot(data, options, colors, labels, full_title, **kwargs)
