import config
from plotting_functions import get_plotting_specs, scatter_plot, read_data

###############################################################################
# Settings: what to plot

beamf_type = 'lcmv'  # can be lcmv or dics

# plot_type can be "corr" for correlation, "foc" for focality or "ori" for
# orientation error
plot_type = 'ori'

###############################################################################
# Read in the data and plot setttings

data = read_data(beamf_type, plot_type)
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
