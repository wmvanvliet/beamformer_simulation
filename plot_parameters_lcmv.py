import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

settings = config.lcmv_settings
settings_columns = ['reg', 'sensor_type', 'pick_ori', 'inversion',
                    'weight_norm', 'normalize_fwd', 'use_noise_cov',
                    'reduce_rank', 'noise']
lcmv = pd.read_csv(config.fname.lcmv_params(noise=config.noise),
                   index_col=0)
lcmv['weight_norm'] = lcmv['weight_norm'].fillna('none')
lcmv['pick_ori'] = lcmv['pick_ori'].fillna('none')
lcmv['dist'] *= 1000  # Measure distance in mm

# Average across the various performance scores
lcmv = lcmv.groupby(settings_columns).agg('mean').reset_index()
del lcmv['vertex']  # No longer needed

assert len(lcmv) == len(settings)

###############################################################################
# Settings for plotting

# what to plot:
# can be "corr" for correlation, "foc" for focality or "ori" for orientation
# error
plot_type = 'foc'

# Colors for plotting
colors_5 = ['navy', 'orangered', 'crimson', 'firebrick', 'seagreen']
colors_6 = ['seagreen', 'yellowgreen', 'orangered', 'firebrick', 'navy',
            'cornflowerblue']

if plot_type == 'corr':
    kwargs = dict(
        y_label='Correlation',
        y_data='corr',
        ylims=(0.0, 1.1),
        xlims=(-1, 72),
        loc='lower left',
        yticks=np.arange(0.0, 1.1, 0.2),
        xticks=np.arange(0, 75, 10),
        yscale='linear')
    title = f'LCMV Correlation: %s, noise={config.noise:.2f}'
elif plot_type == 'foc':
    kwargs = dict(
        y_label='Focality measure',
        y_data='focality',
        ylims=(0.0, 0.006),
        xlims=(-1, 72),
        loc='upper right',
        yticks=np.arange(0.0, 0.041, 0.005),
        xticks=np.arange(0, 75, 10),
        yscale='linear')
    title = f'LCMV Focality: %s, noise={config.noise:.2f}'
elif plot_type == 'ori':
    kwargs = dict(
        y_label='Orienatation error',
        y_data='ori_error',
        ylims=(-5, 90.0),
        xlims=(-1, 72),
        loc='upper left',
        yticks=np.arange(0.0, 90.0, 10.0),
        xticks=np.arange(0, 75, 10),
        yscale='linear')
    title = f'LCMV Orientation error: %s, noise={config.noise:.2f}'
else:
    raise ValueError(f'Do not know plotting type "{plot_type}".')

###############################################################################
# Exchange the -1 with NaN for the orientation error case

if plot_type == 'ori':
    lcmv.loc[(lcmv['ori_error'] == -1), lcmv.columns[-1]] = np.nan
###############################################################################
# Plotting function for decluttering


def scatter_plot(options, colors, labels, title, y_data, loc, y_label, yticks,
                 yscale, ylims, xticks, xlims):
    plt.figure()

    for op, col, label in zip(options, colors, labels):
        x, y = lcmv.query(op)[['dist', y_data]].values.T
        plt.scatter(x, y, color=col, label=label)

    plt.legend(loc=loc)
    plt.title(title)
    plt.xlabel('Localization error [mm]')
    plt.ylabel(y_label)
    plt.yticks(yticks)
    plt.yscale(yscale)
    plt.ylim(ylims)
    plt.xticks(xticks)
    plt.xlim(xlims)

    plt.show()


###############################################################################
# Plot the different NORMALIZATIONS contrasted with each other

options = ['weight_norm=="unit-noise-gain" and normalize_fwd==False',
           'weight_norm=="none" and normalize_fwd==True',
           'weight_norm=="none" and normalize_fwd==False']
labels = ['Weight normalization', 'Lead field normalization',
          'No normalization']
colors = [config.cols['orchid'], config.cols['sky'], config.cols['spring']]
full_title = (title % 'Normalization')

scatter_plot(options, colors, labels, full_title, **kwargs)

###############################################################################
# Plot VECTOR vs SCALAR beamformer considering normalization

if plot_type != 'ori':
    options = ['pick_ori=="none"', 'pick_ori=="max-power"']
    labels = ['Vector beamformers', 'Scalar beamformers']
    colors = [config.cols['cherry'], config.cols['sky']]
    full_title = (title % 'Scalar and vector beamformers')

    scatter_plot(options, colors, labels, full_title, **kwargs)

###############################################################################
# Plot the impact of WHITENING

options = ['use_noise_cov==True', 'use_noise_cov==False']
labels = ['Whitening', 'No whitening']
colors = [config.cols['orchid'], config.cols['sea']]
full_title = (title % 'Whitening')

scatter_plot(options, colors, labels, full_title, **kwargs)

###############################################################################
# Plot different sensor types

options = ['sensor_type=="grad"', 'sensor_type=="mag"', 'sensor_type=="joint"']
labels = ['Gradiometers', 'Magnetometers', 'Joint grad. and mag.']
colors = [config.cols['cherry'], config.cols['purple'], config.cols['forest']]
full_title = (title % 'Sensor types')

scatter_plot(options, colors, labels, full_title, **kwargs)

###############################################################################
# Plot the impact of INVERSION SCHEMES

options = ['inversion=="single"', 'inversion=="matrix"']
labels = ['Single inversion', 'Matrix inversion']
colors = [config.cols['magician'], config.cols['forest']]
full_title = (title % 'Inversion method')

scatter_plot(options, colors, labels, full_title, **kwargs)

###############################################################################
# highlight the impact of RANK REDUCTION

options = ['reduce_rank==True', 'reduce_rank==False']
labels = ['Lead field rank reduction', 'No rank reduction']
colors = [config.cols['purple'], config.cols['spring']]
full_title = (title % 'Lead field rank redcution')

scatter_plot(options, colors, labels, full_title, **kwargs)
