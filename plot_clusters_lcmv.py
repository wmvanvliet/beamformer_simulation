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
plot_type = 'corr'

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

scatter_plot(options, colors, labels, full_title, **kwargs)

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

scatter_plot(options, colors, labels, full_title, **kwargs)
