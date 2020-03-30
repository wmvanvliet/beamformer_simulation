import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def read_data(beamf_type, plot_type):
    """ Read and prepare data for plotting."""
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

    return data


def get_plotting_specs(beamf_type, plot_type):
    """Get all parameters and settings for plotting."""

    if plot_type not in ('corr', 'foc', 'ori'):
        raise ValueError('Do not know plotting type "%s".' % plot_type)
    if beamf_type == 'lcmv':
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
    elif beamf_type == 'dics':
        if plot_type == 'foc':
            kwargs = dict(
                y_label='Focality measure',
                y_data='focality',
                ylims=(0, 0.014),
                xlims=(-1, 85),
                loc='upper left',
                yticks=np.arange(0.0, 0.014, 0.01),
                xticks=np.arange(0, 85, 5),
                yscale='linear')
            title = f'DICS Focality: %s, noise={config.noise:.2f}'
        elif plot_type == 'ori':
            kwargs = dict(
                 y_label='Orientation error',
                 y_data='ori_error',
                 ylims=(-5, 90),
                 xlims=(-1, 85),
                 loc='upper left',
                 yticks=np.arange(0.0, 90, 5),
                 xticks=np.arange(0, 85, 10),
                 yscale='linear')
            title = f'DICS Orientation error: %s, noise={config.noise:.2f}'

    return title, kwargs


def scatter_plot(data, options, colors, labels, title, y_data, loc, y_label,
                 yticks, yscale, ylims, xticks, xlims):
    """Customized plotting function for scatter plots."""

    plt.figure()

    for op, col, label in zip(options, colors, labels):
        x, y = data.query(op)[['dist', y_data]].values.T
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
