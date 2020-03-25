import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

settings = config.dics_settings
settings_columns = ['reg', 'sensor_type', 'pick_ori', 'inversion',
                    'weight_norm', 'normalize_fwd', 'real_filter',
                    'use_noise_cov', 'reduce_rank']

dfs = []
for subject in [1, 4, 5, 6, 7]:
    df = pd.read_csv(config.fname.dics_megset_results(subject=subject), index_col=0)
    df['weight_norm'] = df['weight_norm'].fillna('none')
    df['pick_ori'] = df['pick_ori'].fillna('none')
    df['dist'] *= 1000  # Measure distance in mm
    df['focality'] = abs(df['focality'])
    df['subject'] = subject
    dfs.append(df)
dics = pd.concat(dfs, ignore_index=True)

# Average across the subjects
dics = dics.groupby(settings_columns).agg('mean').reset_index()

assert len(dics) == len(settings)

###############################################################################
# Settings for plotting

# what to plot:
plot_type = 'foc'  # can be "foc" for focality or "ori_error" for orientation error

# Colors for plotting
colors1 = ['navy', 'orangered', 'crimson', 'firebrick', 'seagreen']
colors2 = ['seagreen', 'yellowgreen', 'orangered', 'firebrick', 'navy', 'cornflowerblue']

if plot_type == 'foc':
    y_label = 'Focality measure'
    y_data = 'focality'
    title = f'Focality as a function of localization error'
    ylims = (0.00025, 0.2)
    xlims = (-1, 96)
    loc = 'upper right'
    yticks = np.arange(0.0, 0.06, 0.005)
    xticks = np.arange(0, 96, 5)
    yscale='log'  # or 'log'
elif plot_type == 'ori_error':
    dics = dics.query('ori_error >= 0')
    y_label = 'Orientation error'
    y_data = 'ori_error'
    title = f'Orientation error as a function of localization error'
    ylims = (-5, 90)
    xlims = (-1, 96)
    loc = 'upper right'
    yticks = np.arange(0.0, 90, 5)
    xticks = np.arange(0, 96, 5)
    yscale='linear'  # or 'log'
else:
    raise ValueError(f'Do not know plotting type "{plot_type}".')

###############################################################################
# Plot the different leadfield normalizations contrasted with each other

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)

x, y = dics.query('weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[0], label='Weight normalization')

x, y = dics.query('weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[4], label="Lead field normalization")

x, y = dics.query('weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[3], label='No normalization')

plt.legend(loc=loc)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
plt.yticks(yticks)
plt.yscale(yscale)
plt.ylim(ylims)
plt.xticks(xticks)
plt.xlim(xlims)


###############################################################################
# Plot vector vs scalar beamformer considering normalization

plt.subplot(2, 2, 2)

x, y = dics.query('pick_ori=="none" and weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='No normalization, vector')

x, y = dics.query('pick_ori=="max-power" and weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[1], label='No normalization, scalar')

x, y = dics.query('pick_ori=="none" and weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='LF normalization, vector')

x, y = dics.query('pick_ori=="max-power" and weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[3], label='LF normalization, scalar')

x, y = dics.query('pick_ori=="none" and weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='Weight normalization, vector')

x, y = dics.query('pick_ori=="max-power" and weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[5], label='Weight normalization, scalar')

plt.legend(loc=loc)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
plt.yticks(yticks)
plt.yscale(yscale)
plt.ylim(ylims)
plt.xticks(xticks)
plt.xlim(xlims)


###############################################################################
# Plot different normalizations with and without whitening

plt.subplot(2, 2, 3)

x, y = dics.query('use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='No whitening')

x, y = dics.query('use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[1], label='Whitening')

plt.legend(loc=loc)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
plt.yticks(yticks)
plt.yscale(yscale)
plt.ylim(ylims)
plt.xticks(xticks)
plt.xlim(xlims)


###############################################################################
# Plot different sensor types

plt.subplot(2, 2, 4)

x, y = dics.query('sensor_type=="grad"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='Gradiometers')

x, y = dics.query('sensor_type=="mag"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='Magnetometers')

x, y = dics.query('sensor_type=="joint"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='Joint grads+mags')

plt.legend(loc=loc)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
plt.yticks(yticks)
plt.yscale(yscale)
plt.ylim(ylims)
plt.xticks(xticks)
plt.xlim(xlims)

plt.tight_layout()


###############################################################################
# Explore reduce_rank
plt.figure()

x, y = dics.query('reduce_rank==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='no rank reduction')

x, y = dics.query('reduce_rank==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='with rank reduction')

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
# Explore regularization
plt.figure()

x, y = dics.query('reg==0')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='reg=0')
x, y = dics.query('reg==0.05')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='reg=0.05')
x, y = dics.query('reg==0.1')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='reg=0.1')


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
