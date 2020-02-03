import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

settings = config.dics_settings
settings_columns = ['reg', 'sensor_type', 'pick_ori', 'inversion',
                    'weight_norm', 'normalize_fwd', 'real_filter',
                    'use_noise_cov', 'reduce_rank', 'noise']
dics = pd.read_csv('dics_new_max_ori.csv', index_col=0)
dics['weight_norm'] = dics['weight_norm'].fillna('none')
dics['pick_ori'] = dics['pick_ori'].fillna('none')
dics['dist'] *= 1000  # Measure distance in mm

# Average across the various performance scores
dics = dics.groupby(settings_columns).agg('mean').reset_index()
del dics['vertex']  # No longer needed

assert len(dics) == len(settings)

###############################################################################
# Settings for plotting

# what to plot:
plot_type = 'foc'  # can be "corr" for correlation or "foc" for focality

# Colors for plotting
colors_5 = ['navy', 'orangered', 'crimson', 'firebrick', 'seagreen']
colors_6 = ['seagreen', 'yellowgreen', 'orangered', 'firebrick', 'navy',
            'cornflowerblue']

if plot_type == 'corr':
    y_label = 'Correlation'
    y_data = 'corr'
    title = 'Correlation as a function of localization error, noise=%s' % str(
        config.args.noise)
    ylims = (0.2, 1.1)
    loc = 'lower left'
    yticks = np.arange(0.4, 1.1, 0.2)
elif plot_type == 'foc':
    y_label = 'Focality measure'
    y_data = 'eval'
    title = 'Focality as a function of localization error, noise=%s' % str(
        config.args.noise)
    ylims = (-0.001, 0.041)
    xlims = (-1, 72)
    loc = 'upper right'
    yticks = np.arange(0.0, 0.041, 0.005)
    xticks = np.arange(0, 75, 5)
else:
    raise ValueError('Do not know plotting type "%s".' % plot_type)

###############################################################################
# Plot the different leadfield normalizations contrasted with each other

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)

x, y = dics.query('weight_norm=="unit-noise-gain" and inversion=="matrix" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_5[0], label='Weight normalization')

x, y = dics.query('weight_norm=="none" and normalize_fwd==True and reduce_rank==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_5[1], label='Lead field normalization, reduced rank')

x, y = dics.query('weight_norm=="none" and normalize_fwd==True and inversion=="single" and reduce_rank==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_5[2], label='Lead field normalization, single inversion')

x, y = dics.query('weight_norm=="none" and normalize_fwd==True and inversion=="matrix" and reduce_rank==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_5[3], label='Lead field normalization, full rank')

x, y = dics.query('weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_5[4], label='No normalization')

plt.legend(loc=loc)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
#plt.yticks(yticks)
#plt.yscale('log')
#plt.ylim(ylims)
#plt.xticks(xticks)
#plt.xlim(xlims)


###############################################################################
# Plot vector vs scalar beamformer considering normalization

plt.subplot(2, 2, 2)

x, y = dics.query('pick_ori=="none" and weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[0], label='No normalization, vector')

x, y = dics.query('pick_ori=="max-power" and weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[1], label='No normalization, scalar')

x, y = dics.query('pick_ori=="none" and weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[2], label='LF normalization, vector')

x, y = dics.query('pick_ori=="max-power" and weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[3], label='LF normalization, scalar')

x, y = dics.query('pick_ori=="none" and weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[4], label='Weight normalization, vector')

x, y = dics.query('pick_ori=="max-power" and weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[5], label='Weight normalization, scalar')

plt.legend(loc=loc)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
#plt.yticks(yticks)
#plt.yscale('log')
#plt.ylim(ylims)
#plt.xticks(xticks)
#plt.xlim(xlims)


###############################################################################
# Plot different normalizations with and without whitening

plt.subplot(2, 2, 3)

x, y = dics.query('weight_norm=="none" and normalize_fwd==False and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[0], label='No normalization')

x, y = dics.query('weight_norm=="none" and normalize_fwd==False and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[1], label='No normalization and whitening')

x, y = dics.query('weight_norm=="none" and normalize_fwd==True and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[2], label='LF normalization')

x, y = dics.query('weight_norm=="none" and normalize_fwd==True and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[3], label='LF normalization and whitening')

x, y = dics.query('weight_norm=="unit-noise-gain" and normalize_fwd==False and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[4], label='Weight normalization')

x, y = dics.query('weight_norm=="unit-noise-gain" and normalize_fwd==False and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[5], label='Weight normalization and whitening')

plt.legend(loc=loc)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
#plt.yticks(yticks)
#plt.yscale('log')
#plt.ylim(ylims)
#plt.xticks(xticks)
#plt.xlim(xlims)


###############################################################################
# Plot different sensor types

plt.subplot(2, 2, 4)

x, y = dics.query('sensor_type=="grad" and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[0], label='Gradiometers')

x, y = dics.query('sensor_type=="grad" and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[1], label='Gradiometers, with whitening')

x, y = dics.query('sensor_type=="mag" and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[2], label='Magnetometers')

x, y = dics.query('sensor_type=="mag" and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[3], label='Magnetometers, with whitening')

x, y = dics.query('sensor_type=="joint" and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[4], label='Joint grads+mags')

x, y = dics.query('sensor_type=="joint" and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors_6[5], label='Joint grads+mags, with whitening')


plt.legend(loc=loc)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
#plt.yticks(yticks)
#plt.yscale('log')
#plt.ylim(ylims)
#plt.xticks(xticks)
#plt.xlim(xlims)

plt.tight_layout()
