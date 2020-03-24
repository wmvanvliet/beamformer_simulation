import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

settings = config.lcmv_settings
settings_columns = ['reg', 'sensor_type', 'pick_ori', 'inversion',
                    'weight_norm', 'normalize_fwd', 'use_noise_cov',
                    'reduce_rank', 'noise']
lcmv = pd.read_csv(config.fname.lcmv_megset_results(subject=1), index_col=0)
lcmv['weight_norm'] = lcmv['weight_norm'].fillna('none')
lcmv['pick_ori'] = lcmv['pick_ori'].fillna('none')
lcmv['dist'] *= 1000  # Measure distance in mm

assert len(lcmv) == len(settings)

###############################################################################
# Settings for plotting

# what to plot:
plot_type = 'foc'  # can be "foc" for focality or "ori_error" for orientation error

# Colors for plotting
colors1 = ['navy', 'orangered', 'crimson', 'firebrick', 'seagreen']
colors2 = ['seagreen', 'yellowgreen', 'orangered', 'firebrick', 'navy', 'cornflowerblue']

if plot_type == 'foc':
    y_label = 'Focality measure'
    y_data = 'focs'
    title = f'Focality as a function of localization error'
    ylims = (0, 0.003)
    xlims = (-1, 96)
    loc = 'upper right'
    yticks = np.arange(0.0, 0.041, 0.005)
    xticks = np.arange(0, 96, 5)
    yscale='linear'  # or 'log'
elif plot_type == 'ori_error':
    lcmv = lcmv.query('ori_error >= 0')
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

x, y = lcmv.query('weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[0], label='Weight normalization')

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[4], label="Lead field normalization")

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
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

x, y = lcmv.query('pick_ori=="none" and weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='No normalization, vector')

x, y = lcmv.query('pick_ori=="max-power" and weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[1], label='No normalization, scalar')

x, y = lcmv.query('pick_ori=="none" and weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='LF normalization, vector')

x, y = lcmv.query('pick_ori=="max-power" and weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[3], label='LF normalization, scalar')

x, y = lcmv.query('pick_ori=="none" and weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='Weight normalization, vector')

x, y = lcmv.query('pick_ori=="max-power" and weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
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

x, y = lcmv.query('use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='No whitening')

x, y = lcmv.query('use_noise_cov==True')[['dist', y_data]].values.T
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

x, y = lcmv.query('sensor_type=="grad"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='Gradiometers')

x, y = lcmv.query('sensor_type=="mag"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='Magnetometers')

x, y = lcmv.query('sensor_type=="joint"')[['dist', y_data]].values.T
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
# Explore inversion method
plt.figure()

x, y = lcmv.query('inversion=="matrix"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='matrix inversion')

x, y = lcmv.query('inversion=="single"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='single inversion')

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

x, y = lcmv.query('reg==0')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='reg=0')
x, y = lcmv.query('reg==0.05')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='reg=0.05')
x, y = lcmv.query('reg==0.1')[['dist', y_data]].values.T
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
