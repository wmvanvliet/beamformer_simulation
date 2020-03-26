import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from somato.config import fname

settings = config.lcmv_settings
settings_columns = ['reg', 'sensor_type', 'pick_ori', 'inversion',
                    'weight_norm', 'normalize_fwd', 'use_noise_cov',
                    'reduce_rank']
lcmv = pd.read_csv(fname.dip_vs_lcmv_results, index_col=0)
lcmv['weight_norm'] = lcmv['weight_norm'].fillna('none')
lcmv['pick_ori'] = lcmv['pick_ori'].fillna('none')
lcmv['dist'] *= 1000  # Measure distance in mm

assert len(lcmv) == len(settings)

###############################################################################
# Settings for plotting

# what to plot:
plot_type = 'foc'  # can be "ori_error" for orientation error or "foc" for focality
lcmv = lcmv.query('ori_error >= 0')

# Colors for plotting
colors1 = ['navy', 'orangered', 'crimson', 'firebrick', 'seagreen']
colors2 = ['seagreen', 'yellowgreen', 'orangered', 'firebrick', 'navy', 'cornflowerblue']

if plot_type == 'foc':
    y_label = 'Focality measure'
    y_data = 'focs'
    title = f'Focality as a function of localization error'
    ylims = (-0.001, 0.02)
    xlims = (-1, 72)
    loc = 'upper right'
    yticks = np.arange(0.0, ylims[1], 0.001)
    xticks = np.arange(0, xlims[1], 5)
    yscale = 'linear'  # or 'log'
elif plot_type == 'ori_error':
    y_label = 'Orientation error'
    y_data = 'ori_error'
    title = f'Orientation error as a function of localization error'
    ylims = (-5, 180)
    xlims = (-1, 72)
    loc = 'upper right'
    yticks = np.arange(0.0, ylims[1], 5)
    xticks = np.arange(0, xlims[1], 5)
    yscale = 'linear'  # or 'log'
else:
    raise ValueError(f'Do not know plotting type "{plot_type}".')

###############################################################################
# Adjust font sizes in plots

xsmall_font = 8
small_font = 10
medium_font = 12
big_font = 14

plt.rcParams.update({'font.size': small_font,  # controls default text sizes
                     'axes.titlesize': medium_font,  # fontsize of the subplot titles
                     'axes.labelsize': small_font,  # fontsize of axis labels
                     'xtick.labelsize': xsmall_font,  # fontsize of the x axis ticks
                     'ytick.labelsize': xsmall_font,  # fontsize of the y axis ticks
                     'legend.fontsize': xsmall_font,  # legend fontsize
                     'figure.titlesize': big_font})  # fontsize of the figure title

###############################################################################
# Plot the different leadfield normalizations contrasted with each other

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)

x, y = lcmv.query('weight_norm=="unit-noise-gain" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[0], label='Weight normalization')

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==True and reduce_rank=="leadfield"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[1], label="Lead field normalization, reduce_rank='leadfield'")

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==True and reduce_rank=="denominator"')[
    ['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[2], label="Lead field normalization, reduce_rank='denominator'")

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==True and reduce_rank=="False"')[
    ['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[3], label='Lead field normalization, full rank')

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==True and inversion=="matrix" and reduce_rank=="False"')[
    ['dist', y_data]].values.T

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[4], label='No normalization')

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

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==False and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='No normalization')

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==False and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[1], label='No normalization and whitening')

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==True and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='LF normalization')

x, y = lcmv.query('weight_norm=="none" and normalize_fwd==True and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[3], label='LF normalization and whitening')

x, y = lcmv.query('weight_norm=="unit-noise-gain" and normalize_fwd==False and use_noise_cov==False')[
    ['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='Weight normalization')

x, y = lcmv.query('weight_norm=="unit-noise-gain" and normalize_fwd==False and use_noise_cov==True')[
    ['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[5], label='Weight normalization and whitening')

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

x, y = lcmv.query('sensor_type=="grad" and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='Gradiometers')

x, y = lcmv.query('sensor_type=="grad" and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[1], label='Gradiometers, with whitening')

x, y = lcmv.query('sensor_type=="mag" and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='Magnetometers')

x, y = lcmv.query('sensor_type=="mag" and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[3], label='Magnetometers, with whitening')

x, y = lcmv.query('sensor_type=="joint" and use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='Joint grads+mags')

x, y = lcmv.query('sensor_type=="joint" and use_noise_cov==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[5], label='Joint grads+mags, with whitening')

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
# Different values for reduce_rank
plt.figure()

x, y = lcmv.query('reduce_rank=="False"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='reduce_rank=False')

x, y = lcmv.query('reduce_rank=="denominator"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='reduce_rank="denominator"')

x, y = lcmv.query('reduce_rank=="leadfield"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='reduce_rank="leadfield"')

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
# Reproduce Britta's plot
plt.figure()

x, y = lcmv.query('pick_ori=="none" and weight_norm=="unit-noise-gain" and normalize_fwd==True')[
    ['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='no pick ori, weight norm, fwd norm true')

x, y = lcmv.query('pick_ori=="none" and weight_norm=="unit-noise-gain" and normalize_fwd==False')[
    ['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='no pick ori, weight norm, fwd norm false')

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
