import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

settings_columns = ['reg', 'sensor_type', 'pick_ori', 'inversion',
                    'weight_norm', 'normalize_fwd', 'use_noise_cov',
                    'reduce_rank', 'noise']
df = pd.read_csv('lcmv.csv', index_col=0)
df['weight_norm'] = df['weight_norm'].fillna('none')
df['pick_ori'] = df['pick_ori'].fillna('none')
df['dist'] *= 1000  # Measure distance in mm
df.loc[df['normalize_fwd'] == 'True', 'normalize_fwd'] = True
df.loc[df['normalize_fwd'] == 'False', 'normalize_fwd'] = False
df.loc[df['ori_error'] == -1, 'ori_error'] = np.nan

if 'real_filter' in df.columns:
    settings_columns.insert(6, 'real_filter')
    settings = config.dics_settings
else:
    settings_columns.append('project_pca')
    settings = config.lcmv_settings

# Average across the various performance scores
df = df.groupby(settings_columns).agg('mean').reset_index()

# No longer needed
del df['vertex']

assert len(df) == len(settings)

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
    title = f'Focality as a function of localization error, noise={config.noise:.2f}'
    ylims = (0, 0.04)
    xlims = (-1, 85)
    loc = 'upper right'
    yticks = np.arange(0.0, 0.014, 0.01)
    xticks = np.arange(0, 85, 5)
    yscale='linear'  # or 'log'
elif plot_type == 'ori_error':
    df = df.query('ori_error >= 0')
    y_label = 'Orientation error'
    y_data = 'ori_error'
    title = f'Orientation error as a function of localization error, noise={config.noise:.2f}'
    ylims = (-5, 90)
    xlims = (-1, 85)
    loc = 'upper right'
    yticks = np.arange(0.0, 90, 5)
    xticks = np.arange(0, 85, 5)
    yscale='linear'  # or 'log'
else:
    raise ValueError(f'Do not know plotting type "{plot_type}".')

###############################################################################
# Plot the different leadfield normalizations contrasted with each other

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)

x, y = df.query('weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[1], label='unit-noise-gain')

x, y = df.query('weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[0], label="Lead field normalization")

x, y = df.query('weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[4], label='No normalization')

x, y = df.query('weight_norm=="sqrtm"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors1[3], label='sqrtm')

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

x, y = df.query('pick_ori=="none" and weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='No normalization, vector')

x, y = df.query('pick_ori=="max-power" and weight_norm=="none" and normalize_fwd==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[1], label='No normalization, scalar')

x, y = df.query('pick_ori=="none" and weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='LF normalization, vector')

x, y = df.query('pick_ori=="max-power" and weight_norm=="none" and normalize_fwd==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[3], label='LF normalization, scalar')

x, y = df.query('pick_ori=="none" and weight_norm!="none"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='Weight normalization, vector')

x, y = df.query('pick_ori=="max-power" and weight_norm!="none"')[['dist', y_data]].values.T
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

x, y = df.query('use_noise_cov==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='No whitening')

x, y = df.query('use_noise_cov==True')[['dist', y_data]].values.T
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

x, y = df.query('sensor_type=="grad"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='Gradiometers')

x, y = df.query('sensor_type=="mag"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='Magnetometers')

x, y = df.query('sensor_type=="joint"')[['dist', y_data]].values.T
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

x, y = df.query('inversion=="matrix"')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='matrix inversion')

x, y = df.query('inversion=="single"')[['dist', y_data]].values.T
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
# Explore real vs complex filter
# plt.figure()
# 
# x, y = df.query('real_filter==True and normalize_fwd==True and weight_norm=="none"')[['dist', y_data]].values.T
# plt.scatter(x, y, color=colors2[0], label='real filter, normalized leadfield')
# 
# x, y = df.query('real_filter==False and normalize_fwd==True and weight_norm=="none"')[['dist', y_data]].values.T
# plt.scatter(x, y, color=colors2[1], label='complex filter, normalized leadfield')
# 
# x, y = df.query('real_filter==True and normalize_fwd==False and weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
# plt.scatter(x, y, color=colors2[2], label='real filter, unit-noise-gain')
# 
# x, y = df.query('real_filter==False and normalize_fwd==False and weight_norm=="unit-noise-gain"')[['dist', y_data]].values.T
# plt.scatter(x, y, color=colors2[3], label='complex filter, unit-noise-gain')
# 
# plt.legend(loc=loc)
# plt.title(title)
# plt.xlabel('Localization error [mm]')
# plt.ylabel(y_label)
# plt.yticks(yticks)
# plt.yscale(yscale)
# plt.ylim(ylims)
# plt.xticks(xticks)
# plt.xlim(xlims)
# 
# plt.show()

###############################################################################
# Explore PCA projection
plt.figure()

x, y = df.query('pick_ori=="vector" and weight_norm=="unit-noise-gain" and project_pca==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[0], label='unit-noise-gain')
x, y = df.query('pick_ori=="vector" and weight_norm=="unit-noise-gain" and project_pca==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[1], label='unit-noise-gain (PCA proj)')

x, y = df.query('pick_ori=="vector" and weight_norm=="none" and normalize_fwd==True and project_pca==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[2], label='leadfield norm')
x, y = df.query('pick_ori=="vector" and weight_norm=="none" and normalize_fwd==True and project_pca==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[3], label='leadfield norm (PCA proj)')

x, y = df.query('pick_ori=="vector" and weight_norm=="sqrtm" and project_pca==False')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[4], label='sqrtm')
x, y = df.query('pick_ori=="vector" and weight_norm=="sqrtm" and project_pca==True')[['dist', y_data]].values.T
plt.scatter(x, y, color=colors2[5], label='sqrtm (PCA proj)')

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
