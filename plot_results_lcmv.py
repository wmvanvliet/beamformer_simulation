import config
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bar
from config import (regs, sensor_types, pick_oris, inversions, weight_norms,
                    normalize_fwds, use_noise_covs, reduce_ranks)

settings = config.lcmv_settings
# some settings actually do not exist:
settings = [row for row in settings if not (
    row[3] == 'single' and row[-1] is True)]

dfs = []
for vertex in tqdm(range(3756), total=3756):
    try:
        df = pd.read_csv(config.fname.lcmv_results(noise=config.noise,
                                                   vertex=vertex))
        df['vertex'] = vertex
        df['noise'] = config.noise
        dfs.append(df)
    except Exception as e:
        print(e)

lcmv = pd.concat(dfs, ignore_index=True)
lcmv['pick_ori'].fillna('none', inplace=True)
lcmv['weight_norm'].fillna('none', inplace=True)

p2p_avg = []
fancy_avg = []
corr_avg = []
for setting in settings:
    # construct query
    setting = tuple(['none' if s is None else s for s in setting])

    qu = ("reg==%.2f and sensor_type=='%s' and pick_ori=='%s' and "
          "inversion=='%s' and weight_norm=='%s' and normalize_fwd==%s and "
          "use_noise_cov==%s and reduce_rank==%s") % setting

    sel = lcmv.query(qu).dropna()

    if len(sel) < 1000:
        import ipdb
        ipdb.set_trace()
        print(qu)  # only print settings if loading failed.
        print('Not enough voxels. Did this run fail?')
        continue

    # Create dist stc from simulated data
    vert_sel = sel['vertex'].to_numpy()
    p2p_avg.append(sel['dist'].to_numpy().mean())
    fancy_avg.append(sel['eval'].to_numpy().mean())
    corr_avg.append(sel['corr'].to_numpy().mean())

# check if the loading went alright
if not len(p2p_avg) == len(settings) or np.isnan(p2p_avg).any():
    raise ValueError('Something presumably went wrong when loading the data '
                     'for p2p_avg.')
elif not len(fancy_avg) == len(settings) or np.isnan(fancy_avg).any():
    raise ValueError('Something presumably went wrong when loading the data '
                     'for fancy_avg')
elif not len(corr_avg) == len(settings) or np.isnan(corr_avg).any():
    raise ValueError('Something presumably went wrong when loading the data '
                     'for corr_avg.')


###############################################################################
# Settings for plotting

# what to plot:
plot_type = 'foc'  # can be "corr" for correlation or "foc" for focality

# Colors for plotting
colors_5 = ['navy', 'orangered', 'crimson', 'firebrick', 'seagreen']
colors_6 = ['seagreen', 'yellowgreen', 'orangered', 'crimson', 'navy',
            'cornflowerblue']

if plot_type == 'corr':
    y_label = 'Correlation'
    y_data = copy.copy(corr_avg)
    title = 'Correlation as a function of localization error, noise=%s' % str(
        config.args.noise)
    ylims = (0.2, 1.1)
    loc = 'lower left'
    yticks = np.arange(0.4, 1.1, 0.2)
elif plot_type == 'foc':
    y_label = 'Focality measure'
    y_data = copy.copy(fancy_avg)
    title = 'Focality as a function of localization error, noise=%s' % str(
        config.args.noise)
    ylims = (-0.001, 0.007)
    loc = 'upper right'
    yticks = np.arange(0.0, 0.007, 0.002)
else:
    raise ValueError('Do not know plotting type "%s".' % plot_type)

###############################################################################
# Plot the different leadfield normalizations contrasted with each other

w_norm = []
lf_norm_rankred = []
lf_norm_single = []
lf_norm = []
no_norm = []
for ii, setting in enumerate(settings):
    if setting[4] == 'unit-noise-gain':  # weight normalization
        w_norm.append(ii)
    elif setting[5] is True and setting[4] is None:  # leadfield norm
        # leadfield normalization cancels if also weight normalization
        if setting[7] is False and setting[3] == 'single':  # single inversion
            lf_norm_single.append(ii)
        elif setting[7] is False and setting[3] == 'matrix':  # no rank reduct.
            lf_norm.append(ii)
        elif setting[7] is True:
            lf_norm_rankred.append(ii)  # rank reduction
    elif setting[4] is None and setting[5] is False:  # no normalization at all
        no_norm.append(ii)

# Plotting
labels = ['Weight normalization',
          'Lead field normalization, reduced rank',
          'Lead field normalization, single inversion',
          'Lead field normalization, full rank',
          'No normalization']

for idx, color, label in zip([w_norm, lf_norm_rankred, lf_norm_single,
                              lf_norm, no_norm],
                             colors_5, labels):
    plt.scatter(np.array(p2p_avg)[idx] * 1000, np.array(y_data)[idx],
                color=color, label=label)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
plt.ylim(ylims)
plt.legend(loc=loc)
plt.yticks(yticks, [str(round(x, 3)) for x in yticks])
plt.show()

###############################################################################
# Plot vector vs scalar beamformer considering normalization

no_norm_vec = []
lf_norm_vec = []
w_norm_vec = []
w_norm_scalar = []
lf_norm_scalar = []
no_norm_scalar = []
for ii, setting in enumerate(settings):
    if setting[2] == 'max-power':  # scalar beamformers
        if setting[4] == 'unit-noise-gain':  # weight norm
            w_norm_scalar.append(ii)
        elif setting[4] is None and setting[5] is True:  # lf norm
            lf_norm_scalar.append(ii)
        elif setting[4] is None and setting[5] is False:  # no norm
            no_norm_scalar.append(ii)
    else:  # vector beamformers
        if setting[4] == 'unit-noise-gain':  # weight norm
            w_norm_vec.append(ii)
        elif setting[4] is None and setting[5] is True:  # lf norm
            lf_norm_vec.append(ii)
        elif setting[4] is None and setting[5] is False:  # no norm
            no_norm_vec.append(ii)

# Plotting
labels = ['No normalization, vector',
          'No normalization, scalar',
          'LF normalization, vector',
          'LF normalization, scalar',
          'Weight normalization, vector',
          'Weight normalization, scalar']
for idx, color, label in zip([no_norm_vec, no_norm_scalar, lf_norm_vec,
                              lf_norm_scalar, w_norm_vec, w_norm_scalar],
                             colors_6, labels):
    plt.scatter(np.array(p2p_avg)[idx] * 1000, np.array(y_data)[idx],
                color=color, label=label)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
plt.ylim(ylims)
plt.legend(loc=loc)
plt.yticks(yticks, [str(round(x, 3)) for x in yticks])
plt.show()


###############################################################################
# Plot different normalizations with and without whitening

no_norm = []
lf_norm = []
w_norm = []
no_norm_whiten = []
lf_norm_whiten = []
w_norm_whiten = []
for ii, setting in enumerate(settings):
    if setting[6] is True:  # whitening
        if setting[4] == 'unit-noise-gain':  # weight norm
            w_norm_whiten.append(ii)
        elif setting[4] is None and setting[5] is True:  # leadfield norm
            lf_norm_whiten.append(ii)
        elif setting[4] is None and setting[5] is False:  # no normalization
            no_norm_whiten.append(ii)
    else:  # no whitening
        if setting[4] == 'unit-noise-gain':  # weight norm
            w_norm.append(ii)
        elif setting[4] is None and setting[5] is True:  # leadfield norm
            lf_norm.append(ii)
        elif setting[4] is None and setting[5] is False:  # no normalization
            no_norm.append(ii)

# Plotting
labels = ['No normalization',
          'No normalization and whitening',
          'LF normalization',
          'LF normalization and whitening',
          'Weight normalization',
          'Weight normalization and whitening']
for idx, color, label in zip([no_norm, no_norm_whiten, lf_norm, lf_norm_whiten,
                              w_norm, w_norm_whiten], colors_6, labels):
    plt.scatter(np.array(p2p_avg)[idx] * 1000, np.array(fancy_avg)[idx],
                color=color, label=label)
plt.title(title)
plt.xlabel('Localization error [mm]')
plt.ylabel(y_label)
plt.ylim(ylims)
plt.legend(loc=loc)
plt.yticks(yticks, [str(round(x, 3)) for x in yticks])
plt.show()
