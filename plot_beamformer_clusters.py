import config
from plotting_functions import get_plotting_specs, scatter_plot, read_data

###############################################################################
# Settings: what to plot

beamf_type = 'dics'  # can be lcmv or dics

# plot_type can be "corr" for correlation, "foc" for focality or "ori" for
# orientation error
plot_type = 'foc'

###############################################################################
# Read in the data and plotting settings
select_vertices = 'shallow'  # 'deep', 'shallow', or None
data = read_data(beamf_type, plot_type, select_vertices)
title, kwargs = get_plotting_specs(beamf_type, plot_type, select_vertices)

###############################################################################
# WEIGHT vs. LEAD FIELD vs. NO NORMALIZATION

options = ['weight_norm=="unit-noise-gain" and normalize_fwd==False',
           'weight_norm=="none" and normalize_fwd==True',
           'weight_norm=="none" and normalize_fwd==False']

labels = ['Weight normalization',
          'Lead field normalization',
          'No normalization']

colors = [config.cols['forest'], config.cols['magician'],
          config.cols['sky']]
full_title = 'Normalization comparison'

scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# WEIGHT NORMALIZATION

base = 'weight_norm=="unit-noise-gain" and normalize_fwd==False and %s'

if beamf_type == 'lcmv':
    options = [base % 'inversion=="matrix" and reduce_rank==False',
               base % 'inversion=="matrix" and reduce_rank==True',
               base % 'inversion=="single"',
               base % 'inversion=="matrix" and reduce_rank==False and \
               pick_ori=="none"']
    labels = ['matrix inversion + no rank reduction',
              'matrix inversion + reduce rank',
              'single inversion',
              'matrix inversion + no rank reduction + vector']
elif beamf_type == 'dics':
    options = [base % 'real_filter==True',
               base % 'real_filter==False and use_noise_cov==True and \
               pick_ori=="max-power"',
               base % 'real_filter==False and use_noise_cov==True and \
               pick_ori=="none"',
               base % 'real_filter==False and use_noise_cov==False']
    labels = ['real Filter',
              'complex filter + whitening + scalar',
              'complex filter + whitening + vector',
              'complex filter + no whitening']

colors = [config.cols['forest'], config.cols['magician'],
          config.cols['sky'], config.cols['spring']]
full_title = 'Weight Normalization'

scatter_plot(data, options, colors, labels, full_title, **kwargs)

# ###############################################################################
# LEAD FIELD NORMALIZATION

base = 'weight_norm=="none" and normalize_fwd==True and %s'

if beamf_type == 'lcmv':
    options = [base % 'inversion=="single"',
               base % 'inversion=="matrix" and reduce_rank==True and \
               pick_ori=="max-power"',
               base % 'inversion=="matrix" and reduce_rank==True and \
               pick_ori=="none"',
               base % 'inversion=="matrix" and reduce_rank==False']
    labels = ['single inversion',
              'matrix inversion + reduce rank + scalar',
              'matrix inversion + reduce rank + vector',
              'matrix inversion + no rank reduction']
elif beamf_type == 'dics':
    options = [base % 'inversion=="single" and pick_ori=="max-power"',
               base % 'inversion=="single" and pick_ori=="none"',
               base % 'inversion=="matrix" and reduce_rank==True and \
               use_noise_cov==True',
               base % 'inversion=="matrix" and reduce_rank==True and \
               use_noise_cov==False',
               base % 'inversion=="matrix" and reduce_rank==False']
    labels = ['single inversion + scalar',
              'single inversion + vector',
              'matrix inversion + reduced rank + whitening',
              'matrix inversion + reduced rank + no whitening',
              'matrix inversion + no rank reduction']

colors = [config.cols['orchid'], config.cols['magician'],
          config.cols['forest'], config.cols['sky'],
          config.cols['cherry']]
full_title = 'Lead field normalization'

scatter_plot(data, options, colors, labels, full_title, **kwargs)

# ###############################################################################
# # NO NORMALIZATION

base = 'weight_norm=="none" and normalize_fwd==False and %s'

if beamf_type == 'lcmv':
    options = [base % 'inversion=="single"',
               base % 'inversion=="matrix" and reduce_rank==False',
               base % 'inversion=="matrix" and reduce_rank==True and \
               pick_ori=="none"',
               base % 'inversion=="matrix" and reduce_rank==True and \
               pick_ori=="max-power"']
    labels = ['single inversion',
              'matrix inversion + rank reduction',
              'matrix inversion + reduce rank + vector',
              'matrix inversion + reduce rank + scalar']
elif beamf_type == 'dics':
    options = [base % 'inversion=="single" and pick_ori=="none"',
               base % 'inversion=="single" and pick_ori=="max-power"',
               base % 'inversion=="matrix" and reduce_rank==False',
               base % 'inversion=="matrix" and reduce_rank==True']
    labels = ['single inversion + vector',
              'single inversion + scalar',
              'matrix inversion + rank reduction',
              'matrix inversion + reduce rank']

colors = [config.cols['sky'], config.cols['magician'],
          config.cols['forest'], config.cols['orchid'],
          config.cols['spring']]
full_title = 'No normalization'

scatter_plot(data, options, colors, labels, full_title, **kwargs)
