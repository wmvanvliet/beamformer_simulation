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

data = read_data(beamf_type, plot_type)
title, kwargs = get_plotting_specs(beamf_type, plot_type)

###############################################################################
# WEIGHT NORMALIZATION

base = 'weight_norm=="unit-noise-gain" and normalize_fwd==False and %s'
options = [base % 'inversion=="matrix" and reduce_rank==False',
           base % 'inversion=="matrix" and reduce_rank==True',
           base % 'inversion=="single"',
           base % 'inversion=="matrix" and reduce_rank==False and \
           pick_ori=="none"']

labels = ['matrix inversion + no rank reduction',
          'matrix inversion + reduce rank',
          'single inversion',
          'matrix inversion + no rank reduction + vector']
colors = [config.cols['forest'], config.cols['magician'],
          config.cols['orchid'],
          config.cols['sea']]
full_title = ('Weight Normalization')

scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# LEAD FIELD NORMALIZATION

base = 'weight_norm=="none" and normalize_fwd==True and %s'
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
colors = [config.cols['orchid'], config.cols['magician'],
          config.cols['forest'], config.cols['sky'], config.cols['cherry']]
full_title = 'Lead field normalization'

scatter_plot(data, options, colors, labels, full_title, **kwargs)

###############################################################################
# NO NORMALIZATION

base = 'weight_norm=="none" and normalize_fwd==False and %s'
options = [base % 'inversion=="single"',
           base % 'inversion=="matrix" and reduce_rank==False',
           base % 'inversion=="matrix" and reduce_rank==True and \
           pick_ori=="none"',
           base % 'inversion=="matrix" and reduce_rank==True and \
           pick_ori=="max-power"']
colors = [config.cols['sky'], config.cols['magician'],
          config.cols['forest'], config.cols['orchid']]
full_title = 'No normalization'

scatter_plot(data, options, colors, labels, full_title, **kwargs)
