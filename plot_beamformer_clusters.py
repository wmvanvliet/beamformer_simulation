import config
from plotting_functions import get_plotting_specs, scatter_plot, read_data

###############################################################################
# Settings: what to plot

beamf_type = 'lcmv'  # can be lcmv or dics

# plot_type can be "corr" for correlation, "foc" for focality or "ori" for
# orientation error
plot_type = 'foc'

###############################################################################
# Read in the data and plotting settings

data = read_data(beamf_type, plot_type)
title, kwargs = get_plotting_specs(beamf_type, plot_type)

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

scatter_plot(data, options, colors, labels, full_title, **kwargs)

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

scatter_plot(data, options, colors, labels, full_title, **kwargs)
