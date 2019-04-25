import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import mne
import surfer
from mayavi import mlab
from itertools import product

import config
from config import fname

fwd = mne.read_forward_solution(fname.fwd) 

dfs = []
for vertex in tqdm(range(3765), total=3765):
    try: 
        df = pd.read_csv(fname.lcmv_results(noise=config.noise, vertex=vertex))
        df['vertex'] = vertex
        df['noise'] = config.noise
        dfs.append(df)
    except Exception as e:
        print(e)
lcmv = pd.concat(dfs, ignore_index=True)
lcmv['pick_ori'].fillna('none', inplace=True)
lcmv['weight_norm'].fillna('none', inplace=True)
#lcmv = lcmv.set_index(['reg', 'sensor_type', 'pick_ori', 'weight_norm', 'noise', 'use_noise_cov', 'depth'])

regs = [0.05, 0.1, 0.5]
sensor_types = ['joint', 'grad', 'mag']
pick_oris = ['none', 'normal', 'max-power']
weight_norms = ['unit-noise-gain', 'none']
use_noise_covs = [True, False]
depths = [True, False]
settings = list(product(regs, sensor_types, pick_oris, weight_norms,
                        use_noise_covs, depths))


html_header = (
    '<html><head><link rel="stylesheet" type="text/css" href="style.css"></head><body>'
    '<table><tr>'
    '<th>reg</th>'
    '<th>sensor type</th>'
    '<th>pick_ori</th>'
    '<th>weight_norm</th>'
    '<th>use_noise_cov</th>'
    '<th>depth</th>'
    '<th>outside view</th>'
    '<th>inside view</th>'
    '</tr>')

html_footer = '</body></table>'

html_table = ''
for i, setting in enumerate(settings):
    q = ("reg==%.1f and sensor_type=='%s' and pick_ori=='%s' and "
         "weight_norm=='%s' and use_noise_cov==%s and depth==%s" % setting)
    sel = lcmv.query(q).dropna()

    reg, sensor_type, pick_ori, weight_norm, use_noise_cov, depth = setting

    # Skip some combinations
    if weight_norm == 'unit-noise-gain' and depth == True:
        continue
    if sensor_type == 'joint' and use_noise_cov == False:
        continue

    # Create the brain plots
    mlab.figure(i, size=(600, 500))
    vertices = fwd['src'][1]['vertno'][sel['vertex']]
    brain1 = surfer.Brain('sample', hemi='rh', surf='white', figure=i)
    #brain1.add_data(sel['eval'], vertices=vertices)
    brain1.add_data(sel['dist'], vertices=vertices, smoothing_steps=5)
    #brain1.scale_data_colormap(0, 0.001, 0.002, transparent=False)
    brain1.scale_data_colormap(0, 0.075, 0.15, transparent=False)
    mlab.view(0, 90, 250, [33, -10, 35])
    mlab.savefig('html/lcmv/%03d_out.png' % i)
    mlab.view(-180, 90, 300, [33, -10, 35])
    mlab.savefig('html/lcmv/%03d_in.png' % i)
    mlab.close(i)

    # Add row to the HTML table
    html_table += '<tr><td>' + '</td><td>'.join([str(s) for s in setting]) + '</td>'
    html_table += '<td><img src="lcmv/%03d_out.png"></td><td><img src="lcmv/%03d_in.png"></td>' % (i, i)

    with open('html/lcmv.html', 'w') as f:
        f.write(html_header)
        f.write(html_table)
        f.write(html_footer)
