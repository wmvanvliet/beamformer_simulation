import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import mne
import surfer
from mayavi import mlab
mlab.options.offscreen = True
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

regs = [0.05, 0.1]
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
    '<th colspan="2">P2P distance</th>'
    '<th colspan="2">Fancy metric</th>'
    '</tr>')
html_header = '''
    <html>
    <head>
        <link rel="stylesheet" type="text/css" href="style.css">
        <script src="filter.js"></script>
    </head>
    <body>
    <table>
    <tr>
        <th>reg</th>
        <th>sensor type</th>
        <th>pick_ori</th>
        <th>weight_norm</th>
        <th>use_noise_cov</th>
        <th>depth</th>
        <th colspan="2">P2P distance</th>
        <th colspan="2">Fancy metric</th>
    </tr>
    <tr>
        <td><input type="text" onkeyup="filter(0, this)" placeholder="reg"></td>
        <td><input type="text" onkeyup="filter(1, this)" placeholder="sensor type"></td>
        <td><input type="text" onkeyup="filter(2, this)" placeholder="pick_ori"></td>
        <td><input type="text" onkeyup="filter(3, this)" placeholder="weight_norm"></td>
        <td><input type="text" onkeyup="filter(4, this)" placeholder="use_noise_doc"></td>
        <td><input type="text" onkeyup="filter(5, this)" placeholder="depth"></td>
        <td colspan="2"></td>
        <td colspan="2"></td>
    </tr>
'''

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
    if weight_norm == 'none' and depth == True:
        continue
    if sensor_type == 'joint' and use_noise_cov == False:
        continue

    # Add row to the HTML table
    html_table += '<tr><td>' + '</td><td>'.join([str(s) for s in setting]) + '</td>'

    try:
        if len(sel) < 1000:
            raise RuntimeError('not enough vertices')

        # Create the brain plots
        mlab.figure(1, size=(600, 500))
        vertices = fwd['src'][1]['vertno'][sel['vertex']]
        brain = surfer.Brain('sample', hemi='rh', surf='white', figure=1)
        brain.add_data(sel['dist'], vertices=vertices, smoothing_steps=5)
        brain.scale_data_colormap(0, 0.075, 0.15, transparent=False)
        mlab.view(0, 90, 250, [33, -10, 35])
        mlab.savefig('html/lcmv/%03d_dist_out.png' % i)
        mlab.view(-180, 90, 300, [33, -10, 35])
        mlab.savefig('html/lcmv/%03d_dist_in.png' % i)
        mlab.close(1)

        mlab.figure(1, size=(600, 500))
        vertices = fwd['src'][1]['vertno'][sel['vertex']]
        brain = surfer.Brain('sample', hemi='rh', surf='white', figure=1)
        brain.add_data(sel['eval'], vertices=vertices, smoothing_steps=5)
        brain.scale_data_colormap(0, 0.001, 0.002, transparent=False)
        mlab.view(0, 90, 250, [33, -10, 35])
        mlab.savefig('html/lcmv/%03d_eval_out.png' % i)
        mlab.view(-180, 90, 300, [33, -10, 35])
        mlab.savefig('html/lcmv/%03d_eval_in.png' % i)
        mlab.close(1)
        html_table += '<td><img src="lcmv/%03d_dist_out.png"></td><td><img src="lcmv/%03d_dist_in.png"></td>' % (i, i)
        html_table += '<td><img src="lcmv/%03d_eval_out.png"></td><td><img src="lcmv/%03d_eval_in.png"></td>' % (i, i)
    except Exception as e:
        html_table += '<td colspan="2">%s</td>' % str(e)
    html_table += '</tr>'

    with open('html/lcmv.html', 'w') as f:
        f.write(html_header)
        f.write(html_table)
        f.write(html_footer)
