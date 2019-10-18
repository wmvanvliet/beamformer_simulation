import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import mne
import surfer
from mayavi import mlab
mlab.options.offscreen = True
from itertools import product
from utils import set_directory

import config
from config import fname

fwd = mne.read_forward_solution(fname.fwd_man)

dfs = []
for vertex in tqdm(range(3765), total=3765):
    try: 
        df = pd.read_csv(fname.dics_results(noise=config.noise, vertex=vertex))
        df['vertex'] = vertex
        df['noise'] = config.noise
        dfs.append(df)
    except Exception as e:
        print(e)
dics = pd.concat(dfs, ignore_index=True)
dics['pick_ori'].fillna('none', inplace=True)
dics['weight_norm'].fillna('none', inplace=True)

regs = [0.05, 0.1, 0.5]
sensor_types = ['grad', 'mag']
pick_oris = ['none', 'normal', 'max-power']
inversions = ['single', 'matrix']
weight_norms = ['unit-noise-gain', 'none']
normalize_fwds = [True, False]
real_filters = [True, False]
settings = list(product(regs, sensor_types, pick_oris, inversions,
                        weight_norms, normalize_fwds, real_filters))

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
        <th>inversion</th>
        <th>weight_norm</th>
        <th>normalize_fwd</th>
        <th>real_filter</th>
        <th colspan="2">P2P distance</th>
        <th colspan="2">Fancy metric</th>
    </tr>
    <tr>
        <td><input type="text" onkeyup="filter(0, this)" placeholder="reg"></td>
        <td><input type="text" onkeyup="filter(1, this)" placeholder="sensor type"></td>
        <td><input type="text" onkeyup="filter(2, this)" placeholder="pick_ori"></td>
        <td><input type="text" onkeyup="filter(3, this)" placeholder="inversion"></td>
        <td><input type="text" onkeyup="filter(4, this)" placeholder="weight_norm"></td>
        <td><input type="text" onkeyup="filter(5, this)" placeholder="normalize_fwd"></td>
        <td><input type="text" onkeyup="filter(6, this)" placeholder="real_filter"></td>
        <td colspan="2"></td>
        <td colspan="2"></td>
    </tr>
'''

html_footer = '</body></table>'

html_table = ''
for i, setting in enumerate(settings):
    q = ("reg==%.1f and sensor_type=='%s' and pick_ori=='%s' and "
         "inversion=='%s' and weight_norm=='%s' and normalize_fwd==%s and real_filter==%s" % setting)
    sel = dics.query(q).dropna()

    reg, sensor_type, pick_ori, inversion, weight_norm, normalize_fwd, real_filters = setting

    # Skip some combinations
    if weight_norm == 'unit-noise-gain' and normalize_fwd == True:
        continue
    if weight_norm == 'none' and normalize_fwd == False:
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
        mlab.savefig('html/dics/%03d_dist_out.png' % i)
        mlab.view(-180, 90, 300, [33, -10, 35])
        mlab.savefig('html/dics/%03d_dist_in.png' % i)
        mlab.close(1)

        mlab.figure(1, size=(600, 500))
        vertices = fwd['src'][1]['vertno'][sel['vertex']]
        brain = surfer.Brain('sample', hemi='rh', surf='white', figure=1)
        brain.add_data(sel['eval'], vertices=vertices, smoothing_steps=5)
        brain.scale_data_colormap(0, 0.0005, 0.001, transparent=False)
        mlab.view(0, 90, 250, [33, -10, 35])
        mlab.savefig('html/dics/%03d_eval_out.png' % i)
        mlab.view(-180, 90, 300, [33, -10, 35])
        mlab.savefig('html/dics/%03d_eval_in.png' % i)
        mlab.close(1)
        html_table += '<td><img src="dics/%03d_dist_out.png"></td><td><img src="dics/%03d_dist_in.png"></td>' % (i, i)
        html_table += '<td><img src="dics/%03d_eval_out.png"></td><td><img src="dics/%03d_eval_in.png"></td>' % (i, i)
    except Exception as e:
        html_table += '<td colspan="2">%s</td>' % str(e)
    html_table += '</tr>'

    set_directory('html')
    with open('html/dics.html', 'w') as f:
        f.write(html_header)
        f.write(html_table)
        f.write(html_footer)
