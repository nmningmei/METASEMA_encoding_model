#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:22:54 2019

@author: nmei
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')

model = 'Image2vec Encoding Models'
experiment = 'metasema'
here = '7 rois image2vec'
cv = 'Random Partition 100 folds'
working_dir = '../../../../results/{}/RP/{}'.format(experiment,here)
figure_dir = '../../../../figures/{}/RP/{}'.format(experiment,here)
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
df = pd.concat([pd.read_csv(f) for f in glob(os.path.join(working_dir,'*.csv'))])
df = df.sort_values(['condition','roi','model_name','sub'])
roi_order = np.sort(pd.unique(df['roi']))
N = len(pd.unique(df['sub']))
usual = dict(
        linestyle = '--',
        color = 'black',
        alpha = 0.5)
df = df.sort_values(['condition'])
g = sns.catplot(x = 'roi',
                y = 'mean_variance',
                hue = 'condition',
#                hue_order = roi_order,
                row = 'model_name',
#                row_order = ['Fast_Text','Glove','Word2Vec'],
                data = df,
                kind = 'violin',
                aspect = 3,
                height = 8,
                **{'cut':0,
                   'inner':'quartile',
                   'split':True,}
                )
(g.set_axis_labels('ROIs','Variance Explained')
  .set_titles("{row_name}"))
#[ax.axhline(df['mean_variance'].mean(),**usual) for ax in g.axes.flatten()]
g.fig.suptitle('{},{}\nPartition: {}\nN = {}'.format(
        model,
        experiment,
        cv,
        N),
            y = 1.05)
g.savefig(os.path.join(figure_dir,
                       'I2V encoding models ({}).png'.format(experiment)),
        dpi = 400,
        bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,
                       'I2V encoding models ({} light).png'.format(experiment)),
#        dpi = 400,
        bbox_inches = 'tight')

df_read = df[df['condition'] == 'read']
df_reenact = df[df['condition'] == 'reenact']

df_read = df_read.sort_values(['roi','model_name','sub'])
df_reenact = df_reenact.sort_values(['roi','model_name','sub'])

var_mean_diff = df_reenact['mean_variance'] - df_read['mean_variance']
var_best_diff = df_reenact['best_variance'] - df_read['best_variance']

df_diff = df_read.copy()
df_diff['mean_variance'] = var_mean_diff
df_diff['best_variance'] = var_best_diff
df_diff['condition'] = 'reenact - read'

g = sns.catplot(x = 'roi',
                y = 'mean_variance',
                hue = 'model_name',
                data = df_diff,
                kind = 'bar',
                aspect = 3,
                )
(g.set_axis_labels('ROIs','Difference of Mean Variance Explained'))
g.fig.suptitle('Difference of variance explained between reenact and read\n{}, {}, Ridge Regression (alpha = 100)\npartition = {},N = {}'.format(
        model,
        experiment,
        cv,
        N),
               y = 1.15)





