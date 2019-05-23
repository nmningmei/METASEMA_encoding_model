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

model = 'Word2vec Encoding Models'
experiment = 'metasema'
here = '7 rois word2vec'
cv = 'Random Partition 100 folds'
working_dir = '../../../../results/{}/RP/{}'.format(experiment,here)
figure_dir = '../../../../figures/{}/RP/{}'.format(experiment,here)
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)
df = pd.concat([pd.read_csv(f) for f in glob(os.path.join(working_dir,'*.csv'))])
roi_order = np.sort(pd.unique(df['roi']))
N = len(pd.unique(df['sub']))
usual = dict(
        linestyle = '--',
        color = 'black',
        alpha = 0.5)
g = sns.catplot(x = 'condition',
                y = 'mean_variance',
                hue = 'roi',
                hue_order = roi_order,
                row = 'model_name',
                row_order = ['Fast_Text','Glove','Word2Vec'],
                data = df,
                kind = 'violin',
                aspect = 3,
                height = 8,
                **{'cut':0,
                   'inner':'quartile',}
                )
(g.set_axis_labels('Condition','Variance Explained')
  .set_titles("{row_name}"))
#[ax.axhline(df['mean_variance'].mean(),**usual) for ax in g.axes.flatten()]
g.fig.suptitle('{},{}\nPartition: {}\nN = {}'.format(
        model,
        experiment,
        cv,
        N),
            y = 1.05)
g.savefig(os.path.join(figure_dir,
                       'W2V encoding models ({}).png'.format(experiment)),
        dpi = 400,
        bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,
                       'W2V encoding models ({} light).png'.format(experiment)),
#        dpi = 400,
        bbox_inches = 'tight')


#g = sns.catplot(x = 'condition',
#                y = 'determination',
#                hue = 'roi',
#                hue_order = roi_order,
#                row = 'word2vec_model',
#                row_order = ['Fast_Text','Glove','Word2Vec'],
#                data = df,
#                kind = 'violin',
#                aspect = 3,
#                height = 8,
#                **{'cut':0,
#                   'inner':'quartile',}
#                )
#(g.set_axis_labels('Condition','Variance Explained (Kay et al 2013)')
#  .set_titles("{row_name}"))
##[ax.axhline(df['determination'].mean(),0.06,**usual) for ax in g.axes.flatten()]
#g.fig.suptitle('{},{}\nPartition: {}\nN = {}'.format(
#        model,
#        experiment,
#        cv,
#        N),
#            y = 1.05)
#g.savefig(os.path.join(figure_dir,
#                       'W2V encoding models (kay) ({}).png'.format(experiment)),
#        dpi = 400,
#        bbox_inches = 'tight')
#g.savefig(os.path.join(figure_dir,
#                       'W2V encoding models (kay) ({} light).png'.format(experiment)),
##        dpi = 400,
#        bbox_inches = 'tight')













