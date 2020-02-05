#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:49:55 2020

@author: nmei
"""


import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import os
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt
from mne.stats import fdr_correction
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')
import matplotlib
font = {'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)
matplotlib.rcParams['font.weight']= 'bold'
from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

model = 'Image2vec Encoding Models'
experiment = 'metasema'
alpha = int(1e2)
here = 'encoding_model_15_ROIs_arrays'
model_name = 'Ridge Regression'
cv = 'Random Partition 300 folds'
multiple_correction_method = 'FDR Benjamini-Hochberg'
working_dir  = '../../../../results/{}/RP/{}'.format(experiment,here)
#working_dir = '/bcbl/home/home_n-z/nmei/bench_marking/results/{}/RP/{}'.format(experiment,here)
here = 'compare word2vec and image2vec 15 roi'
figure_dir = '../../../../figures/{}/RP/{}'.format(experiment,here)
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

working_data = glob(os.path.join(working_dir,'*.npy'))
df_collect = dict(
        sub_name = [],
        roi = [],
        condition = [],
        model_name = [],
        path = [],
        scores = [],
        )
def _append(df_collect,mapping):
    for key,values in mapping.items():
        df_collect[key].append(values)
    return df_collect
for f in tqdm(working_data):
    try:
        _,_,sub_name,roi,condition,model = f.split(' ')
        model = model.split('.')[0]
    except:
        _,_,sub_name,roi,condition,model1,model2 = f.split(' ')
        model = f'{model1} {model2.split(".")[0]}'
    df_collect = _append(df_collect,mapping = dict(
            sub_name=sub_name,
            roi=roi,
            condition=condition,
            model_name=model,
            path=f,
            scores=np.load(f).mean(0)
            ))
df_collect = pd.DataFrame(df_collect)
df_collect['model_name'] = df_collect['model_name'].map({'fast text':'Fast Text',
                                         'glove':'GloVe',
                                         'word2vec':'Word2Vec',
                                         'concatenated_word2vec':'Word Embedding',
                                         'VGG19':'VGG19',
                                         'DenseNet169':'DenseNet169',
                                         'MobileNetV2':'MobileNetV2'})
df_collect['Model'] = df_collect['model_name'].map({'Fast Text':'W2V',
                                    'GloVe':'W2V',
                                    'Word2Vec':'W2V',
                                    'Word Embedding':'W2V',
                                    'VGG19':'I2V',
                                    'DenseNet169':'I2V',
                                    'MobileNetV2':'I2V'})
df = dict(
        sub_name = [],
        roi = [],
        condition = [],
        model_type = [],
        scores = [],
        positive_voxels = [],
        )
for (sub_name,roi,condition,Model),df_sub in df_collect.groupby(['sub_name',
    'roi','condition','Model']):
    temp = df_sub['scores'].values.mean(0)
    df = _append(df,
                 mapping = dict(sub_name = sub_name,
                                roi = roi,
                                condition = condition,
                                model_type = Model,
                                scores = temp,
                                positive_voxels = np.sum(temp >=0)))
df = pd.DataFrame(df)

df['roi_name'] = df['roi'].apply(lambda x:x.split('_')[-1])
df['roi_name'] = df['roi_name'].map({'frontpole':'Frontal Pole', 
                                     'fusif':'Fusirorm Gyrus', 
                                     'infpar':'Inferior Parietal Lobe', 
                                     'inftemp':'Inferior Temporal Lobe', 
                                     'lofc':'Lateral Orbitofrontal Cortex', 
                                     'mofc':'Medial Orbitfrontal Cortex', 
                                     'mtemp':'Medial Temporal Lobe',
                                     'parsoper':'Pars Opercularis', 
                                     'parsorbi':'Pars Orbitalis', 
                                     'parstri':'Pars Triangularis', 
                                     'phipp':'Parahippocampal Gyrus', 
                                     'postcing':'Posterior Cingulate Gyrus', 
                                     'precun':'Precuneus',
                                     'sfrontal':'Superior Frontal Gyrus', 
                                     'tempole':'Anterior Temporal Lobe'})
df['roi_name_br'] = df['roi_name'].map({'Frontal Pole':'FP', 
                                        'Fusirorm Gyrus':'FFG', 
                                        'Inferior Parietal Lobe':'IPL', 
                                        'Inferior Temporal Lobe':'ITL', 
                                        'Lateral Orbitofrontal Cortex':'LOFC', 
                                        'Medial Orbitfrontal Cortex':'MOFC', 
                                        'Medial Temporal Lobe':'MTL',
                                        'Pars Opercularis':'POP', 
                                        'Pars Orbitalis':'POR', 
                                        'Pars Triangularis':'PTR', 
                                        'Parahippocampal Gyrus':'PHG', 
                                        'Posterior Cingulate Gyrus':'PCG', 
                                        'Precuneus':'Precuneus',
                                        'Superior Frontal Gyrus':'SFG', 
                                        'Anterior Temporal Lobe':'ATL'})
df['ROIs'] = df['roi_name']
df['Conditions'] = df['condition']

sort_by = ['sub_name','roi_name','condition']
df_i2v = df[df['model_type']=='I2V'].sort_values(sort_by)
df_w2v = df[df['model_type']=='W2V'].sort_values(sort_by)

fig,ax  = plt.subplots(figsize = (24,20))
ax = sns.scatterplot(df_w2v['positive_voxels'].values,
                     df_i2v['positive_voxels'].values,
                     hue = df_i2v['ROIs'].values,
                     style = df_i2v['Conditions'].values,
                     ax = ax,
                     )
ax.plot([0,600],[0,600],linestyle = '--',color = 'black',alpha = .4,)
ax.set(xlim=(-10,550),
       ylim=(-10,550),
       xlabel = 'Word embedding models',
       ylabel = 'Computer vision models',
       title = 'Number of Positive Variance Explained Voxels')
fig.savefig(os.path.join(figure_dir,'positive voxels.jpeg'),
            bbox_inches = 'tight',)
fig.savefig(os.path.join(figure_dir,'positive voxels (high).jpeg'),
            dpi = 400,
            bbox_inches = 'tight',)

df_voxel = dict(sub_name=[],
                roi_name=[],
                condition=[],
                score_i=[],
                score_w=[],
                )
for ((sub_name,roi_name,condition),df_i2v_sub),(_,df_w2v_sub) in zip(df_i2v.groupby(['sub_name','roi_name','condition',]),
                 df_w2v.groupby(['sub_name','roi_name','condition',])):
    for ii,ww in zip(df_i2v_sub['scores'].values[0],df_w2v_sub['scores'].values[0]):
        df_voxel = _append(df_voxel,
                           mapping = dict(sub_name=sub_name,
                                          roi_name=roi_name,
                                          condition=condition,
                                          score_i=ii,
                                          score_w=ww,))
df_voxel = pd.DataFrame(df_voxel)
df_voxel['ROIs'] = df_voxel['roi_name']
df_voxel['Conditions'] = df_voxel['condition']
idx = np.logical_or(df_voxel['score_i'].apply(lambda x:x>=0).values,
                    df_voxel['score_w'].apply(lambda x:x>=0).values)

df_voxel_plot = df_voxel[idx]

idx = np.logical_or(df_voxel['score_i'].apply(lambda x:-10<x<0).values,
                    df_voxel['score_w'].apply(lambda x:-10<x<0).values)
df_voxel_negative = df_voxel[idx]

fig,ax = plt.subplots(figsize = (24,20))
ax.scatter(df_voxel_negative['score_w'].values,
           df_voxel_negative['score_i'].values,
           marker = '*',
           s = 1,
           color = 'black',
           alpha = 0.5,
           )
ax = sns.scatterplot('score_w','score_i',
                     hue='ROIs',
                     style='Conditions',
                     data = df_voxel_plot,
                     ax = ax,
                     )
ax.plot([-600,600],[-600,600],linestyle = '--',color = 'black',alpha = .4,)
vims = df_voxel['score_i'].max() * 1.1
ax.set(xlim=(-vims,vims),
       ylim=(-vims,vims),
       xlabel = 'Word embedding models',
       ylabel = 'Computer vision models',
       title = 'Variance Explained of Individual Voxels',
       )
fig.savefig(os.path.join(figure_dir,'voxel wise scores.jpeg'),
            bbox_inches = 'tight',)
fig.savefig(os.path.join(figure_dir,'voxel wise scores (high).jpeg'),
            dpi = 500,
            bbox_inches = 'tight',)
plt.close('all')


































