#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:24:14 2019

@author: nmei
"""

import pandas as pd
import numpy as np
from glob import glob
import os
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt
from mne.stats import fdr_correction
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

model = 'Image2vec Encoding Models'
experiment = 'metasema'
alpha = int(1e2)
here = 'encoding model 15 ROIs'
model_name = 'Ridge Regression'
cv = 'Random Partition 300 folds'
multiple_correction_method = 'FDR Benjamini-Hochberg'
working_dir  = '../../../../results/{}/RP/{}'.format(experiment,here)
here = 'compare word2vec and image2vec 15 roi'
figure_dir = '../../../../figures/{}/RP/{}'.format(experiment,here)
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)


df = pd.concat([pd.read_csv(f) for f in glob(os.path.join(working_dir,'*.csv'))])
df['roi_name'] = df['roi_name'].apply(lambda x:x.split('_')[-1])
roi_order = np.sort(pd.unique(df['roi_name']))
N = len(pd.unique(df['sub_name']))


df['model_name'] = df['model_name'].map({'fast_text':'Fast Text',
                                         'glove':'GloVe',
                                         'word2vec':'Word2Vec',
                                         'vgg19':'VGG19',
                                         'densenet121':'DenseNet1211',
                                         'mobilenetv2_1':'MobileNetV2'})
df['Model'] = df['model_name'].map({'Fast Text':'W2V',
                                     'GloVe':'W2V',
                                     'Word2Vec':'W2V',
                                     'VGG19':'I2V',
                                     'DenseNet1211':'I2V',
                                     'MobileNetV2':'I2V'})
df = df.sort_values(['roi_name','condition','Model','model_name','sub_name'])
df['condition'] = df['condition'].map({'read':'Sallow Process','reenact':'Deep Process'})

df = df.groupby(['roi_name','condition','Model','model_name','sub_name']).mean().reset_index()

temp = dict(
        F = [],
        df_nonimator = [],
        df_denominator = [],
        p = [],
        model = [],
        condition = [],
        roi = [],
        )
for (model,condition,roi),df_sub in df.groupby(['Model','condition','roi_name']):
    df_sub
    df_sub['mean_variance_norm'] = (df_sub['mean_variance'] - df_sub['mean_variance'].mean())/df_sub['mean_variance'].std()
    anova = ols('mean_variance_norm ~ model_name',data = df_sub).fit()
    aov_table = sm.stats.anova_lm(anova,typ=2)
    print(aov_table)
    temp['F'].append(aov_table['F']['model_name'])
    temp['df_nonimator'].append(aov_table['df']['model_name'])
    temp['df_denominator'].append(aov_table['df']['Residual'])
    temp['p'].append(aov_table['PR(>F)']['model_name'])
    temp['model'].append(model)
    temp['condition'].append(condition)
    temp['roi'].append(roi)
anova_results = pd.DataFrame(temp)
temp = []
from statsmodels.stats.multitest import multipletests
for roi,df_sub in anova_results.groupby('condition'):
    df_sub = df_sub.sort_values('p')
#    converter = utils.MCPConverter(pvals = df_sub['p'].values)
#    d = converter.adjust_many()
    pvals = df_sub['p'].values
    ps_corrected = multipletests(pvals,method='fdr_bh',is_sorted=True)
    df_sub['p_corrected'] = ps_corrected[1]
    temp.append(df_sub)
anova_results = pd.concat(temp)
anova_results['stars'] = anova_results['p_corrected'].apply(utils.stars)
anova_results = anova_results.sort_values(['roi','condition','model'])
anova_results.to_csv('../../../../results/{}/RP/{}/one way ANOVA.csv'.format(experiment,'encoding 15 stats'),
                     index = False)


g = sns.catplot(x = 'roi_name',
                y = 'mean_variance',
                hue = 'model_name',
                hue_order = ['VGG19', 
                             'DenseNet1211', 
                             'MobileNetV2', 
                             'Fast Text', 
                             'GloVe',
                             'Word2Vec'],
                row = 'condition',
                data = df,
                kind = 'bar',
                aspect = 6,
                sharey = False,)
g._legend.set_title('Encoding Models')
(g.set_axis_labels("ROIs","Mean Variance Explained")
  .set_titles("{row_name}")
  .set(ylim = (0, 0.06)))
g.axes[0][0].set(title='Shallow Process')
g.axes[1][0].set(title='Deep Process')
k = {'I2V':-0.25,
     'W2V':0.175}
j = 0.15
l = 0.0005
for ax,condition in zip(g.axes.flatten(),['Sallow Process','Deep Process']):
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    df_sub = anova_results[anova_results['condition'] == condition]
    for ii,((roi),df_star_sub) in enumerate(df_sub.groupby(['roi',])):
        for model,df_star_sub_model in df_star_sub.groupby(['model']):
            ax.hlines(0.043,ii+k[model]-j,ii+k[model]+j)
            ax.vlines(ii+k[model]-j,0.043+l,0.043-l)
            ax.vlines(ii+k[model]+j,0.043+l,0.043-l)
            ax.annotate(df_star_sub_model['stars'].values[0],
                        xy=(ii+k[model]-0.1,0.045))
g.savefig(os.path.join(figure_dir,'fig5.png'),
          dpi = 400,
          bbox_inches = 'tight')
g.fig.suptitle("Comparison between Computer Vision and Word Embedding Models\nAverage Variance Explained\nN = {}, {} (alpha = {})\n{} corrected for multiple one-way ANOVAs".format(
        N,model_name,alpha,multiple_correction_method),
            y = 1.15)
g.savefig(os.path.join(figure_dir,'mean variance explained ({}).png'.format(model_name)),
          dpi = 400,
          bbox_inches = 'tight')

g.savefig(os.path.join(figure_dir,'fig5 (light).png'),
#          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,'mean variance explained ( {} light).png'.format(model_name)),
#          dpi = 400,
          bbox_inches = 'tight')



# comparison
df_img = df[df['Model'] == 'I2V'].groupby(['condition','roi_name','sub_name','model_name']).mean().reset_index()
df_word = df[df['Model'] == 'W2V'].groupby(['condition','roi_name','sub_name','model_name']).mean().reset_index()

temp = []
for imageNet_model in pd.unique(df_img['model_name']):
    for word2vec_model in pd.unique(df_word['model_name']):
        df_image2vec = df_img[df_img['model_name'] == imageNet_model]
        df_word2vec = df_word[df_word['model_name'] == word2vec_model]
        var_mean_diff = df_image2vec['mean_variance'].values - df_word2vec['mean_variance'].values
        var_best_diff = df_image2vec['best_variance'].values - df_word2vec['best_variance'].values
        df_diff = df_image2vec.copy()
        df_diff['mean_variance'] = var_mean_diff
        df_diff['best_variance'] = var_best_diff
        df_diff = df_diff[['condition',
                           'roi_name',
                           'sub_name',
                           'best_variance',
                           'mean_variance']]
        df_diff['imageNet'] = imageNet_model
        df_diff['wordNet'] = word2vec_model
        temp.append(df_diff)

df_difference = pd.concat(temp)
df_difference['Model'] = df_difference['imageNet'] + ' - ' + df_difference['wordNet']

df_plot = pd.melt(df_difference,
                  id_vars = ['condition','roi_name','sub_name','Model'],
                  value_vars = ['mean_variance','best_variance'],
                  var_name = 'Variance Explained',
                  value_name = 'Differences of Variance Explained')

df_stat = dict(
        condition = [],
        roi = [],
        model = [],
        diff_mean = [],
        diff_std = [],
        ps_mean = [],
        ps_std = [],)
col = 'mean_variance'
for (condition,roi,model),df_sub in df_difference.groupby(['condition','roi_name','Model']):
    df_sub
    ps = utils.resample_ttest(df_sub[col].values,baseline = 0,
                              n_ps = 100, n_permutation = int(5e4))
    df_stat['condition'].append(condition)
    df_stat['roi'].append(roi)
    df_stat['model'].append(model)
    df_stat['diff_mean'].append(df_sub[col].values.mean())
    df_stat['diff_std'].append(df_sub[col].values.std())
    df_stat['ps_mean'].append(ps.mean())
    df_stat['ps_std'].append(ps.std())
df_stat = pd.DataFrame(df_stat)

temp = []
for (condition),df_sub in df_stat.groupby(['condition']):
    df_sub = df_sub.sort_values('ps_mean')
#    converter = utils.MCPConverter(pvals = df_sub['p'].values)
#    d = converter.adjust_many()
    pvals = df_sub['ps_mean'].values
    ps_corrected = multipletests(pvals,method='fdr_bh',is_sorted=True)
    df_sub['ps_corrected'] = ps_corrected[1]
    temp.append(df_sub)
df_stat = pd.concat(temp)
df_stat['stars'] = df_stat['ps_corrected'].apply(utils.stars)

df_plot = df_plot.sort_values(['roi_name','condition','Model','sub_name'])
df_stat = df_stat.sort_values(['roi','condition','model'])
df_stat.to_csv('../../../../results/{}/RP/{}/pair t tests.csv'.format(experiment,'encoding 15 stats'),
                     index = False)


g = sns.catplot(x = 'roi_name',
                y = 'Differences of Variance Explained',
                hue = 'Model',
                hue_order = np.sort(pd.unique(df_plot['Model'])),
                row = 'condition',
                data = df_plot[df_plot['Variance Explained'] == 'mean_variance'],
                kind = 'bar',
                aspect = 6,
                sharey = False,)
(g.set_axis_labels('ROIs','$\Delta$ Variance Explained'))
g._legend.set_title('Pairs of Encoding Models')
g.axes[0][0].set(title = 'Shallow Process')
g.axes[1][0].set(title = 'Deep Process')
g.savefig(os.path.join(figure_dir,'fig6.png'),
          dpi = 400,
          bbox_inches = 'tight')
g.fig.suptitle('Difference of Variance Explained by the Computer Vision Models and Word Embedding Models\nComputer Vision Model - Word Embedding Model\n{} corrected,{}, alpha = {}'.format(
        multiple_correction_method,model_name,alpha),
              y = 1.08)
g.savefig(os.path.join(figure_dir,'Difference of Variance Explained by the Image2Vec and Word2Vec.png'),
          dpi = 400,
          bbox_inches = 'tight')

g.savefig(os.path.join(figure_dir,'diff.png'),
          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,'fig6 (light).png'),
#          dpi = 400,
          bbox_inches = 'tight')

df_condition = dict(
        roi = [],
        condition = [],
        ps_mean = [],
        ps_std = [],
        diff_mean = [],
        diff_std = [],)
for (roi,condition),df_sub in df.groupby(['roi_name','condition']):
    df_sub_img = df_sub[df_sub['Model'] == 'I2V'].groupby(['sub_name']).mean().reset_index()
    df_sub_word = df_sub[df_sub['Model'] == 'W2V'].groupby(['sub_name']).mean().reset_index()
    a = df_sub_img['mean_variance'].values
    b = df_sub_word['mean_variance'].values
    ps = utils.resample_ttest_2sample(a,b,
                                      n_ps = 100,
                                      n_permutation = int(5e4),
                                      one_tail = False,
                                      match_sample_size = True)
    df_condition['roi'].append(roi)
    df_condition['condition'].append(condition)
    df_condition['ps_mean'].append(ps.mean())
    df_condition['ps_std'].append(ps.std())
    df_condition['diff_mean'].append(np.mean(a - b))
    df_condition['diff_std'].append(np.std(a - b))
df_condition = pd.DataFrame(df_condition)

temp = []
for condition, df_sub in df_condition.groupby(['condition']):
    df_sub = df_sub.sort_values('ps_mean')
#    converter = utils.MCPConverter(pvals = df_sub['p'].values)
#    d = converter.adjust_many()
    pvals = df_sub['ps_mean'].values
    ps_corrected = multipletests(pvals,method='fdr_bh',is_sorted=True)
    df_sub['ps_corrected'] = ps_corrected[1]
    temp.append(df_sub)
df_condition = pd.concat(temp)


d = df_img.groupby(['condition','roi_name','sub_name']).mean().reset_index()['mean_variance'] -\
    df_word.groupby(['condition','roi_name','sub_name']).mean().reset_index()['mean_variance']
df_plot = df_img.groupby(['condition','roi_name','sub_name']).mean().reset_index().copy()
df_plot['mean_variance'] = d.values

df_diff_diff = dict(
        roi = [],
        ps_mean = [],
        ps_std = [],
        diff_mean = [],
        diff_std = [],)
for roi,df_sub in df_plot.groupby(['roi_name']):
    df_read = df_sub[df_sub['condition'] == 'Sallow Process']#.sort_values(['sub'])
    df_reenact = df_sub[df_sub['condition'] == 'Deep Process']#.sort_values(['sub'])
    a = df_read['mean_variance'].values
    b = df_reenact['mean_variance'].values
    ps = utils.resample_ttest_2sample(a,b,
                                      n_ps = 100,
                                      n_permutation = int(5e4),
                                      one_tail = False,
                                      match_sample_size = True)
    df_diff_diff['roi'].append(roi)
    df_diff_diff['ps_mean'].append(ps.mean())
    df_diff_diff['ps_std'].append(ps.std())
    df_diff_diff['diff_mean'].append(np.mean(np.abs(a - b)))
    df_diff_diff['diff_std'].append(np.std(np.abs(a - b)))
df_diff_diff = pd.DataFrame(df_diff_diff)
df_diff_diff = df_diff_diff.sort_values(['ps_mean'])
pvals = df_diff_diff['ps_mean'].values
ps_corrected = multipletests(pvals,method='fdr_bh',is_sorted=True)
df_diff_diff['ps_corrected'] = ps_corrected[1]
df_diff_diff['star'] = df_diff_diff['ps_corrected'].apply(utils.stars)


df_plot = df_plot.sort_values(['roi_name'])
df_diff_diff = df_diff_diff.sort_values(['roi'])
df_diff_diff.to_csv('../../../../results/{}/RP/{}/difference of difference.csv'.format(experiment,'encoding 15 stats'),
                     index = False)

unique_rois = pd.unique(df_plot['roi_name'])
n_axis = {name:ii < 7 for ii,name in enumerate(unique_rois)}
df_plot['subplots'] = df_plot['roi_name'].map(n_axis)
df_diff_diff['subplots'] = df_diff_diff['roi'].map(n_axis)
fig,axes = plt.subplots(figsize = (20,10),nrows = 2,
                        sharey = True)
for (subplot,df_sub),ax in zip(df_plot.groupby(['subplots']),axes):
    df_diff_diff_sub = df_diff_diff[df_diff_diff['subplots'] == subplot]
    ax = sns.violinplot(x = 'roi_name',
                        y = 'mean_variance',
                        hue = 'condition',
                        split = True,
                        cut = 0,
                        inner = 'quartile',
                        data = df_sub,
                        ax = ax)
    ax.set(xlabel = '',ylabel = '$\Delta$ Variance Explained',
           ylim = (0.,0.045))
    ax.legend(bbox_to_anchor=(1.3, 0.6))
    ax.get_legend().set_title('Conditions')
    ax.get_legend().texts[0].set_text('Sallow Process')
    ax.get_legend().texts[1].set_text('Deep Process')
    for ii in range(len(pd.unique(df_sub['roi_name']))):
        ax.annotate(df_diff_diff['star'].values[ii],
              xy = (ii,0.04))
    if subplot > 0:
        ax.get_legend().remove()
fig.tight_layout()
fig.savefig(os.path.join(figure_dir,'fig7.png'),
          dpi = 400,
          bbox_inches = 'tight')
fig.suptitle('Difference of Computer Vision Models and Word Embedding Models\n{} correction for multiple paired-sample t-tests'.format(
        multiple_correction_method),
               y=1.05)

fig.savefig(os.path.join(figure_dir,'diff of diff.png'),
          dpi = 400,
          bbox_inches = 'tight')
fig.savefig(os.path.join(figure_dir,'fig7 (light).png'),
#          dpi = 400,
          bbox_inches = 'tight')




















