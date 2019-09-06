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
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style('whitegrid')
sns.set_context('poster')
from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils

model = 'Image2vec Encoding Models'
experiment = 'metasema'
alpha = int(1e2)
here = '7 rois image2vec'
model_name = 'Ridge Regression'
cv = 'Random Partition 100 folds'
img_dir = '../../../../results/{}/RP/{}'.format(experiment,here)
here = '7 rois word2vec'
word_dir = '../../../../results/{}/RP/{}'.format(experiment,here)
here = 'compare word2vec and image2vec'
figure_dir = '../../../../figures/{}/RP/{}'.format(experiment,here)
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)


df_img = pd.concat([pd.read_csv(f) for f in glob(os.path.join(img_dir,'*.csv'))])
df_word = pd.concat([pd.read_csv(f) for f in glob(os.path.join(word_dir,'*.csv'))])
roi_order = np.sort(pd.unique(df_img['roi']))
N = len(pd.unique(df_img['sub']))


df_img['Model'] = 'Image2vec'
df_word['Model'] = 'Word2vec'
df = pd.concat([df_img,df_word])
df = df.sort_values(['roi','condition','Model','sub'])
df['condition'] = df['condition'].map({'read':'Shallow Processing',
                                       'reenact':'Deep Processing'})

temp = dict(
        F = [],
        df_nonimator = [],
        df_denominator = [],
        p = [],
        model = [],
        condition = [],
        roi = [],
        )
for (model,condition,roi),df_sub in df.groupby(['Model','condition','roi']):
    df_sub
    anova = ols('mean_variance ~ model_name',data = df_sub).fit()
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
for roi,df_sub in anova_results.groupby('condition'):
    df_sub = df_sub.sort_values('p')
    converter = utils.MCPConverter(pvals = df_sub['p'].values)
    d = converter.adjust_many()
    df_sub['p_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
anova_results = pd.concat(temp)
anova_results['stars'] = anova_results['p_corrected'].apply(utils.stars)
anova_results = anova_results.sort_values(['roi','condition','model'])


g = sns.catplot(x = 'roi',
                y = 'mean_variance',
                hue = 'model_name',
                hue_order = ['vgg19', 
                             'densenet121', 
                             'mobilenetv2_1', 
                             'Fast_Text', 
                             'Glove',
                             'Word2Vec'],
                row = 'condition',
                data = df,
                kind = 'bar',
                aspect = 3,
                sharey = False,)
g._legend.set_title('Encoding Models')
(g.set_axis_labels("ROIs","Mean Variance Explained")
  .set_titles("{row_name}"))
g.axes[0][0].set(title='Shallow Processing')
g.axes[1][0].set(title='Deep Processing')
k = {'Image2vec':-0.25,
     'Word2vec':0.175}
j = 0.15
l = 0.0005
for ax,condition in zip(g.axes.flatten(),['Shallow Processing','Deep Processing']):
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.set(ylim=(0,0.045))
    df_sub = anova_results[anova_results['condition'] == condition]
    for ii,((roi),df_star_sub) in enumerate(df_sub.groupby(['roi',])):
        for model,df_star_sub_model in df_star_sub.groupby(['model']):
            ax.hlines(0.038,ii+k[model]-j,ii+k[model]+j)
            ax.vlines(ii+k[model]-j,0.038+l,0.038-l)
            ax.vlines(ii+k[model]+j,0.038+l,0.038-l)
            ax.annotate(df_star_sub_model['stars'].values[0],xy=(ii+k[model]-0.1,0.04))
g.savefig(os.path.join(figure_dir,'fig1.png'),
          dpi = 400,
          bbox_inches = 'tight')
g.fig.suptitle("Comparison between Computer Vision and Word Embedding Models\nAverage Variance Explained\nN = {}, {} (alpha = {})\nBonforroni corrected for multiple one-way ANOVAs".format(
        N,model_name,alpha),
            y = 1.15)
g.savefig(os.path.join(figure_dir,'mean variance explained ({}).png'.format(model_name)),
          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,'fig1.pdf'),
          format = 'pdf',
          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,'mean variance explained ( {} light).png'.format(model_name)),
#          dpi = 400,
          bbox_inches = 'tight')

# comparison
df_img = df_img.groupby(['condition','roi','sub','model_name']).mean().reset_index()
df_word = df_word.groupby(['condition','roi','sub','model_name']).mean().reset_index()

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
        df_diff = df_diff[['condition','roi','sub','best_variance','mean_variance']]
        df_diff['imageNet'] = imageNet_model
        df_diff['wordNet'] = word2vec_model
        temp.append(df_diff)

df_difference = pd.concat(temp)
df_difference['Model'] = df_difference['imageNet'] + ' - ' + df_difference['wordNet']

df_plot = pd.melt(df_difference,
                  id_vars = ['condition','roi','sub','Model'],
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
for (condition,roi,model),df_sub in df_difference.groupby(['condition','roi','Model']):
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
    df_sub = df_sub.sort_values(['ps_mean'])
    converter = utils.MCPConverter(pvals = df_sub['ps_mean'].values)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
df_stat = pd.concat(temp)
df_stat['stars'] = df_stat['ps_corrected'].apply(utils.stars)

df_plot = df_plot.sort_values(['roi','condition','Model','sub'])
df_stat = df_stat.sort_values(['roi','condition','model'])



g = sns.catplot(x = 'roi',
                y = 'Differences of Variance Explained',
                hue = 'Model',
                hue_order = np.sort(pd.unique(df_plot['Model'])),
                row = 'condition',
                data = df_plot[df_plot['Variance Explained'] == 'mean_variance'],
                kind = 'bar',
                aspect = 3,
                sharey = False,)
(g.set_axis_labels('ROIs','$\Delta$ Variance Explained'))
g._legend.set_title('Pairs of Encoding Models')
g.axes[0][0].set(title = 'Shallow Processing')
g.axes[1][0].set(title = 'Deep Processing')
g.savefig(os.path.join(figure_dir,'fig2.png'),
          dpi = 400,
          bbox_inches = 'tight')
g.fig.suptitle('Difference of Variance Explained by the Computer Vision Models and Word Embedding Models\nComputer Vision Model - Word Embedding Model\nbonferroni corrected,{}, alpha = {}'.format(
        model_name,alpha),
              y = 1.08)
g.savefig(os.path.join(figure_dir,'Difference of Variance Explained by the Image2Vec and Word2Vec.png'),
          dpi = 400,
          bbox_inches = 'tight')
g.savefig(os.path.join(figure_dir,'fig2.pdf'),
          format = 'pdf',
          dpi = 400,
          bbox_inches = 'tight')

df_condition = dict(
        roi = [],
        condition = [],
        ps_mean = [],
        ps_std = [],
        diff_mean = [],
        diff_std = [],)
for (roi,condition),df_sub in df.groupby(['roi','condition']):
    df_sub_img = df_sub[df_sub['Model'] == 'Image2vec'].groupby(['sub']).mean().reset_index()
    df_sub_word = df_sub[df_sub['Model'] == 'Word2vec'].groupby(['sub']).mean().reset_index()
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
    df_sub = df_sub.sort_values(['ps_mean'])
    converter = utils.MCPConverter(pvals = df_sub['ps_mean'].values)
    d = converter.adjust_many()
    df_sub['ps_corrected'] = d['bonferroni'].values
    temp.append(df_sub)
df_condition = pd.concat(temp)


d = df_img.groupby(['condition','roi','sub']).mean().reset_index()['mean_variance']- df_word.groupby(['condition','roi','sub']).mean().reset_index()['mean_variance']
df_plot = df_img.groupby(['condition','roi','sub']).mean().reset_index().copy()
df_plot['mean_variance'] = d.values

df_diff_diff = dict(
        roi = [],
        ps_mean = [],
        ps_std = [],
        diff_mean = [],
        diff_std = [],)
for roi,df_sub in df_plot.groupby(['roi']):
    df_read = df_sub[df_sub['condition'] == 'read']#.sort_values(['sub'])
    df_reenact = df_sub[df_sub['condition'] == 'reenact']#.sort_values(['sub'])
    
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
converter = utils.MCPConverter(pvals = df_diff_diff['ps_mean'].values)
d = converter.adjust_many()
df_diff_diff['ps_corrected'] = d['bonferroni'].values
df_diff_diff['star'] = df_diff_diff['ps_corrected'].apply(utils.stars)


df_plot = df_plot.sort_values(['roi'])
df_diff_diff = df_diff_diff.sort_values(['roi'])

g = sns.catplot(x = 'roi',
                y = 'mean_variance',
                hue = 'condition',
                data = df_plot,
                kind = 'violin',
                aspect = 3,
                legend=False,
                **dict(split = True,
                       cut = 0,
                       inner = 'quartile'))
(g.set_axis_labels('ROIs','$\Delta$ Variance Explained')
  .set(ylim=(0,0.04)))
legend = plt.legend(loc='center left', bbox_to_anchor=(1., 0.5), ncol=1)
legend.set_title('Conditions')
legend.texts[0].set_text('Shallow')
legend.texts[1].set_text('Deep')
for ii in range(7):
    g.axes[0][0].annotate(df_diff_diff['star'].values[ii],
          xy = (ii,0.035))
g.savefig(os.path.join(figure_dir,'fig3.png'),
          dpi = 400,
          bbox_inches = 'tight')
g.fig.suptitle('Difference of Computer Vision Models and Word Embedding Models\nBonforroni Corrected for multiple t-tests',
               y=1.05)

g.savefig(os.path.join(figure_dir,'fig3.pdf'),
          format = 'pdf',
          dpi = 400,
          bbox_inches = 'tight')



















