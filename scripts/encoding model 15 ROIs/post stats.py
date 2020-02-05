#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:54:32 2019

@author: nmei
"""

import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('poster')
import statsmodels.api as sm
from statsmodels.formula.api import ols
from shutil import copyfile
copyfile('../../../utils.py','utils.py')
import utils
from matplotlib.ticker import FormatStrFormatter


experiment = 'metasema'
here = 'encoding_model_15_ROIs'
working_dir = '../../../../results/{}/RP/{}'.format(experiment,here)
figure_dir = '../../../../figures/{}/RP/{}'.format(experiment,here)
if not os.path.exists(figure_dir):
    os.mkdir(figure_dir)


working_data = glob(os.path.join(working_dir,'*.csv'))
df = pd.concat([pd.read_csv(f) for f in working_data]).groupby(
        ['sub_name',
         'roi_name',
         'model_name',
         'condition',
         ]).mean().reset_index()
N = len(pd.unique(df['sub_name']))
alpha = 100


feature_dict = {'vgg19':'image2vec', 
                'densenet121':'image2vec', 
                'mobilenetv2_1':'image2vec',
                'Fast_Text':'Word2vec', 
                'Glove':'Word2vec', 
                'Word2Vec':'Word2vec',}
df['feature_type'] = df['model_name'].map(feature_dict)
hue_order = ['vgg19', 'densenet121', 'mobilenetv2_1',
             'Fast_Text', 'Glove', 'Word2Vec',
             ]
df = pd.concat([df[df['model_name'] == model_name] for model_name in hue_order])

temp = dict(
        F = [],
        df_nonimator = [],
        df_denominator = [],
        p = [],
        feature_type = [],
        condition = [],
        roi_name = [],
        )
for (feat,condition,roi),df_sub in df.groupby(['feature_type','condition','roi_name']):
    anova = ols('mean_variance ~ model_name',data = df_sub).fit()
    aov_table = sm.stats.anova_lm(anova,typ=2)
    print(aov_table)
    temp['F'].append(aov_table['F']['model_name'])
    temp['df_nonimator'].append(aov_table['df']['model_name'])
    temp['df_denominator'].append(aov_table['df']['Residual'])
    temp['p'].append(aov_table['PR(>F)']['model_name'])
    temp['feature_type'].append(feat)
    temp['condition'].append(condition)
    temp['roi_name'].append(roi)
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
anova_results = anova_results.sort_values(['roi_name','condition','feature_type'])




g = sns.catplot(x = 'roi_name',
                y = 'mean_variance',
                hue = 'model_name',
                row = 'condition',
                data = df,
                kind = 'bar',
                aspect = 4,
                )
g._legend.set_title('Encoding Models')
(g.set_axis_labels("","Mean Variance Explained")
  .set_titles("{row_name}")
  .set(ylim=(0,0.05)))
g.axes[-1][0].set_xticklabels(g.axes[-1][0].xaxis.get_majorticklabels(),
      rotation = 45,
      horizontalalignment = 'center')
g.axes[0][0].set(title = 'Read')
g.axes[1][0].set(title = 'Think')
k = {'image2vec':-0.25,
     'Word2vec':0.175}
j = 0.15
l = 0.0005
height = 0.045
for ax,condition in zip(g.axes.flatten(),['read','reenact']):
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    df_sub = anova_results[anova_results['condition'] == condition]
    for ii,((roi),df_star_sub) in enumerate(df_sub.groupby(['roi_name',])):
        for model,df_star_sub_model in df_star_sub.groupby(['feature_type']):
            ax.hlines(height,ii+k[model]-j,ii+k[model]+j)
            ax.vlines(ii+k[model]-j,height+l,height-l)
            ax.vlines(ii+k[model]+j,height+l,height-l)
            ax.annotate(df_star_sub_model['stars'].values[0],xy=(ii+k[model]-0.25,height + 0.002))
g.fig.suptitle("Comparison between Computer Vision and Word Embedding Models\nAverage Variance Explained\nN = {}, {} (alpha = {})\nBonforroni corrected for multiple one-way ANOVAs".format(
        N,"Ridge Regression",alpha),
            y = 1.15)

g.savefig(os.path.join(figure_dir,
                       'mean variance explained.png'),
        dpi = 400,
        bbox_inches = 'tight')

results = []
for image in ['densenet121','mobilenetv2_1','vgg19',]:
    for word in ['Fast_Text', 'Glove', 'Word2Vec',]:
        df_image = df[df['model_name'] == image]
        df_word = df[df['model_name'] == word]
        
        df_image = df_image.sort_values(['sub_name','roi_name','condition'])
        df_word = df_word.sort_values(['sub_name','roi_name','condition'])
        
        MV_diff = df_image['mean_variance'].values - df_word['mean_variance'].values
        BV_diff = df_image['best_variance'].values - df_word['best_variance'].values
        
        df_diff = df_image.copy()
        df_diff['mean_variance'] = MV_diff
        df_diff['best_variance'] = BV_diff
        df_diff['model_name'] = f"{image} - {word}"
        results.append(df_diff)
df_diff = pd.concat(results)

g = sns.catplot(x = 'roi_name',
                y = 'mean_variance',
                hue = 'model_name',
                row = 'condition',
                data = df_diff,
                kind = 'bar',
                aspect = 4,
                )
(g.set_axis_labels("","$\Delta$ Mean Variance Explained")
  .set_titles("{row_name}"))
g.axes[-1][0].set_xticklabels(g.axes[-1][0].xaxis.get_majorticklabels(),
      rotation = 45,
      horizontalalignment = 'center')
g.axes[0][0].set(title = 'Read')
g.axes[1][0].set(title = 'Think')
g.fig.suptitle('Difference of Computer Vision Models and Word Embedding Models\nBonforroni Corrected for multiple t-tests',
               y=1.05)

g.savefig(os.path.join(figure_dir,
                       'difference of variance explained by image2vec and word2vec.png'),
        dpi = 400,
        bbox_inches = 'tight')

df_condition = dict(
        roi = [],
        condition = [],
        ps_mean = [],
        ps_std = [],
        diff_mean = [],
        diff_std = [],)
for (roi,condition),df_sub in df.groupby(['roi_name','condition']):
    df_sub_img = df_sub[df_sub['feature_type'] == 'image2vec'].groupby(['sub_name']).mean().reset_index()
    df_sub_word = df_sub[df_sub['feature_type'] == 'Word2vec'].groupby(['sub_name']).mean().reset_index()
    a = df_sub_img['mean_variance'].values
    b = df_sub_word['mean_variance'].values
    ps = utils.resample_ttest_2sample(a,b,
                                      n_ps = 100,
                                      n_permutation = int(1e4),
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


df_diff_diff = dict(
        roi = [],
        ps_mean = [],
        ps_std = [],
        diff_mean = [],
        diff_std = [],)
for roi,df_sub in df_diff.groupby(['roi_name']):
    df_read = df_sub[df_sub['condition'] == 'read']#.sort_values(['sub'])
    df_reenact = df_sub[df_sub['condition'] == 'reenact']#.sort_values(['sub'])
    
    df_read = df_read.sort_values(['sub_name','roi_name','feature_type']).reset_index()
    df_reenact = df_reenact.sort_values(['sub_name','roi_name','feature_type']).reset_index()
    
    a = df_read['mean_variance'].values
    b = df_reenact['mean_variance'].values
    ps = utils.resample_ttest_2sample(a,b,
                                      n_ps = 100,
                                      n_permutation = int(1e4),
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

roi_order = pd.unique(df['roi_name'])
df_diff = pd.concat([df_diff[df_diff['roi_name'] == name] for name in roi_order])
df_diff_diff = pd.concat([df_diff_diff[df_diff_diff['roi'] == name] for name in roi_order])

g = sns.catplot(x = 'roi_name',
                y = 'mean_variance',
                hue = 'condition',
                data = df_diff,
                kind = 'violin',
                aspect = 4,
                **dict(split = True,
                       cut = 0,
                       inner = 'quartile'))
(g.set_axis_labels('ROIs','$\Delta$ Variance Explained'))
g._legend.set_title('Conditions')
g._legend.texts[0].set_text('Read')
g._legend.texts[1].set_text('Think')
g.axes[-1][0].set_xticklabels(g.axes[-1][0].xaxis.get_majorticklabels(),
      rotation = 45,
      horizontalalignment = 'center')
for ii in range(len(pd.unique(df_diff['roi_name']))):
    g.axes[0][0].annotate(df_diff_diff['star'].values[ii],
          xy = (ii,0.045))
g.fig.suptitle('Difference of Computer Vision Models and Word Embedding Models\nBonforroni Corrected for multiple t-tests',
               y=1.05)
g.savefig(os.path.join(figure_dir,
                       'difference of difference.png'),
        dpi = 400,
        bbox_inches = 'tight')










