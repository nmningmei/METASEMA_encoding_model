#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:39:17 2019

@author: nmei

This script systematically test the encoding model (Ridge regression) with different 
embedding features predicting the BOLD signal, within each small ROIs (15 in total)

"""

import os
import numpy  as np
import pandas as pd
from shutil import copyfile
copyfile("../../../utils.py","utils.py")
import utils
from glob import glob
from tqdm import tqdm
from sklearn.utils           import shuffle
from sklearn.model_selection import cross_validate
from sklearn                 import metrics,linear_model
from collections             import defaultdict

def AIC(y_true,y_pred,n_features):
    VE = metrics.r2_score(y_true,y_pred,multioutput='raw_values')
    VE[VE < 0] = np.nan
    VE_mean = np.nanmean(VE)
    aic = 2 * np.log(n_features) - 2 * np.log(VE_mean)
    return aic

def AIC_corrected(y_true,y_pred,n_features,n_observations):
    VE = metrics.r2_score(y_true,y_pred,multioutput='raw_values')
    VE[VE < 0] = np.nan
    VE_mean = np.nanmean(VE)
    aic = 2 * np.log(n_features) - 2 * np.log(VE_mean)
    aicc = aic + (2 * np.log(n_features))*(np.log(n_features) + 1) / (np.log(n_observations) - np.log(n_features) - 1)
    return aicc

def BIC(y_true,y_pred,n_features,n_observations):
    VE = metrics.r2_score(y_true,y_pred,multioutput='raw_values')
    VE = metrics.r2_score(y_true,y_pred,multioutput='raw_values')
    VE[VE < 0] = np.nan
    VE_mean = np.nanmean(VE)
    bic = np.log(n_features) * np.log(n_observations) - 2 * np.log(VE_mean)
    return bic

def CP(y_true,y_pred,n_features,n_observations):
    SSE = np.sum((y_true - y_pred)**2)
    SS  = np.sum((y_true - y_true.mean())**2)
    cp = SSE/SS - np.log(n_observations) + 2 * np.log(n_features)
    return cp

## parameters
experiment              = 'metasema'
working_dir             = '../../../../../{}/preprocessed_uncombined_with_invariant/'.format(experiment) # where the data locates
here                    = 'encoding_model_15_ROIs_old'
saving_dir              = '../../../../results/{}/RP/{}'.format(experiment,here) # where the outputs will go
if not os.path.exists(saving_dir):
    os.mkdir(saving_dir)

image2vec_dir           = '../../../../results/{}/img2vec_features_old'.format(experiment)
word2vec_dir            = '../../../../results/{}/word2vec_features'.format(experiment)


label_map               = dict(animal  =[1,0],
                               tool    =[0,1])

sub                     = '3439'# star means all subjects
average                 = True # averaging the trainig data
transfer                = False # do I do domain adaptation
print_train             = False # do I want to see the training process
concatenate             = False # specifically for domain adaptation
n_splits                = 300 # number of cross validation
n_jobs                  = 1 # 
alpha                   = 100

# get the data file names
working_fmri            = np.sort(glob(os.path.join(working_dir,'{}/*.npy'.format(sub))))
working_data            = np.sort(glob(os.path.join(working_dir,'{}/*.csv'.format(sub))))

# get the encoding model features
image2vec_vecs          = [pd.read_csv(f) for f in glob(os.path.join(image2vec_dir,'img2vec*.csv'))]
image2vec_names         = [f.split(' ')[-1].split('.')[0].replace("(","").replace(")","") for f in glob(os.path.join(image2vec_dir,'img2vec*.csv'))]
word2vec_vecs           = [pd.read_csv(f) for f in glob(os.path.join(word2vec_dir,'*.csv'))]
word2vec_names          = [f.split('/')[-1].split('.')[0] for f in glob(os.path.join(word2vec_dir,'*.csv'))]

for fmri,csv in zip(working_fmri,working_data):
    fmri,csv
    sub_name            = fmri.split('/')[-2]
    roi_name            = fmri.split('/')[-1].split('.')[0]
    
    # load the BOLD signal
    fmri_data_          = np.load(fmri)
    # load attributes come with the BOLD signal
    df_data_            = pd.read_csv(csv)
    
    for condition in ['read','reenact']:
        # pick condition
        idx_pick        = df_data_['context'] == condition
        fmri_data       = fmri_data_[idx_pick]
        df_data         = df_data_[idx_pick]
        
        # average over the volumes belong to the same trial
        if average:
            fmri_data,df_data   = utils.groupby_average(fmri_data,
                                                        df_data.reset_index(),
                                                        groupby = ['id'])
            df_data             = df_data.reset_index()
        
        # something we need for defining the cross validation method
        BOLD    = fmri_data.copy()
        targets = np.array([label_map[item] for item in df_data['targets'].values])
        groups  = df_data['words'].values
        
        # to remove the low variant voxels and standardize the BOLD signal
        from sklearn.feature_selection import VarianceThreshold
        from sklearn.preprocessing     import StandardScaler
        variance_threshold      = VarianceThreshold()
        BOLD                    = variance_threshold.fit_transform(BOLD)
        scaler                  = StandardScaler()
        BOLD                    = scaler.fit_transform(BOLD)
        
        # word embedding
        # convert the words into embedding features
        for word2vec_vec,word2vec_name in zip(word2vec_vecs,word2vec_names):
            csv_filename            = os.path.join(saving_dir,'{} {} {} {} {} {}.csv'.format(
                                        experiment,
                                        here,
                                        sub_name,
                                        roi_name,
                                        condition,
                                        word2vec_name))
            processed               = glob(os.path.join(saving_dir,'*.csv'))
            if csv_filename in processed: # don't repeat what have done
                print(csv_filename)
                pass
            else:
                if n_splits >= 100:
                    idxs_test       = utils.customized_partition(df_data,['id','words'],n_splits)
                    while utils.check_train_test_splits(idxs_test): # just in case
                        idxs_test   = utils.customized_partition(df_data,['id','words'],n_splits = n_splits)
                    idxs_train      = [shuffle(np.array([idx for idx in np.arange(df_data.shape[0]) if (idx not in idx_test)])) for idx_test in idxs_test]
    #                idxs_train      = [utils.check_train_balance(df_data,idx_train,list(label_map.keys())) for idx_train in tqdm(idxs_train)]
                    cv              = zip(idxs_train,idxs_test)
                else:
                    from sklearn.model_selection import GroupShuffleSplit
                    cv = GroupShuffleSplit(n_splits     = n_splits,
                                           test_size    = 0.2,
                                           random_state = 12345)
                    idxs_train,idxs_test = [],[]
                    for idx_train,idx_test in cv.split(BOLD,targets,groups=groups):
                        idxs_train.append(idx_train)
                        idxs_test.append(idx_test)
            
                embedding_features  = np.array([word2vec_vec[word.lower()] for word in df_data['words']])
                
                # define the encoding model
                encoding_model      = linear_model.Ridge(
                                        alpha                       = alpha,        # L2 penalty, higher means more weights are constrained to zero
                                        normalize                   = True,         # normalize the batch features
                                        random_state                = 12345,        # random seeding
                                        )
                # black box cross validation
                res                 = cross_validate(
                                        encoding_model,
                                        embedding_features,
                                        BOLD,
                                        groups                      = groups,
                                        cv                          = zip(idxs_train,idxs_test),
                                        n_jobs                      = n_jobs,
                                        return_estimator            = True,)
                # white box cross validation
                n_coef          = embedding_features.shape[1]
                n_obs           = int(embedding_features.shape[0] * 0.8)
                preds           = np.array([model.predict(embedding_features[idx_test]) for model,idx_test in zip(res['estimator'],idxs_test)])
                scores          = np.array([metrics.r2_score(BOLD[idx_test],y_pred,multioutput = 'raw_values') for idx_test,y_pred in zip(idxs_test,preds)])
                aic             = np.array([AIC(BOLD[idx_test],y_pred,n_coef) for idx_test,y_pred in zip(idxs_test,preds)])
                aicc            = np.array([AIC_corrected(BOLD[idx_test],y_pred,n_coef,n_obs) for idx_test,y_pred in zip(idxs_test,preds)])
                bic             = np.array([BIC(BOLD[idx_test],y_pred,n_coef,n_obs) for idx_test,y_pred in zip(idxs_test,preds)])
                cp              = np.array([CP(BOLD[idx_test],y_pred,n_coef,n_obs) for idx_test,y_pred in zip(idxs_test,preds)])
                mean_variance   = np.array([np.mean(temp[temp >= 0]) for temp in scores])
                try:
                    best_variance   = np.array([np.max(temp[temp >= 0]) for temp in scores])
                except:
                    best_variance   = mean_variance.copy()
                positive_voxels = np.array([np.sum(temp >= 0) for temp in scores])
                determinant     = np.array([1 - np.sum((BOLD[idx_test] - y_pred)**2) / np.sum((BOLD[idx_test] - BOLD[idx_test].mean(0))**2) for idx_test,y_pred in zip(idxs_test,preds)])
                print(word2vec_name,f'MV = {mean_variance.mean():.4f},AIC = {aic.mean():.2f},CP = {cp.mean():.2f}')
                # saving the results
                results                     = defaultdict()
                results['sub_name'         ]= [sub_name] *          n_splits
                results['roi_name'         ]= [roi_name] *          n_splits
                results['model_name'       ]= [word2vec_name] *     n_splits
                results['language'         ]= ['Spanish'] *         n_splits
                results['condition'        ]= [condition] *         n_splits
                results['fold'             ]= np.arange(n_splits) + 1
                results['positive voxels'  ]= positive_voxels
                results['mean_variance'    ]= mean_variance
                results['best_variance'    ]= best_variance
                results['determinant'      ]= determinant
                results['AIC'              ]= aic
                results['AICc'             ]= aicc
                results['BIC'              ]= bic
                results['CP'               ]= cp
                results_to_save             = pd.DataFrame(results)
                results_to_save.to_csv(csv_filename,index=False)
        
        # computer vision
        for img2vec_vec,img2vec_name in zip(image2vec_vecs,image2vec_names):
            csv_filename            = os.path.join(saving_dir,'{} {} {} {} {} {}.csv'.format(
                                        experiment,
                                        here,
                                        sub_name,
                                        roi_name,
                                        condition,
                                        img2vec_name))
            processed               = glob(os.path.join(saving_dir,'*.csv'))
            if csv_filename in processed: # don't repeat what have been done
                print(csv_filename)
                pass
            else:
                if n_splits >= 100:
                    idxs_test       = utils.customized_partition(df_data,['id','words'],n_splits)
                    while utils.check_train_test_splits(idxs_test): # just in case
                        idxs_test   = utils.customized_partition(df_data,['id','words'],n_splits = n_splits)
                    idxs_train      = [shuffle(np.array([idx for idx in np.arange(df_data.shape[0]) if (idx not in idx_test)])) for idx_test in idxs_test]
    #                idxs_train      = [utils.check_train_balance(df_data,idx_train,list(label_map.keys())) for idx_train in tqdm(idxs_train)]
                    cv              = zip(idxs_train,idxs_test)
                else:
                    from sklearn.model_selection import GroupShuffleSplit
                    cv = GroupShuffleSplit(n_splits     = n_splits,
                                           test_size    = 0.2,
                                           random_state = 12345)
                    idxs_train,idxs_test = [],[]
                    for idx_train,idx_test in cv.split(BOLD,targets,groups=groups):
                        idxs_train.append(idx_train)
                        idxs_test.append(idx_test)
                # convert words into embedding features
                embedding_features = np.array([img2vec_vec[word.lower()] for word in df_data['words']])
                # define the encoding model
                encoding_model  = linear_model.Ridge(
                                    alpha                       = alpha,        # L2 penalty, higher means more weights are constrained to zero
                                    normalize                   = True,         # normalize the batch features
                                    random_state                = 12345,        # random seeding
                                            )
                # black box cross validation
                res             = cross_validate(
                                    encoding_model,
                                    embedding_features,
                                    BOLD,
                                    groups                      = groups,
                                    cv                          = zip(idxs_train,idxs_test),
                                    n_jobs                      = n_jobs,
                                    return_estimator            = True,
                                    )
                # white box cross validation
                n_coef          = embedding_features.shape[1]
                n_obs           = int(embedding_features.shape[0] * 0.8)
                preds           = np.array([model.predict(embedding_features[idx_test]) for model,idx_test in zip(res['estimator'],idxs_test)])
                scores          = np.array([metrics.r2_score(BOLD[idx_test],y_pred,multioutput = 'raw_values') for idx_test,y_pred in zip(idxs_test,preds)])
                aic             = np.array([AIC(BOLD[idx_test],y_pred,n_coef) for idx_test,y_pred in zip(idxs_test,preds)])
                aicc            = np.array([AIC_corrected(BOLD[idx_test],y_pred,n_coef,n_obs) for idx_test,y_pred in zip(idxs_test,preds)])
                bic             = np.array([BIC(BOLD[idx_test],y_pred,n_coef,len(idx_test)) for idx_test,y_pred in zip(idxs_test,preds)])
                cp              = np.array([CP(BOLD[idx_test],y_pred,n_coef,n_obs) for idx_test,y_pred in zip(idxs_test,preds)])
                mean_variance   = np.array([np.mean(temp[temp >= 0]) for temp in scores])
                try:
                    best_variance   = np.array([np.max(temp[temp >= 0]) for temp in scores])
                except:
                    best_variance   = mean_variance.copy()
                positive_voxels = np.array([np.sum(temp >= 0) for temp in scores])
                determinant     = np.array([1 - np.sum((BOLD[idx_test] - y_pred)**2) / np.sum((BOLD[idx_test] - BOLD[idx_test].mean(0))**2) for idx_test,y_pred in zip(idxs_test,preds)])
                print(img2vec_name,f'MV = {mean_variance.mean():.4f},AIC = {aic.mean():.2f},CP = {cp.mean():.2f}')
                # saving the results
                results                     = defaultdict()
                results['sub_name'         ]= [sub_name] *          n_splits
                results['roi_name'         ]= [roi_name] *          n_splits
                results['model_name'       ]= [img2vec_name] *     n_splits
                results['language'         ]= ['Spanish'] *         n_splits
                results['condition'        ]= [condition] *         n_splits
                results['fold'             ]= np.arange(n_splits) + 1
                results['positive voxels'  ]= positive_voxels
                results['mean_variance'    ]= mean_variance
                results['best_variance'    ]= best_variance
                results['determinant'      ]= determinant
                results['AIC'              ]= aic
                results['AICc'             ]= aicc
                results['BIC'              ]= bic
                results['CP'               ]= cp
                results_to_save             = pd.DataFrame(results)
                results_to_save.to_csv(csv_filename,index=False)
                




























