#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:36:51 2019

@author: nmei
"""

import os
import numpy  as np
import pandas as pd
from shutil import copyfile
copyfile("../../../utils.py","utils.py")
import utils
from glob import glob
from tqdm import tqdm
from sklearn.utils             import shuffle
from sklearn.model_selection   import cross_validate,cross_val_predict
from sklearn                   import metrics,linear_model
from sklearn.feature_selection import mutual_info_regression
from sklearn.base              import BaseEstimator, ClassifierMixin
from sklearn.multioutput       import MultiOutputRegressor
from collections               import defaultdict
from functools                 import partial

func_ = partial(metrics.r2_score,multioutput = 'uniform_average')
func_.__name__ = 'r2'
scorer = metrics.make_scorer(func_)
from joblib import Parallel,delayed
import gc
gc.collect()
def mutual_func(X,y):
    return mutual_info_regression(X,y,discrete_features = False,
                                  random_state = 12345)
class mutualInformation(BaseEstimator,ClassifierMixin):
    """fit the mutual information for all folds"""
    from sklearn.feature_selection import mutual_info_regression
    def __init__(self, intValue=0, stringParam="defaultValue", otherParam=None):
        """
        Called when initializing the classifier
        """
        self.intValue = intValue
        self.stringParam = stringParam
        
        # THIS IS WRONG! Parameters should have same name as attributes
        self.differentParam = otherParam 
    
    def fit(self,X,y):
        self.mutual_fit = mutual_info_regression(X,y,discrete_features = False,
                                        random_state = 12345)
        return self

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
here                    = 'encoding_model_15_ROIs_k-best'
saving_dir              = '../../../../results/{}/RP/{}'.format(experiment,here) # where the outputs will go
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

image2vec_dir           = '../../../../results/{}/img2vec_features'.format(experiment)
word2vec_dir            = '../../../../results/{}/word2vec_features'.format(experiment)


label_map               = dict(animal  =[1,0],
                               tool    =[0,1])

sub                     = '123'# star means all subjects
average                 = True # averaging the trainig data
transfer                = False # do I do domain adaptation
print_train             = False # do I want to see the training process
concatenate             = False # specifically for domain adaptation
n_splits                = 100 # number of cross validation
n_jobs                  = 20 # 
alpha                   = 100

# get the data file names
working_fmri            = np.sort(glob(os.path.join(working_dir,'{}/*.npy'.format(sub))))
working_data            = np.sort(glob(os.path.join(working_dir,'{}/*.csv'.format(sub))))

# get the encoding model features
image2vec_vecs          = [pd.read_csv(f) for f in glob(os.path.join(image2vec_dir,'img2vec*.csv'))]
image2vec_names         = [f.split(' ')[-1].split('.')[0].replace("(","").replace(")","") for f in glob(os.path.join(image2vec_dir,'img2vec*.csv'))]

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
        
        # computer vision
        for img2vec_vec,img2vec_name in zip(image2vec_vecs,image2vec_names):
            csv_filename            = os.path.join(saving_dir,'{} {} {} {} {} {}.csv'.format(
                                        experiment,
                                        here,
                                        sub_name,
                                        roi_name,
                                        condition,
                                        img2vec_name,))
            if n_splits > 100:
                idxs_test       = utils.customized_partition(df_data,['id','words'],n_splits)
                while utils.check_train_test_splits(idxs_test): # just in case
                    idxs_test   = utils.customized_partition(df_data,['id','words'],n_splits = n_splits)
                idxs_train      = [shuffle(np.array([idx for idx in np.arange(df_data.shape[0]) if (idx not in idx_test)])) for idx_test in idxs_test]
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
#            n_jobs = 12
            mutual = []
            for idx_train in tqdm(idxs_train):
                X,y = embedding_features[idx_train],BOLD[idx_train]
                gc.collect()
                sc = Parallel(n_jobs = n_jobs,verbose = 0)(delayed(mutual_func)(**{
                    'X':X,
                    'y':y[:,ii],}) for ii in range(y.shape[1]))
                sc_ave = np.mean(sc,0)
                sc_ave = pd.Series(sc_ave)
                mutual.append(sc_ave)
            p = []
            for k in np.concatenate([np.arange(100,embedding_features.shape[1],2),[embedding_features.shape[1]]]):
                temp1,temp2 = [],[]
                for idx_train,idx_test,mutual_info in zip(idxs_train,idxs_test,mutual):
                    temp1.append((idx_train,np.array(list(mutual_info.nlargest(k).index))))
                    temp2.append((idx_test,np.array(list(mutual_info.nlargest(k).index))))
                idxs_train_,idxs_test_ = temp1,temp2
                # black box cross validation
                
                def fit(embedding_features,BOLD,idx_train):
                    encoding_model  = linear_model.Ridge(
                                alpha                       = alpha,        # L2 penalty, higher means more weights are constrained to zero
                                normalize                   = True,         # normalize the batch features
                                random_state                = 12345,        # random seeding
                                        )
                    encoding_model.fit(embedding_features[idx_train[0]][:,idx_train[1]],BOLD[idx_train[0]])
                    return encoding_model
                gc.collect()
                res = Parallel(n_jobs = n_jobs,verbose = 0)(delayed(fit)(**{
                        "embedding_features":embedding_features,
                        "BOLD":BOLD,
                        "idx_train":idx_train,}) for idx_train in idxs_train_)
                
                # white box cross validation
                n_coef          = k#embedding_features.shape[1]
                n_obs           = np.mean([len(idx_train) for idx_train in idxs_train])
                preds           = np.array([model.predict(embedding_features[idx_test[0]][:,idx_test[1]]) for model,idx_test in zip(res,idxs_test_)])
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
                print(img2vec_name,f'{k}best,{mean_variance.mean():.4f},{aic.mean():.2f},{cp.mean():.2f}')
                p.append([k,mean_variance.mean(),aic.mean()])
                # saving the results
                results                     = defaultdict()
                results['sub_name'         ]= [sub_name] *          n_splits
                results['roi_name'         ]= [roi_name] *          n_splits
                results['model_name'       ]= [img2vec_name] *      n_splits
                results['language'         ]= ['Spanish'] *         n_splits
                results['condition'        ]= [condition] *         n_splits
                results['k'                ]= [k]*                  n_splits
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
                results_to_save.to_csv(csv_filename.replace('.csv',f' {k}best.csv'),index=False)