#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:09:12 2019

@author: nmei
"""

import os
import numpy as np
import pandas as pd
import pickle
import re
import gc
os.chdir('../../../')
import utils
from glob                                          import glob
from tqdm                                          import tqdm
from mvpa2.mappers.fx                              import mean_group_sample
from sklearn.linear_model                          import Ridge
from sklearn.metrics                               import r2_score
from sklearn.feature_selection                     import VarianceThreshold
from sklearn.preprocessing                         import StandardScaler
#from sklearn.pipeline                              import make_pipeline


## parameters
experiment              = 'metasema'
working_dir             = '../../../../{}/preprocessed_uncombined_with_invariant/'.format(experiment) # where the data locates
here                    = '15 rois word2vec'
saving_dir              = '../results/{}/RP/{}'.format(experiment,here) # where the outputs will go
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
word2vec_dir            = '../results/{}/RP/word2vec features'.format(experiment)
label_map               = dict(animal  =[1,0],
                               tool    =[0,1])
sub                     = '*'# star means all subjects
average                 = False # averaging the trainig data
transfer                = False # do I do domain adaptation
print_train             = False # do I want to see the training process
concatenate             = False # specifically for domain adaptation
n_splits                = 100 # number of cross validation


# get the data file names
working_data            = glob(os.path.join(working_dir,'%s/*.pkl'%sub))
from sklearn.utils import shuffle
wording_data            = shuffle(working_data)
# load word2vec features
word2vec_vecs           = [pickle.load(open(f,'rb')) for f in glob(os.path.join(word2vec_dir,'*.p'))]
word2vec_names          = ['Fast_Text','Glove','Word2Vec']


# set random state
np.random.seed(12345)
for subject in working_data:
    sub_name             = subject.split('/')[-2]
    roi_name             = subject.split('/')[-1].split('.')[0]
    # disable garbage collection to speed up pickle
    gc.disable()
    with open(subject,'rb') as F:
        dataset_            = pickle.load(F)
        F.close()
    gc.enable();print('dataset loaded')
    output_dir          = sub_name
    for condition in ['read','reenact']:
        results = dict(
                    sub                 =[],
                    roi                 =[],
                    condition           =[],
                    mean_variance       =[],
                    best_variance       =[],
                    positive_values     =[],
                    model_name          =[],
                    determination       =[],
                    )
        dataset                 = dataset_[dataset_.sa.context == condition]# select one of the two conditions
        BOLD                    = dataset.samples.astype('float32')
        # to remove the low variant voxels
        variance_threshold      = VarianceThreshold()
        variance_threshold.fit(BOLD)
        scaler                  = StandardScaler()
        scaler.fit(variance_threshold.transform(BOLD))
        csv_filename            = os.path.join(saving_dir,'{} {} {} {} {}.csv'.format(
                                    experiment,
                                    here,
                                    sub_name,
                                    roi_name,
                                    condition))
        processed               = glob(os.path.join(saving_dir,'*.csv'))
        if csv_filename in processed:
            print(csv_filename)
            pass
        else:
            print('partitioning ...')
            idxs_train,idxs_test        = utils.get_train_test_splits(dataset,label_map,n_splits)
            
            if utils.check_train_test_splits(idxs_test):
                idxs_train,idxs_test    = utils.get_train_test_splits(dataset,label_map,n_splits)
            for word2vec_name,word2vec_features in zip(word2vec_names,word2vec_vecs):
                r_squares,scores = [],[]
                for fold, (idx_train,idx_test) in tqdm(enumerate(zip(idxs_train,idxs_test))):
                    if average:
                        tr = dataset[idx_train].get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
                    else:
                        tr = dataset[idx_train]
                    te = dataset[idx_test].get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
                    
    #                scaler          = utils.build_model_dictionary(n_jobs=4)['RandomForest + Linear-SVM']
    #                scaler.steps.pop(-1)
                    
                    
                    features_tr     = np.array([word2vec_features[word.lower()] for word in tr.sa.words])
                    BOLD_tr         = tr.samples.astype('float32')
    #                label_tr        = np.array([label_map[item] for item in tr.sa.targets])
                    features_te     = np.array([word2vec_features[word.lower()] for word in te.sa.words])
                    BOLD_te         = te.samples.astype('float32')
                    
                    BOLD_tr                 = variance_threshold.transform(BOLD_tr)
    #                BOLD_tr                 = scaler.fit_transform(BOLD_tr,label_tr[:,-1])
                    BOLD_tr                 = scaler.transform(BOLD_tr)
                    BOLD_te                 = variance_threshold.transform(BOLD_te)
                    BOLD_te                 = scaler.transform(BOLD_te)
                    
                    clf             = Ridge(alpha                       = 1e2,          # L2 penalty, higher means more weights are constrained to zero
                                            normalize                   = True,         # normalize the batch features
                                            random_state                = 12345,         # random seeding
                                            )
#                    clf             = make_pipeline(MinMaxScaler(),clf)
                    clf.fit(features_tr,BOLD_tr)
                    preds                   = clf.predict(features_te)
                    r_squares.append(1 - np.sum((BOLD_te - preds)**2) / np.sum(BOLD_te**2))
                    scores.append(r2_score(BOLD_te,preds,multioutput='raw_values'))
#                    print(np.sum(scores[-1] > 0))
                cut_score = np.mean(scores,0)
                cut_score[cut_score < 0] = 0
                results['sub'               ].append(sub_name)
                results['roi'               ].append(roi_name)
                results['model_name'        ].append(word2vec_name)
                results['condition'         ].append(condition)
                if not np.isnan(np.mean(cut_score[cut_score > 0])):
                    results['mean_variance'     ].append(np.mean(cut_score[cut_score > 0]))
                    results['best_variance'     ].append(np.max(cut_score[cut_score > 0]))
                else:
                    results['mean_variance'     ].append(0.)
                    results['best_variance'     ].append(0.)
                results['positive_values'   ].append(np.sum(cut_score > 0 ))
                results['determination'     ].append(np.mean(r_squares))
                try:
                    print('{},{},{},{},{},{}'.format(
                            sub_name,
                            roi_name,
                            condition,
                            word2vec_name,
                            np.mean(cut_score[cut_score > 0]),
                            np.max(cut_score[cut_score > 0])))
                except:
                    print('{},{},{},{},{},{}'.format(
                            sub_name,
                            roi_name,
                            condition,
                            word2vec_name,
                            0,
                            0))
            results = pd.DataFrame(results)
            results.to_csv(csv_filename,index=False)





















