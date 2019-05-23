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
from sklearn                                       import linear_model
#from sklearn.metrics                               import r2_score
#from sklearn.feature_selection                     import VarianceThreshold
#from sklearn.preprocessing                         import StandardScaler
#from sklearn.pipeline                              import make_pipeline
from sklearn.utils                                 import shuffle


## parameters
experiment              = 'metasema'
working_dir             = '../../../../{}/preprocessed_with_invariant/'.format(experiment) # where the data locates
here                    = '7 rois image2vec'
saving_dir              = '../results/{}/RP/{}'.format(experiment,here) # where the outputs will go
if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)
image2vec_dir           = '../results/{}/img2vec_features'.format(experiment)
label_map               = dict(animal  =[1,0],
                               tool    =[0,1])
rois                    = {1:'IPL',
                           2:'LTL',
                           3:'FFG & PHG',
                           4:'dmPFC',
                           5:'IFG', 
                           6:'vmPFC',
                           7:'PCG & Precun'}
sub                     = '*'# star means all subjects
average                 = False # averaging the trainig data
transfer                = False # do I do domain adaptation
print_train             = False # do I want to see the training process
concatenate             = False # specifically for domain adaptation
n_splits                = 100 # number of cross validation

# get the data file names
working_data            = glob(os.path.join(working_dir,'%s/roi*.pkl'%sub))
working_data = shuffle(working_data)
# load word2vec features
image2vec_vecs          = [pd.read_csv(f) for f in glob(os.path.join(image2vec_dir,'img2vec*.csv'))]
image2vec_names         = [f.split(' ')[-1].split('.')[0].replace("(","").replace(")","") for f in glob(os.path.join(image2vec_dir,'img2vec*.csv'))]
# set random state
np.random.seed(12345)
for subject in working_data:
    sub_name            = subject.split('/')[-2]
    n_roi               = int(re.findall('\d+',subject.split('/')[-1].split('.')[0])[0])
    roi_name            = rois[n_roi]
    # disable garbage collection to speed up pickle
    gc.disable()
    with open(subject,'rb') as F:
        dataset_            = pickle.load(F)
        F.close()
    gc.enable();print('dataset loaded')
    output_dir          = sub_name
    for condition in ['read','reenact',]:
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
            for image2vec_name,image2vec_features in zip(image2vec_names,image2vec_vecs):
                r_squares,scores = [],[]
                for fold, (idx_train,idx_test) in tqdm(enumerate(zip(idxs_train,idxs_test))):
                    if average:
                        tr = dataset[idx_train].get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
                    else:
                        tr = dataset[idx_train]
                    te = dataset[idx_test].get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
                    
                    features_tr     = np.array([image2vec_features[word.lower()] for word in tr.sa.words])
                    BOLD_tr         = tr.samples.astype('float32')
                    features_te     = np.array([image2vec_features[word.lower()] for word in te.sa.words])
                    BOLD_te         = te.samples.astype('float32')
                    clf             = linear_model.Ridge(
                                            alpha                       = 1e2,          # L2 penalty, higher means more weights are constrained to zero
                                            normalize                   = True,         # normalize the batch features
                                            random_state                = 12345,        # random seeding
                                            )
                    r_squares,scores,pipeline = utils.regression_CV(
                                    BOLD_tr,
                                    BOLD_te,
                                    features_tr,
                                    features_te,
                                    clf,
                                    r_squares,
                                    scores,
                                    scaler = None,
                                    variance_threshold = None,
                                    print_train = False,
                                    )
                cut_score = np.mean(scores,0)
                cut_score[cut_score < 0] = 0
                results['sub'               ].append(sub_name)
                results['roi'               ].append(roi_name)
                results['model_name'        ].append(image2vec_name)
                results['condition'         ].append(condition)
                try:
                    results['best_variance'     ].append(np.max(cut_score[cut_score > 0]))
                    results['mean_variance'     ].append(np.mean(cut_score[cut_score > 0]))
                except:
                    results['mean_variance'     ].append(0.)
                    results['best_variance'     ].append(0.)
                results['positive_values'   ].append(np.sum(cut_score > 0 ))
                results['determination'     ].append(np.mean(r_squares))
                try:
                    print('{},{},{},{},\n with {} voxels: {},{}'.format(
                            sub_name,
                            roi_name,
                            condition,
                            image2vec_name,
                            results['positive_values'][-1],
                            np.mean(cut_score[cut_score > 0]),
                            np.max(cut_score[cut_score > 0])))
                except:
                    print('{},{},{},{},\n with {} voxels: {},{}'.format(
                            sub_name,
                            roi_name,
                            condition,
                            image2vec_name,
                            0,
                            0,
                            0))
            results = pd.DataFrame(results)
            results.to_csv(csv_filename,index=False)





















