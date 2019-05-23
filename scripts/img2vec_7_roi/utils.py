#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:55:43 2018

@author: nmei
"""

import numpy as np
import pandas as pd
import pickle

from sklearn.metrics                               import roc_auc_score,roc_curve
from sklearn.metrics                               import (
                                                           classification_report,
                                                           matthews_corrcoef,
                                                           confusion_matrix,
                                                           f1_score,
                                                           log_loss,
                                                           r2_score
                                                           )

from sklearn.preprocessing                         import (MinMaxScaler,
                                                           OneHotEncoder,
                                                           FunctionTransformer,
                                                           StandardScaler)

from sklearn.pipeline                              import make_pipeline
from sklearn.ensemble.forest                       import _generate_unsampled_indices
from sklearn.utils                                 import shuffle
from sklearn.svm                                   import SVC,LinearSVC
from sklearn.calibration                           import CalibratedClassifierCV
from sklearn.decomposition                         import PCA
from sklearn.dummy                                 import DummyClassifier
from sklearn.feature_selection                     import (SelectFromModel,
                                                           SelectPercentile,
                                                           VarianceThreshold,
                                                           mutual_info_classif,
                                                           f_classif,
                                                           chi2,
                                                           f_regression,
                                                           GenericUnivariateSelect)
from sklearn.model_selection                       import (StratifiedShuffleSplit,
                                                           cross_val_score)
from sklearn.ensemble                              import RandomForestClassifier,BaggingClassifier,VotingClassifier
from sklearn.neural_network                        import MLPClassifier
from xgboost                                       import XGBClassifier
from itertools                                     import product,combinations
from sklearn.base                                  import clone
from sklearn.neighbors                             import KNeighborsClassifier
from sklearn.tree                                  import DecisionTreeClassifier
from collections                                   import OrderedDict

from scipy                                         import stats
from collections                                   import Counter

try:
    #from mvpa2.datasets.base                           import Dataset
    from mvpa2.mappers.fx                              import mean_group_sample
    #from mvpa2.measures                                import rsa
    #from mvpa2.measures.searchlight                    import sphere_searchlight
    #from mvpa2.base.learner                            import ChainLearner
    #from mvpa2.mappers.shape                           import TransposeMapper
    #from mvpa2.generators.partition                    import NFoldPartitioner
except:
    print('pymvpa is not installed')
try:
    from tqdm import tqdm
except:
    print('why is tqdm not installed?')
def resample_ttest(x,baseline = 0.5,n_ps = 100,n_permutation = 5000,one_tail = False):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    Inputs:
    ----------
    x: numpy array vector, the data that is to be compared
    baseline: the single point that we compare the data with
    n_ps: number of p values we want to estimate
    n_permutation: number of permutation we want to perform, the more the further it could detect the strong effects, but it is so unnecessary
    one_tail: whether to perform one-tailed comparison
    """
    import numpy as np
    experiment      = np.mean(x) # the mean of the observations in the experiment
    experiment_diff = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution
    # newexperiment = np.mean(experiment_diff) # just look at the new mean and make sure it is at the baseline
    # simulate/bootstrap null hypothesis distribution
    # 1st-D := number of sample same as the experiment
    # 2nd-D := within one permutation resamping, we perform resampling same as the experimental samples,
    # but also repeat this one sampling n_permutation times
    # 3rd-D := repeat 2nd-D n_ps times to obtain a distribution of p values later
    temp            = np.random.choice(experiment_diff,size=(x.shape[0],n_permutation,n_ps),replace=True)
    temp            = temp.mean(0)# take the mean over the sames because we only care about the mean of the null distribution
    # along each row of the matrix (n_row = n_permutation), we count instances that are greater than the observed mean of the experiment
    # compute the proportion, and we get our p values
    
    if one_tail:
        ps = (np.sum(temp >= experiment,axis=0)+1.) / (n_permutation + 1.)
    else:
        ps = (np.sum(np.abs(temp) >= np.abs(experiment),axis=0)+1.) / (n_permutation + 1.)
    return ps
def resample_ttest_2sample(a,b,n_ps=100,n_permutation=5000,one_tail=False,match_sample_size = True,):
    # when the N is matched just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,baseline=0,n_ps=n_ps,n_permutation=n_permutation,one_tail=one_tail)
        return ps
    else: # when the N is not matched
        difference              = np.mean(a) - np.mean(b)
        concatenated            = np.concatenate([a,b])
        np.random.shuffle(concatenated)
        temp                    = np.zeros((n_permutation,n_ps))
        # the next part of the code is to estimate the "randomized situation" under the given data's distribution
        # by randomized the items in each group (a and b), we can compute the chance level differences
        # and then we estimate the probability of the chance level exceeds the true difference 
        # as to represent the "p value"
        try:
            iterator            = tqdm(range(n_ps),desc='ps')
        except:
            iterator            = range(n_ps)
        for n_p in iterator:
            for n_permu in range(n_permutation):
                idx_a           = np.random.choice(a    = [0,1],
                                                   size = (len(a)+len(b)),
                                                   p    = [float(len(a))/(len(a)+len(b)),
                                                           float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                idx_b           = np.logical_not(idx_a)
                d               = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                if np.isnan(d):
                    idx_a       = np.random.choice(a        = [0,1],
                                                   size     = (len(a)+len(b)),
                                                   p        = [float(len(a))/(len(a)+len(b)),
                                                               float(len(b))/(len(a)+len(b))]
                                                   ).astype(np.bool)
                    idx_b       = np.logical_not(idx_a)
                    d           = np.mean(concatenated[idx_a]) - np.mean(concatenated[idx_b])
                temp[n_permu,n_p] = d
        if one_tail:
            ps = (np.sum(temp >= difference,axis=0)+1.) / (n_permutation + 1.)
        else:
            ps = (np.sum(np.abs(temp) >= np.abs(difference),axis=0)+1.) / (n_permutation + 1.)
        return ps

class MCPConverter(object):
    import statsmodels as sms
    """
    https://gist.github.com/naturale0/3915e2def589553e91dce99e69d138cc
    https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
    input: array of p-values.
    * convert p-value into adjusted p-value (or q-value)
    """
    def __init__(self, pvals, zscores = None):
        self.pvals                    = pvals
        self.zscores                  = zscores
        self.len                      = len(pvals)
        if zscores is not None:
            srted                     = np.array(sorted(zip(pvals.copy(), zscores.copy())))
            self.sorted_pvals         = srted[:, 0]
            self.sorted_zscores       = srted[:, 1]
        else:
            self.sorted_pvals         = np.array(sorted(pvals.copy()))
        self.order                    = sorted(range(len(pvals)), key=lambda x: pvals[x])
    
    def adjust(self, method           = "holm"):
        import statsmodels as sms
        """
        methods = ["bonferroni", "holm", "bh", "lfdr"]
         (local FDR method needs 'statsmodels' package)
        """
        if method is "bonferroni":
            return [np.min([1, i]) for i in self.sorted_pvals * self.len]
        elif method is "holm":
            return [np.min([1, i]) for i in (self.sorted_pvals * (self.len - np.arange(1, self.len+1) + 1))]
        elif method is "bh":
            p_times_m_i = self.sorted_pvals * self.len / np.arange(1, self.len+1)
            return [np.min([p, p_times_m_i[i+1]]) if i < self.len-1 else p for i, p in enumerate(p_times_m_i)]
        elif method is "lfdr":
            if self.zscores is None:
                raise ValueError("Z-scores were not provided.")
            return sms.stats.multitest.local_fdr(abs(self.sorted_zscores))
        else:
            raise ValueError("invalid method entered: '{}'".format(method))
            
    def adjust_many(self, methods = ["bonferroni", "holm", "bh", "lfdr"]):
        if self.zscores is not None:
            df = pd.DataFrame(np.c_[self.sorted_pvals, self.sorted_zscores], columns=["p_values", "z_scores"])
            for method in methods:
                df[method] = self.adjust(method)
        else:
            df = pd.DataFrame(self.sorted_pvals, columns=["p_values"])
            for method in methods:
                if method is not "lfdr":
                    df[method] = self.adjust(method)
        return df
def binarize(labels):
    """
    By Usman
    """
    try:
        if len(np.unique(labels)) > 2: raise ValueError
        return (labels == 1).astype(int)
    except:
        exit("more than 2 classes, fix it!")
def load_preprocessed(pre_fil):
    """
    By Usman
    
    Load all preprocessed data for a specific roi (stored @ pre_fil).

    Inputs:
    -----------
    pre_fil: str

    Returns:
    --------
    feat_ds: pymvpa2 Dataset
    """
    import gc
    # disable garbage collection to speed up pickle
    gc.disable()
    with open(pre_fil, 'rb') as f:
        feat_ds = pickle.load(f)

    feat_ds.sa['id'] = feat_ds.sa.id.astype(int)
#    feat_ds.sa['targets'] = binarize(feat_ds.sa.targets.astype(float).astype(int))

    return feat_ds

def get_blocks(dataset__,label_map,key_type='labels'):
    """
    # use ids, chunks,and labels to make unique blocks of the pre-average dataset, because I don't want to 
    # average the dataset until I actually want to, but at the same time, I want to balance the data for 
    # both the training and test set.
    """
    ids                     = dataset__.sa.id.astype(int)
    chunks                  = dataset__.sa.chunks
    words                   = dataset__.sa.words
    if key_type == 'labels':
        try: # in metasema
            labels              = np.array([label_map[item] for item in dataset__.sa.targets])[:,-1]
        except:# not in metasema
            labels              = np.array([label_map[item] for item in dataset__.sa.targets])
        
    elif key_type == 'words':
        labels              = np.array([label_map[item] for item in dataset__.sa.words])
    sample_indecies         = np.arange(len(labels))
    blocks                  = [np.array([ids[ids             == target],
                                         chunks[ids          == target],
                                         words[ids           == target],
                                         labels[ids          == target],
                                         sample_indecies[ids == target]
                                         ]) for target in np.unique(ids)]
    block_labels            = np.array([np.unique(ll[-2]) for ll in blocks]).ravel()
    return blocks,block_labels
def customized_partition(dataset__,label_map):
    """
    To customize the random partitioning, this function would randomly select instance of volumes
    that correspond to unique words used in an experiment from different scanning blocks to form
    the test set. 
    By doing so, we create a quasi-leave-one-block-out cross-validation
    """
    unique_words    = np.unique(dataset__.sa.words)
    unique_chunks   = np.unique(dataset__.sa.chunks)
    try: # in metasema
        labels              = np.array([label_map[item] for item in dataset__.sa.targets])[:,-1]
    except:# not in metasema
        labels              = np.array([label_map[item] for item in dataset__.sa.targets])
    words           = dataset__.sa.words
    chunks          = dataset__.sa.chunks
    blocks,block_labels     = get_blocks(dataset__,label_map,key_type='labels')
    sample_indecies         = np.arange(len(labels))
    test            = []
    check           = []
    for n in range(int(1e4)):
        # randomly pick one of the scanning blocks
        random_chunk    = np.random.choice(unique_chunks,size=1,replace=False)[0]
        # get indecis of the words in this block from the whole dataset
        working_words   = words[chunks == random_chunk]
        # variable "blocks" is a list. Each item contains: id, chunk, word, label,sample indecies
        # and we only need to access the "chunk"
        working_block   = [block for block in blocks if (int(np.unique(block[1])[0]) == random_chunk)]
        # picke a word randomly from the words in the picked block
        random_word     = np.random.choice(working_words,size=1,replace=False)[0]
        if random_word not in check: # we need to check if the word has been selected before
            for block in working_block:
                if (np.unique(block[2])[0] == random_word) and (random_word not in check):
                    test.append(block[-1].astype(int))
                    check.append(block[2][0])
#                    print(test,check)
                if len(check) == len(unique_words):
                    break
            if len(check) == len(unique_words):
                break
        if len(check) == len(unique_words):
            break
    test = np.concatenate(test,0).flatten()
    train = np.array([idx for idx in sample_indecies if (idx not in test)])
    return train,test
def get_train_test_splits(dataset,label_map,n_splits):
    idxs_train,idxs_test = [],[]
    np.random.seed(12345)
    used_test = []
    fold = -1
    for abc in range(int(1e3)):
#        print('paritioning ...')
        idx_train,idx_test = customized_partition(dataset,label_map,)
        current_sample = np.sort(idx_test)
        candidates = [np.sort(item) for item in used_test if (len(item) == len(idx_test))]
        if any([np.sum(current_sample == item) == len(current_sample) for item in candidates]):
            pass
        else:
            fold += 1
            used_test.append(idx_test)
            idxs_train.append(idx_train)
            idxs_test.append(idx_test)
#            print('done, get fold {}'.format(fold))
            if fold == n_splits - 1:
                break
    return idxs_train,idxs_test
def check_train_test_splits(idxs_test):
    temp = []
    for ii,item1 in enumerate(idxs_test):
        for jj,item2 in enumerate(idxs_test):
            if not ii == jj:
                if len(item1) == len(item2):
                    sample1 = np.sort(item1)
                    sample2 = np.sort(item2)
                    
                    temp.append(np.sum(sample1 == sample2) == len(sample1))
    temp = np.array(temp)
    return any(temp)
def check_train_balance(dataset__,idx_train,keys):
    Counts = dict(Counter(dataset__[idx_train].sa.targets))
    if np.abs(Counts[keys[0]] - Counts[keys[1]]) > 2:
        if Counts[keys[0]] > Counts[keys[1]]:
            key_major = keys[0]
            key_minor = keys[1]
        else:
            key_major = keys[1]
            key_minor = keys[0]
        
        ids_major = dataset__[idx_train].sa.id[dataset__[idx_train].sa.targets == key_major]
        
        for n in range(len(idx_train)):
            random_pick = np.random.choice(np.unique(ids_major),size = 1)[0]
            # print(random_pick,np.unique(ids_major))
            idx_train = np.array([item for item,id_temp in zip(idx_train,dataset__[idx_train].sa.id) if (id_temp != random_pick)])
            ids_major = np.array([item for item in ids_major if (item != random_pick)])
            new_counts = dict(Counter(dataset__[idx_train].sa.targets))
            if np.abs(new_counts[keys[0]] - new_counts[keys[1]]) > 4:
                if new_counts[keys[0]] > new_counts[keys[1]]:
                    key_major = keys[0]
                    key_minor = keys[1]
                else:
                    key_major = keys[1]
                    key_minor = keys[0]
                
                ids_major = dataset__[idx_train].sa.id[dataset__[idx_train].sa.targets == key_major]
            elif np.abs(new_counts['1.0'] - new_counts['2.0']) < 4:
                break
    return idx_train
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold         = roc_curve(target, predicted)
    i                           = np.arange(len(tpr)) 
    roc                         = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t                       = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def cross_CV( source_set,           # pymvpa dataset - source dataset
              target_set,           # pymvpa dataset - target dataset 
              idx_train_source,     # indecies of the training set in the source dataset
              idx_train_target,     # indecies of the training set in the target dataset
              idx_test_target,      # indecies of the test set in the target dataset
              pipeline,             # pipeline that contains a scaler, a feature processor, and an estimator
              label_map,            # one hot label map used in deep learning models
              results,              # the dictionary for storing the results
              sub_name,             # subject name
              fold,                 # cross validation fold
              model_name,           # name of the pipeline
              language,             # language of the stimuli
              condition_target,
              condition_source,
              roi_name = None,      # not None if we work on selected ROI data
              average = True,       # averaging the trainig data
              transfer = False,     # do I do domain adaptation
              print_train = False,
              concatenate = False,
              flip = False,
              to_save = True,
              MultiOutput = False,
              ):
    """
    unified pipeline for decoding across many experiments and conditions 
    
    Inputs:
    --------
    source_set:         pymvpa dataset, used to get the primary training dataset
    target_set:         pymvpa dataset, used to get the secondary training dataset and/or the testing dataset
    idx_train_source:   indecies of the training set in the source dataset
    idx_train_target:   indecies of the training set in the target dataset
    idx_test_target:    indecies of the test set in the target dataset
    pipeline:           scitkit learn pipeline/estimator, predict_proba required
    label_map:          a dictionary for creating the one hot labels for cross validation
    results:            a dictionary for storing the results and other attributes
    sub_name:           subject's coded name
    fold:               a counting unit
    model_name:         name of the pipeline
    language:           language in which the stimuli is
    condition_target:   condition/conscious in which is the subject is for the target data
    condition_source:   condition/conscious in which is the subject is for the source data
    roi_name:           name of the ROI
    average:            control whether to average the training set/s
    transfer:           indicator of whether this pipeline is in transfer learning/domain adaptation
    print_train:        to print out the training informations
    concatenate:        domain adaptation specific, whether to combine the primary and secondary training dataset
    flip:               doubling the size of the training dataset by flipping the training set matrix left-right, and concatenate them
    to_save:            control whether to save the outputs
    ----------------------------------------------------------------------------------------------------------
    Returns:
    --------
    results:            a dictionary for storing the results and other attributes
    score_baseline1:    roc auc
    score_baseline2:    matthews correlation coefficient
    score_baseline3:    f1 score
    score_baseline4:    log loss value
    """
    # select the training and testing split again to make sure we do the right cv
    tr_source           = source_set[idx_train_source]
    tr_target           = target_set[idx_train_target]
    te                  = target_set[idx_test_target]
    # average the test data for the baseline model
    try:
        te              = te.get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
    except:
        print('it is socialcon, and it is already been averaged')
    # average the train data for baseline to improve the signal to noise ratio
    if average:
        tr_source       = tr_source.get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
        tr_target       = tr_target.get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
    # get the numpy arrays
    X_train_source      = tr_source.samples.astype('float32')
    X_train_target      = tr_target.samples.astype('float32')
    X_test              = te.samples.astype('float32')
    # transfer the string labels to integers
    y_train_source      = np.array([label_map[item] for item in tr_source.targets])
    y_train_target      = np.array([label_map[item] for item in tr_target.targets])
    y_test              = np.array([label_map[item] for item in te.targets])
    
    
    # whether to combine the primary and secondary datasets
    if concatenate:
        X_train         = np.concatenate([X_train_source,X_train_target])
        y_train         = np.concatenate([y_train_source,y_train_target])
    else:
        X_train         = X_train_source
        y_train         = y_train_source
    # whether to double the size of training dataset
    if flip:
        X_train_flip    = np.fliplr(X_train)
        X_train         = np.concatenate([X_train,X_train_flip])
        y_train         = np.concatenate([y_train,y_train])
    # check the size of the labels, if it does NOT have 2 columns, make it so
    if y_train.shape[-1]== 2:
        labels_train    = y_train
        labels_test     = y_test
    else:
        labels_train    = OneHotEncoder().fit_transform(y_train.reshape(-1,1)).toarray()
        labels_test     = OneHotEncoder().fit_transform(y_test.reshape(-1,1)).toarray()
    
    if print_train:
        print('train on {} samples, test on {} samples by {}'.format(
                X_train.shape[0],
                te.shape[0],
                model_name))
    # shuffle the order of trials for both features and targets
    np.random.seed(12345)
    X_train,labels_train    = shuffle(X_train,labels_train)
    # train the classification pipeline
    if MultiOutput:
        pipeline.fit(X_train,labels_train)
        
        pred_ = np.array(pipeline.predict_proba(X_test))[:,:,-1]
        if print_train:
            print('test labels',labels_test,'prediction',pred_)
        score_baseline1         = np.array([roc_auc_score(a,b) for a,b in zip(labels_test.T,pred_)])
        threshold_              = np.array([Find_Optimal_Cutoff(a,b) for a,b in zip(labels_test.T,pred_)])
        score_baseline2         = np.array([matthews_corrcoef(a,b>t) for a,b,t in zip(labels_test.T,pred_,threshold_)])
        score_baseline3         = np.array([f1_score(a,b>t) for a,b,t in zip(labels_test.T,pred_,threshold_)])
        score_baseline4         = np.array([log_loss(a,b) for a,b in zip(labels_test.T,pred_)])
        cm                      = np.array([confusion_matrix(a,b>t) for a,b,t in zip(labels_test.T,pred_,threshold_)])
        tn, fp, fn, tp          = cm.mean(0).flatten()
    else:
        pipeline.fit(X_train,labels_train[:,-1])
        if print_train:
            print('training score = {:.2}'.format(pipeline.score(X_train,labels_train[:,-1])))
        # provide probabilistic predictions on the test data
        pred_                   = pipeline.predict_proba(X_test)
        if print_train:
            print('test labels',labels_test[:,-1],'prediction',pred_[:,-1])
        """"
        Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
         For binary y_true, y_score is supposed to be the score of the class with greater label.
        """
        score_baseline1         = roc_auc_score(labels_test[:,-1],pred_[:,-1])
        threshold_              = 0.5#np.array([Find_Optimal_Cutoff(labels_test[:,ii],pred_[:,ii])[0] for ii in range(labels_test.shape[-1])])
        score_baseline2         = matthews_corrcoef(labels_test[:,-1],pred_[:,-1]>threshold_)
        score_baseline3         = f1_score(labels_test[:,-1],pred_[:,-1]>threshold_, average='weighted',labels=np.unique(pred_[:,-1]>threshold_))
        score_baseline4         = log_loss(labels_test[:,-1],pred_[:,-1])
        cm                      = confusion_matrix(labels_test[:,-1],pred_[:,-1]>threshold_).ravel()
        tn, fp, fn, tp          = cm
        if print_train:
            print(classification_report(labels_test[:,-1],pred_[:,-1]>threshold_))
    if to_save:
        if MultiOutput:
            results['tn'                    ].append(tn)
            results['fp'                    ].append(fp)
            results['fn'                    ].append(fn)
            results['tp'                    ].append(tp)
            results['fold'                  ].append(fold+1)
            results['sub'                   ].append(sub_name)
            if roi_name is not None:
                results['roi'               ].append(roi_name)
            results['model'                 ].append(model_name)
            results['roc_auc'               ].append(score_baseline1.mean())
            results['matthews_correcoef'    ].append(score_baseline2.mean())
            results['f1_score'              ].append(score_baseline3.mean())
            results['log_loss'              ].append(score_baseline4.mean())
            results['language'              ].append(language)
            results['transfer'              ].append(transfer)
            results['condition_target'      ].append(condition_target)
            results['condition_source'      ].append(condition_source)
            results['concatenate'           ].append(concatenate)
            results['flip'                  ].append(flip)
        else:
            results['tn'                    ].append(tn)
            results['fp'                    ].append(fp)
            results['fn'                    ].append(fn)
            results['tp'                    ].append(tp)
            results['fold'                  ].append(fold+1)
            results['sub'                   ].append(sub_name)
            if roi_name is not None:
                results['roi'               ].append(roi_name)
            results['model'                 ].append(model_name)
            results['roc_auc'               ].append(score_baseline1)
            results['matthews_correcoef'    ].append(score_baseline2)
            results['f1_score'              ].append(score_baseline3)
            results['log_loss'              ].append(score_baseline4)
            results['language'              ].append(language)
            results['transfer'              ].append(transfer)
            results['condition_target'      ].append(condition_target)
            results['condition_source'      ].append(condition_source)
            results['concatenate'           ].append(concatenate)
            results['flip'                  ].append(flip)
        
        return results,(score_baseline1,score_baseline2,score_baseline3,score_baseline4)
    else:
        return (score_baseline1,score_baseline2,score_baseline3,score_baseline4)
def regression_CV(
            source_BOLD,
            target_BOLD,
            source_feature,
            target_feature,
            pipeline,
            r_squares,
            scores,
            n_jobs = 1,
            scaler = None,
            variance_threshold = None,
            print_train = False,
            voxel_constrain = False,
            source_label = None,
            ):
    if scaler is None:
        scaler = StandardScaler()
    if variance_threshold is None:
        variance_threshold = VarianceThreshold()
        variance_threshold.fit(source_BOLD)
    source_BOLD                 = variance_threshold.transform(source_BOLD)
    source_BOLD                 = scaler.fit_transform(source_BOLD)
    target_BOLD                 = variance_threshold.transform(target_BOLD)
    target_BOLD                 = scaler.transform(target_BOLD)
    original_shape              = target_BOLD.shape
    if voxel_constrain:
        xgb = XGBClassifier(
                        learning_rate                           = 1e-3, # not default
                        max_depth                               = 10, # not default
                        n_estimators                            = 50, # not default
                        objective                               = 'binary:logistic', # default
                        booster                                 = 'gbtree', # default
                        subsample                               = 0.9, # not default
                        colsample_bytree                        = 0.9, # not default
                        reg_alpha                               = 0, # default
                        reg_lambda                              = 1, # default
                        random_state                            = 12345, # not default
                        importance_type                         = 'gain', # default
                        n_jobs                                  = n_jobs,# default to be 1
                        )
        RF = SelectFromModel(xgb,
                        prefit                                  = False,
#                        threshold                               = '1.96*mean' # induce sparsity
                        )
        source_BOLD = RF.fit_transform(source_BOLD,source_label)
        
    
    pipeline.fit(source_feature,source_BOLD)
    preds                   = pipeline.predict(target_feature)
    if voxel_constrain:
        target_BOLD = RF.transform(target_BOLD)
        RF_threshold = RF.threshold_
        feature_importance = RF.estimator_.feature_importances_
        idx_voxel, = np.where(feature_importance > RF_threshold)
        score = r2_score(target_BOLD,preds,multioutput='raw_values')
        empty_shell = np.zeros(original_shape[1])
        empty_shell[idx_voxel] = score
        score = empty_shell
    else:
        score = variance_threshold.inverse_transform(r2_score(target_BOLD,preds,multioutput='raw_values').reshape(1, -1))[0,:]
        
    r_squares.append(1 - np.sum((target_BOLD - preds)**2) / np.sum(target_BOLD**2))
    scores.append(score)
    if print_train:
        print(np.sum(scores[-1] > 0))

    return r_squares,scores,pipeline
def similiar_CV( 
              source_set,           # pymvpa dataset - source dataset
              target_set,           # pymvpa dataset - target dataset 
              pipeline,             # pipeline that contains a scaler, a feature processor, and an estimator
              label_map,            # one hot label map used in deep learning models
              results,              # the dictionary for storing the results
              sub_name,             # subject name
              model_name,           # name of the pipeline
              language,             # language of the stimuli
              condition_target,
              condition_source,
              cv,                   # 
              roi_name = None,      # not None if we work on selected ROI data
              average = True,       # averaging the trainig data
              transfer = False,     # do I do domain adaptation
              print_train = False,
              flip = False,
              to_save = True,
              ):
    """
    unified pipeline for decoding across many experiments and conditions 
    
    Inputs:
    --------
    source_set:         pymvpa dataset, used to get the primary training dataset
    target_set:         pymvpa dataset, used to get the secondary training dataset and/or the testing dataset
    pipeline:           scitkit learn pipeline/estimator, predict_proba required
    label_map:          a dictionary for creating the one hot labels for cross validation
    results:            a dictionary for storing the results and other attributes
    sub_name:           subject's coded name
    fold:               a counting unit
    model_name:         name of the pipeline
    language:           language in which the stimuli is
    cv: 
    condition_target:   condition/conscious in which is the subject is for the target data
    condition_source:   condition/conscious in which is the subject is for the source data
    roi_name:           name of the ROI
    average:            control whether to average the training set/s
    transfer:           indicator of whether this pipeline is in transfer learning/domain adaptation
    print_train:        to print out the training informations
    flip:               doubling the size of the training dataset by flipping the training set matrix left-right, and concatenate them
    to_save:            control whether to save the outputs
    ----------------------------------------------------------------------------------------------------------
    Returns:
    --------
    
    """
    # select the training and testing split again to make sure we do the right cv
    tr_source           = source_set
    tr_target           = target_set
    
    # average the train data for baseline to improve the signal to noise ratio
    if average:
        tr_source       = tr_source.get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
        tr_target       = tr_target.get_mapped(mean_group_sample(['chunks', 'id'],order = 'occurrence'))
    # get the numpy arrays
    X_train_source      = tr_source.samples.astype('float32')
    y_train_source      = np.zeros(X_train_source.shape[0])
    X_train_target      = tr_target.samples.astype('float32')
    y_train_target      = np.ones(X_train_target.shape[0])
    # whether to double the size of training dataset
    if flip:
        X_train_source_flip = np.fliplr(X_train_source)
        X_train_source_ = np.concatenate([X_train_source,X_train_source_flip])
        y_train_source_ = np.concatenate([y_train_source,y_train_source])
        X_train_target_flip = np.fliplr(X_train_target)
        X_train_target_ = np.concatenate([X_train_target,X_train_target_flip])
        y_train_target_ = np.concatenate([y_train_target,y_train_target])
    else:
        X_train_source_ = X_train_source
        y_train_source_ = y_train_source
        X_train_target_ = X_train_target
        y_train_target_ = y_train_target
    # shuffle the order of trials for both features and targets
    np.random.seed(12345)
#    print(X_train_source_.shape,
#          y_train_source_.shape,
#          X_train_target_.shape,
#          y_train_target_.shape)
    X_train_source_,y_train_source_ = shuffle(X_train_source_,y_train_source_)
    X_train_target_,y_train_target_ = shuffle(X_train_target_,y_train_target_)
    X_train                         = np.concatenate([X_train_source_,X_train_target_])
    labels_train                    = np.concatenate([y_train_source_,y_train_target_])
    score1, score2 = [],[]
    for train,test in cv.split(X_train,labels_train):
        X,y = X_train[train],labels_train[train]
        X_,y_ = X_train[test],labels_train[test]
        clf = clone(pipeline)
        clf.fit(X,y)
        preds = clf.predict_proba(X_)[:,-1]
        score1_ = roc_auc_score(y_,preds)
        threshold_ = Find_Optimal_Cutoff(y_,preds)
        score2_ = f1_score(y_,preds>threshold_)
        score1.append(score1_)
        score2.append(score2_)
    scores = [np.mean(score1),np.mean(score2)]
    results['sub'].append(sub_name)
    results['roi'].append(roi_name)
    results['model'].append(model_name)
    results['language'].append(language)
    results['condition_source'].append(condition_source)
    results['condition_target'].append(condition_target)
    results['roc_auc'].append(scores[0])
    results['f1_score'].append(scores[1])
    results['transfer'].append(transfer)
    results['flip'].append(False)
    return results,(np.mean(score1),np.mean(score2))
def build_feature_selector_dictionary(print_train = False,class_weight = 'balanced',n_jobs = 1):
    xgb = XGBClassifier(
                        learning_rate                           = 1e-3, # not default
                        max_depth                               = 100, # not default
                        n_estimators                            = 300, # not default
                        objective                               = 'binary:logistic', # default
                        booster                                 = 'gbtree', # default
                        subsample                               = 0.9, # not default
                        colsample_bytree                        = 0.9, # not default
                        reg_alpha                               = 0, # default
                        reg_lambda                              = 1, # default
                        random_state                            = 12345, # not default
                        importance_type                         = 'gain', # default
                        n_jobs                                  = n_jobs,# default to be 1
                                              )
    RF = SelectFromModel(xgb,
                        prefit                                  = False,
                        threshold                               = '1.96*mean' # induce sparsity
                        )
    uni = SelectPercentile(mutual_info_classif) # so annoying that I cannot control the random state
    
    return {'RandomForest':make_pipeline(MinMaxScaler(),
                                         RF,),
            'MutualInfo': make_pipeline(MinMaxScaler(),
                                        uni,)
            }
    
def build_model_dictionary(print_train = False,class_weight = 'balanced',n_jobs = 1):
    svm = LinearSVC(penalty = 'l2', # default
                    dual = True, # default
                    tol = 1e-3, # not default
                    random_state = 12345, # not default
                    max_iter = int(1e3), # default
                    class_weight = class_weight, # not default
                    )
    svm = CalibratedClassifierCV(base_estimator = svm,
                                 method = 'sigmoid',
                                 cv = 3)
    xgb = XGBClassifier(
                        learning_rate                           = 1e-3, # not default
                        max_depth                               = 100, # not default
                        n_estimators                            = 300, # not default
                        objective                               = 'binary:logistic', # default
                        booster                                 = 'gbtree', # default
                        subsample                               = 0.9, # not default
                        colsample_bytree                        = 0.9, # not default
                        reg_alpha                               = 0, # default
                        reg_lambda                              = 1, # default
                        random_state                            = 12345, # not default
                        importance_type                         = 'gain', # default
                        n_jobs                                  = n_jobs,# default to be 1
                                              )
    bagging = BaggingClassifier(base_estimator                  = svm,
                                 n_estimators                   = 30, # not default
                                 max_features                   = 0.9, # not default
                                 max_samples                    = 0.9, # not default
                                 bootstrap                      = True, # default
                                 bootstrap_features             = True, # default
                                 random_state                   = 12345, # not default
                                                 )
    RF = SelectFromModel(xgb,
                        prefit                                  = False,
                        threshold                               = '1.96*mean' # induce sparsity
                        )
    uni = SelectPercentile(mutual_info_classif) # so annoying that I cannot control the random state
    knn = KNeighborsClassifier()
    tree = DecisionTreeClassifier(random_state = 12345,
                                  class_weight = class_weight)
    dummy = DummyClassifier(strategy = 'uniform',random_state = 12345,)
    models = OrderedDict([
            ['None + Dummy',                     make_pipeline(MinMaxScaler(),
                                                               dummy,)],
            ['None + Linear-SVM',                make_pipeline(MinMaxScaler(),
                                                              svm,)],
            ['None + Ensemble-SVMs',             make_pipeline(MinMaxScaler(),
                                                              bagging,)],
            ['None + KNN',                       make_pipeline(MinMaxScaler(),
                                                              knn,)],
            ['None + Tree',                      make_pipeline(MinMaxScaler(),
                                                              tree,)],
            ['PCA + Dummy',                      make_pipeline(MinMaxScaler(),
                                                               PCA(),
                                                               dummy,)],
            ['PCA + Linear-SVM',                 make_pipeline(MinMaxScaler(),
                                                              PCA(),
                                                              svm,)],
            ['PCA + Ensemble-SVMs',              make_pipeline(MinMaxScaler(),
                                                              PCA(),
                                                              bagging,)],
            ['PCA + KNN',                        make_pipeline(MinMaxScaler(),
                                                              PCA(),
                                                              knn,)],
            ['PCA + Tree',                       make_pipeline(MinMaxScaler(),
                                                              PCA(),
                                                              tree,)],
            ['Mutual + Dummy',                   make_pipeline(MinMaxScaler(),
                                                               uni,
                                                               dummy,)],
            ['Mutual + Linear-SVM',              make_pipeline(MinMaxScaler(),
                                                              uni,
                                                              svm,)],
            ['Mutual + Ensemble-SVMs',           make_pipeline(MinMaxScaler(),
                                                              uni,
                                                              bagging,)],
            ['Mutual + KNN',                     make_pipeline(MinMaxScaler(),
                                                              uni,
                                                              knn,)],
            ['Mutual + Tree',                    make_pipeline(MinMaxScaler(),
                                                              uni,
                                                              tree,)],
            ['RandomForest + Dummy',             make_pipeline(MinMaxScaler(),
                                                               RF,
                                                               dummy,)],
            ['RandomForest + Linear-SVM',        make_pipeline(MinMaxScaler(),
                                                              RF,
                                                              svm,)],
            ['RandomForest + Ensemble-SVMs',     make_pipeline(MinMaxScaler(),
                                                              RF,
                                                              bagging,)],
            ['RandomForest + KNN',               make_pipeline(MinMaxScaler(),
                                                              RF,
                                                              knn,)],
            ['RandomForest + Tree',              make_pipeline(MinMaxScaler(),
                                                              RF,
                                                              tree,)],]
            )
    return models

def estimator_search_grids():
    return {
            'RandomForest':{'RandomForestClassifier__criterion':['entropy','gini'],
                            'RandomForestClassifier__max_depth':[2,4,8,16,32,64],
#                            'RandomForestClassifier__min_samples_split':[2,0.2,0.5],
#                            'RandomForestClassifier__min_samples_leaf':[0.1,0.3,0.5],
                            'RandomForestClassifier__min_impurity_decrease':[0.001,0.01,0.1],
                            'RandomForestClassifier__n_estimators':[100,200,300,400,500],
                                      },
#            'Linear-SVM':{'svc__C':np.logspace(-2,2,num=5,)},
            'Linear-SVM':{'LinearSVC__C':np.logspace(-2,2,num=5,),
                          'LinearSVC__penalty':['l1','l2'],
                          },
            'Xgboost':{'XGBClassifier__max_depth':[2,4,8,16,],#32,64],
                       'XGBClassifier__n_estimators':[100,200,300,400,500],
                       'XGBClassifier__booster':['gbtree','gblinear','dart'],
                       'XGBClassifier__subsample':[1.,0.9,0.8],
                       'XGBClassifier__reg_alpha':np.logspace(-3,2,6),
                       'XGBClassifier__reg_lambda':np.logspace(-3,2,6),
                       'XGBClassifier__learning_rate':[1e-3,1e-2,1e-1],
                             },
            
            
            }
def feature_selection_search_grids():
    return {
            'RandomForest':{'selectfrommodel__estimator__criterion':['entropy','gini'],
                            'selectfrommodel__estimator__max_depth':[2,4,8,16,],#32,64],
#                            'selectfrommodel__estimator__min_samples_split':[2,0.2,0.5],
#                            'selectfrommodel__estimator__min_samples_leaf':[0.1,0.3,0.5],
                            'selectfrommodel__estimator__min_impurity_decrease':[0.001,0.01,0.1],
                            'selectfrommodel__estimator__n_estimators':[100,200,300,400,500],
                              },
            'None':{},
            'PCA':{'pca__n_components':[None,.95,.99],},
            }
def eta_squared(aov):
    aov['eta_sq'] = 'NaN'
    aov['eta_sq'] = aov[:-1]['sum_sq']/sum(aov['sum_sq'])
    return aov
 
def omega_squared(aov):
    mse_ = aov['sum_sq'][-1]/aov['df'][-1]
    aov['omega_sq'] = 'NaN'
    aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse_))/(sum(aov['sum_sq'])+mse_)
    return aov




#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
################## from https://github.com/parrt/random-forest-importances/blob/master/src/rfpimp.py#L237 #############
################## reference http://explained.ai/rf-importance/index.html #############################################
def oob_classifier_accuracy(rf, X_train, y_train):
    """
    Adjusted... 
    Compute out-of-bag (OOB) accuracy for a scikit-learn random forest
    classifier. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L425
    """
    try:
        X                   = X_train.values
    except:
        X                   = X_train.copy()
    try:
        y                   = y_train.values
    except: 
        y                   = y_train.copy()

    n_samples               = len(X)
    n_classes               = len(np.unique(y))
    # preallocation
    predictions             = np.zeros((n_samples, n_classes))
    for tree in rf.estimators_: # for each decision tree in the random forest - I have put 1 tree in the forest
        # Private function used to _parallel_build_trees function.
        unsampled_indices   = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds          = tree.predict_proba(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds

    predicted_class_indexes = np.argmax(predictions, axis=1)# threshold the probabilistic predictions
    predicted_classes       = [rf.classes_[i] for i in predicted_class_indexes] # use the thresholded indicies to obtain a binary prediction

    oob_score               = sum(y==predicted_classes) / float(len(y))
    return oob_score
def sample(X_valid, y_valid, n_samples):
    """
    Not sure what this is doing
    Only if the n_sample is less than the total number of samples, it subsamples the data???? Maybe?
    """
    if n_samples < 0: 
        n_samples                   = len(X_valid)
    n_samples                       = min(n_samples, len(X_valid))
    if n_samples < len(X_valid):
        ix                          = np.random.choice(len(X_valid), n_samples)
        X_valid                     = X_valid.iloc[ix].copy(deep=False)  # shallow copy
        y_valid                     = y_valid.iloc[ix].copy(deep=False)
    return X_valid, y_valid
def permutation_importances_raw(rf, X_train, y_train, metric, n_samples=5000):
    """
    Return array of importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    """
    X_sample, y_sample          = shuffle(X_train, y_train)
    # get a baseline out-of-bag sampled decoding score
    if metric is None:
        baseline                    = oob_classifier_accuracy(rf, X_sample, y_sample)
    else:
        baseline                    = metric(y_sample,rf.predict_proba(X_sample)[:,-1])
    
    # make suer that we work on the copy of the raw data
    X_train                     = X_sample.copy() # shallow copy
    X_train                     = pd.DataFrame(X_train)
    y_train                     = y_sample
    imp                         = []
#    for n_ in range(100):
#        imp_temp = []
#        for col in X_train.columns: # for each feature
#            save                = X_train[col].copy() # save the original
#            X_train[col]        = np.random.uniform(save.min(),save.max(),size=save.shape) # reorder
#            # oob score after reorder 1 and only 1 feature. 
#            # In orther words, how much information is gone when the feature becomes unimformative
#            try:
#                m                   = metric(rf, X_train, y_train)
#            except:
#                m                   = metric(y_train,rf.predict_proba(X_train.values)[:,-1])
#            X_train[col]        = save # restore the feature
#            imp_temp.append(baseline - m)
#        imp.append(imp_temp)
    
    for col in tqdm(X_train.columns): # for each feature
        save                = X_train[col].copy() # save the original
        X_train[col]        = np.random.uniform(save.min(),save.max(),size=save.shape) # reorder
        # oob score after reorder 1 and only 1 feature. 
        # In orther words, how much information is gone when the feature becomes unimformative
        if metric is None:
            m                   = oob_classifier_accuracy(rf, X_train, y_train)
        else:
            m                   = metric(y_train,rf.predict_proba(X_train.values)[:,-1])
        X_train[col]        = save # restore the feature
        imp.append(baseline - m)
        
    return np.array(imp)#.mean(0)
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
def permutation_importances(rf, X_train, y_train, metric=None, n_samples = 5000,feature_names = None):
    """
    Call the function, and just to make it pretty
    """
    imp = permutation_importances_raw(rf, X_train, y_train, metric, n_samples)
    imp = softmax(imp)
    I = pd.DataFrame(data={'Feature':feature_names, 'Importance':imp})
    I = I.set_index('Feature')
    return I
######################################################################################################
######################################################################################################
def no_feature_selection(X):
    return X
def build_feature_selectors():
    np.random.seed(12345)
    xgb = XGBClassifier(
                        learning_rate                           = 1e-3, 
                        n_estimators                            = 300, 
                        objective                               = 'binary:logistic', 
                        booster                                 = 'gbtree', 
                        subsample                               = 0.8, 
                        reg_alpha                               = 1e-4, 
                        reg_lambda                              = 1e-4, 
                        random_state                            = 12345, 
                        importance_type                         = 'gain',
                                              )
    pca = PCA()
    RF = SelectFromModel(xgb,prefit = False)
    identical_transformer = FunctionTransformer(func = no_feature_selection,
                                                inverse_func = no_feature_selection)
    
    return {'None':make_pipeline(MinMaxScaler(), identical_transformer),
            'PCA':make_pipeline(MinMaxScaler(),pca),
            'RF':make_pipeline(MinMaxScaler(),RF),
            }

def posthoc_multiple_comparison(df_sub,depvar = '',factor='',n_ps=100,n_permutation=5000):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            ps_mean = [],
            ps_std = [],
            level1 = [],
            level2 = []
            )
    from itertools import combinations
    unique_levels = pd.unique(df_sub[factor])
    pairs = combinations(unique_levels,2)
    try:
        iterator = tqdm(pairs,desc='pairs')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[df_sub[factor] == level1].groupby(['sub',factor]).mean().reset_index()
        b = df_sub[df_sub[factor] == level2].groupby(['sub',factor]).mean().reset_index()
        if a[depvar].values.mean() < b[depvar].values.mean():
            level1,level2 = level2,level1
            a = df_sub[df_sub[factor] == level1].groupby(['sub',factor]).mean().reset_index()
            b = df_sub[df_sub[factor] == level2].groupby(['sub',factor]).mean().reset_index()
        ps = resample_ttest_2sample(a[depvar].values,
                                    b[depvar].values,
                                    n_ps=n_ps,
                                    n_permutation=n_permutation)
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['level1'].append(level1)
        results['level2'].append(level2)
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['ps_mean'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['ps_mean'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results
def posthoc_multiple_comparison_scipy(df_sub,depvar = '',factor='',):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            pval = [],
            t = [],
            df = [],
            level1 = [],
            level2 = []
            )
    from itertools import combinations
    unique_levels = pd.unique(df_sub[factor])
    pairs = combinations(unique_levels,2)
    try:
        iterator = tqdm(pairs,desc='pairs')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[df_sub[factor] == level1].groupby(['sub',factor]).mean().reset_index()
        b = df_sub[df_sub[factor] == level2].groupby(['sub',factor]).mean().reset_index()
        if a[depvar].values.mean() < b[depvar].values.mean():
            level1,level2 = level2,level1
            a = df_sub[df_sub[factor] == level1].groupby(['sub',factor]).mean().reset_index()
            b = df_sub[df_sub[factor] == level2].groupby(['sub',factor]).mean().reset_index()
        t,pval = stats.ttest_rel(a[depvar].values,
                                 b[depvar].values,)
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(df_sub)*2-2)
        results['level1'].append(level1)
        results['level2'].append(level2)
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['pval'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['pval'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results
def posthoc_multiple_comparison_interaction(df_sub,
                                            depvar = '',
                                            unique_levels = [],
                                            n_ps=100,
                                            n_permutation=5000,):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            ps_mean = [],
            ps_std = [],
            level1 = [],
            level2 = []
            )
    unique_factor1 = np.unique(df_sub[unique_levels[0]])
    unique_factor2 = np.unique(df_sub[unique_levels[1]])
    pairs = [[a,b] for a in unique_factor1 for b in unique_factor2]
    pairs = combinations(pairs,2)
    try:
        iterator = tqdm(pairs,desc='interaction')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[(df_sub[unique_levels[0]] == level1[0]) & (df_sub[unique_levels[1]] == level1[1])]
        b = df_sub[(df_sub[unique_levels[0]] == level2[0]) & (df_sub[unique_levels[1]] == level2[1])]
        if a[depvar].values.mean() < b[depvar].values.mean():
            level1,level2 = level2,level1
        a = df_sub[(df_sub[unique_levels[0]] == level1[0]) & (df_sub[unique_levels[1]] == level1[1])]
        b = df_sub[(df_sub[unique_levels[0]] == level2[0]) & (df_sub[unique_levels[1]] == level2[1])]
        ps = resample_ttest_2sample(a[depvar].values,
                                    b[depvar].values,
                                    n_ps=n_ps,
                                    n_permutation=n_permutation)
        results['ps_mean'].append(ps.mean())
        results['ps_std'].append(ps.std())
        results['level1'].append('{}_{}'.format(level1[0],level1[1]))
        results['level2'].append('{}_{}'.format(level2[0],level2[1]))
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['ps_mean'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['ps_mean'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results
def posthoc_multiple_comparison_interaction_scipy(df_sub,
                                            depvar = '',
                                            unique_levels = [],):
    """
    post hoc multiple comparison with bonferroni correction procedure
    main effect only so far
    factor: the main effect we want to test
    
    """
    results = dict(
            pval = [],
            t = [],
            df = [],
            level1 = [],
            level2 = []
            )
    unique_factor1 = np.unique(df_sub[unique_levels[0]])
    unique_factor2 = np.unique(df_sub[unique_levels[1]])
    pairs = [[a,b] for a in unique_factor1 for b in unique_factor2]
    pairs = combinations(pairs,2)
    try:
        iterator = tqdm(pairs,desc='interaction')
    except:
        iterator = pairs
    for (level1,level2) in iterator:
        a = df_sub[(df_sub[unique_levels[0]] == level1[0]) & (df_sub[unique_levels[1]] == level1[1])]
        b = df_sub[(df_sub[unique_levels[0]] == level2[0]) & (df_sub[unique_levels[1]] == level2[1])]
        if a[depvar].values.mean() < b[depvar].values.mean():
            level1,level2 = level2,level1
        a = df_sub[(df_sub[unique_levels[0]] == level1[0]) & (df_sub[unique_levels[1]] == level1[1])]
        b = df_sub[(df_sub[unique_levels[0]] == level2[0]) & (df_sub[unique_levels[1]] == level2[1])]
        t,pval = stats.ttest_rel(a[depvar].values,
                                 b[depvar].values,
                                    )
        results['pval'].append(pval)
        results['t'].append(t)
        results['df'].append(len(a)+len(b) - 2)
        results['level1'].append('{}_{}'.format(level1[0],level1[1]))
        results['level2'].append('{}_{}'.format(level2[0],level2[1]))
    results = pd.DataFrame(results)
    
    idx_sort = np.argsort(results['pval'].values)
    results = results.iloc[idx_sort,:]
    pvals = results['pval'].values
    converter = MCPConverter(pvals=pvals)
    d = converter.adjust_many()
    results['p_corrected'] = d['bonferroni'].values
    
    return results

def stars(x):
    if x < 0.001:
        return '***'
    elif x < 0.01:
        return '**'
    elif x < 0.05:
        return '*'
    else:
        return 'n.s.'

def compute_xy(df_sub,position_map,hue_map):
    df_add = []
    for ii,row in df_sub.iterrows():
        xtick = int(row['window']) - 1
        attribute1_x = xtick + position_map[hue_map[row['attribute1']]]
        attribute2_x = xtick + position_map[hue_map[row['attribute2']]]
        row['x1'] = attribute1_x
        row['x2'] = attribute2_x
        df_add.append(row.to_frame().T)
    df_add = pd.concat(df_add)
    return df_add


































