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
def resample_ttest(x,baseline = 0.5,n_ps = 100,n_permutation = 10000,one_tail = False,
                   n_jobs = 12, verbose = 0):
    """
    http://www.stat.ucla.edu/~rgould/110as02/bshypothesis.pdf
    https://www.tau.ac.il/~saharon/StatisticsSeminar_files/Hypothesis.pdf
    Inputs:
    ----------
    x: numpy array vector, the data that is to be compared
    baseline: the single point that we compare the data with
    n_ps: number of p values we want to estimate
    one_tail: whether to perform one-tailed comparison
    """
    import numpy as np
    # t statistics with the original data distribution
    t_experiment = (np.mean(x) - baseline) / (np.std(x) / np.sqrt(x.shape[0]))
    null            = x - np.mean(x) + baseline # shift the mean to the baseline but keep the distribution
    from joblib import Parallel,delayed
    import gc
    gc.collect()
    def t_statistics(null,size,):
        """
        null: shifted data distribution
        size: tuple of 2 integers (n_for_averaging,n_permutation)
        """
        null_dist = np.random.choice(null,size = size,replace = True)
        t_null = (np.mean(null_dist,0) - baseline) / (np.std(null_dist,0) / np.sqrt(null_dist.shape[0]))
        if one_tail:
            return ((np.sum(t_null >= t_experiment)) + 1) / (size[1] + 1)
        else:
            return ((np.sum(np.abs(t_null) >= np.abs(t_experiment))) + 1) / (size[1] + 1) /2
    ps = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                    'null':null,
                    'size':(null.shape[0],int(n_permutation)),}) for i in range(n_ps))
    
    return np.array(ps)
def resample_ttest_2sample(a,b,n_ps=100,n_permutation = 10000,
                           one_tail=False,
                           match_sample_size = True,
                           n_jobs = 6,
                           verbose = 0):
    # when the samples are dependent just simply test the pairwise difference against 0
    # which is a one sample comparison problem
    if match_sample_size:
        difference  = a - b
        ps          = resample_ttest(difference,baseline=0,
                                     n_ps=n_ps,n_permutation=n_permutation,
                                     one_tail=one_tail,
                                     n_jobs=n_jobs,
                                     verbose=verbose,)
        return ps
    else: # when the samples are independent
        t_experiment,_ = stats.ttest_ind(a,b,equal_var = False)
        def t_statistics(a,b):
            group = np.random.choice(np.concatenate([a,b]),size = int(len(a) + len(b)),replace = True)
            new_a = group[:a.shape[0]]
            new_b = group[a.shape[0]:]
            t_null,_ = stats.ttest_ind(new_a,new_b,equal_var = False)
            return t_null
        from joblib import Parallel,delayed
        import gc
        gc.collect()
        ps = np.zeros(n_ps)
        for ii in range(n_ps):
            t_null_null = Parallel(n_jobs = n_jobs,verbose = verbose)(delayed(t_statistics)(**{
                            'a':a,
                            'b':b}) for i in range(n_permutation))
            if one_tail:
                ps[ii] = ((np.sum(t_null_null >= t_experiment)) + 1) / (n_permutation + 1)
            else:
                ps[ii] = ((np.sum(np.abs(t_null_null) >= np.abs(t_experiment))) + 1) / (n_permutation + 1) / 2
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

def add_track(df_sub):
    n_rows = df_sub.shape[0]
    temp = '+'.join(str(item + 10) for item in df_sub['index'].values)
    df_sub = df_sub.iloc[1,:].to_frame().T
    df_sub['n_volume'] = n_rows
    df_sub['time_indices'] = temp
    return df_sub
def groupby_average(fmri,df,groupby = ['trials']):
    BOLD_average = np.array([np.mean(fmri[df_sub.index],0) for _,df_sub in df.groupby(groupby)])
    df_average = pd.concat([add_track(df_sub) for ii,df_sub in df.groupby(groupby)])
    return BOLD_average,df_average

def customized_partition(df_data,groupby_column = ['id','words'],n_splits = 100,):
    """
    modified for unaveraged volumes
    """
    idx_object = dict(ids = [],idx = [],words = [])
    for label,df_sub in df_data.groupby(groupby_column):
        idx_object['ids'].append(label[0])
        idx_object['idx'].append(df_sub.index.tolist())
        idx_object['words'].append(label[-1])
    df_object = pd.DataFrame(idx_object)
    
    idxs_test       = []
    for counter in range(int(1e4)):
        idx_test = [np.random.choice(item['idx'].values) for ii,item in df_object.groupby(groupby_column[-1])]
        if counter >= n_splits:
            return [np.concatenate(item) for item in idxs_test]
            break
        if counter > 0:
            temp = []
            for used in idxs_test:
                used_temp = [','.join(str(ii) for ii in item) for item in used]
                idx_test_temp = [','.join(str(ii) for ii in item) for item in idx_test]
                a = set(used_temp)
                b = set(idx_test_temp)
                temp.append(len(a.intersection(b)) != len(idx_test))
            if all(temp) == True:
                idxs_test.append(idx_test)
        else:
            idxs_test.append(idx_test)
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
def check_train_balance(df,idx_train,keys):
    Counts = dict(Counter(df.iloc[idx_train]['targets'].values))
    if np.abs(Counts[keys[0]] - Counts[keys[1]]) > 2:
        if Counts[keys[0]] > Counts[keys[1]]:
            key_major = keys[0]
            key_minor = keys[1]
        else:
            key_major = keys[1]
            key_minor = keys[0]
            
        ids_major = df.iloc[idx_train]['id'][df.iloc[idx_train]['targets'] == key_major]
        
        idx_train_new = idx_train.copy()
        for n in range(len(idx_train_new)):
            random_pick = np.random.choice(np.unique(ids_major),size = 1)[0]
            # print(random_pick,np.unique(ids_major))
            idx_train_new = np.array([item for item,id_temp in zip(idx_train_new,df.iloc[idx_train_new]['id']) if (id_temp != random_pick)])
            ids_major = np.array([item for item in ids_major if (item != random_pick)])
            new_counts = dict(Counter(df.iloc[idx_train_new]['targets']))
            if np.abs(new_counts[keys[0]] - new_counts[keys[1]]) > 3:
                if new_counts[keys[0]] > new_counts[keys[1]]:
                    key_major = keys[0]
                    key_minor = keys[1]
                else:
                    key_major = keys[1]
                    key_minor = keys[0]
                
                ids_major = df.iloc[idx_train_new]['id'][df.iloc[idx_train_new]['targets'] == key_major]
            elif np.abs(new_counts[keys[0]] - new_counts[keys[1]]) < 3:
                break
        return idx_train_new
    else:
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
    
def build_model_dictionary(print_train = False,
                           class_weight = 'balanced',
                           remove_invariant = False,
                           n_jobs = 1):
    np.random.seed(12345)
    svm = LinearSVC(penalty = 'l2', # default
                    dual = True, # default
                    tol = 1e-3, # not default
                    random_state = 12345, # not default
                    max_iter = int(1e3), # default
                    class_weight = class_weight, # not default
                    )
    svm = CalibratedClassifierCV(base_estimator = svm,
                                 method = 'sigmoid',
                                 cv = 8)
    xgb = XGBClassifier(
                        learning_rate                           = 1e-3, # not default
                        max_depth                               = 10, # not default
                        n_estimators                            = 100, # not default
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
                        threshold                               = 'median' # induce sparsity
                        )
    uni = SelectPercentile(mutual_info_classif,50) # so annoying that I cannot control the random state
    knn = KNeighborsClassifier()
    tree = DecisionTreeClassifier(random_state = 12345,
                                  class_weight = class_weight)
    dummy = DummyClassifier(strategy = 'uniform',random_state = 12345,)
    if remove_invariant:
        RI = VarianceThreshold()
        models = OrderedDict([
                ['None + Dummy',                     make_pipeline(RI,MinMaxScaler(),
                                                                   dummy,)],
                ['None + Linear-SVM',                make_pipeline(RI,MinMaxScaler(),
                                                                  svm,)],
                ['None + Ensemble-SVMs',             make_pipeline(RI,MinMaxScaler(),
                                                                  bagging,)],
                ['None + KNN',                       make_pipeline(RI,MinMaxScaler(),
                                                                  knn,)],
                ['None + Tree',                      make_pipeline(RI,MinMaxScaler(),
                                                                  tree,)],
                ['PCA + Dummy',                      make_pipeline(RI,MinMaxScaler(),
                                                                   PCA(),
                                                                   dummy,)],
                ['PCA + Linear-SVM',                 make_pipeline(RI,MinMaxScaler(),
                                                                  PCA(),
                                                                  svm,)],
                ['PCA + Ensemble-SVMs',              make_pipeline(RI,MinMaxScaler(),
                                                                  PCA(),
                                                                  bagging,)],
                ['PCA + KNN',                        make_pipeline(RI,MinMaxScaler(),
                                                                  PCA(),
                                                                  knn,)],
                ['PCA + Tree',                       make_pipeline(RI,MinMaxScaler(),
                                                                  PCA(),
                                                                  tree,)],
                ['Mutual + Dummy',                   make_pipeline(RI,MinMaxScaler(),
                                                                   uni,
                                                                   dummy,)],
                ['Mutual + Linear-SVM',              make_pipeline(RI,MinMaxScaler(),
                                                                  uni,
                                                                  svm,)],
                ['Mutual + Ensemble-SVMs',           make_pipeline(RI,MinMaxScaler(),
                                                                  uni,
                                                                  bagging,)],
                ['Mutual + KNN',                     make_pipeline(RI,MinMaxScaler(),
                                                                  uni,
                                                                  knn,)],
                ['Mutual + Tree',                    make_pipeline(RI,MinMaxScaler(),
                                                                  uni,
                                                                  tree,)],
                ['RandomForest + Dummy',             make_pipeline(RI,MinMaxScaler(),
                                                                   RF,
                                                                   dummy,)],
                ['RandomForest + Linear-SVM',        make_pipeline(RI,MinMaxScaler(),
                                                                  RF,
                                                                  svm,)],
                ['RandomForest + Ensemble-SVMs',     make_pipeline(RI,MinMaxScaler(),
                                                                  RF,
                                                                  bagging,)],
                ['RandomForest + KNN',               make_pipeline(RI,MinMaxScaler(),
                                                                  RF,
                                                                  knn,)],
                ['RandomForest + Tree',              make_pipeline(RI,MinMaxScaler(),
                                                                  RF,
                                                                  tree,)],]
                )
    else:
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


































