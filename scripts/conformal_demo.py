import csv
import pdb
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

def fix_randomness(seed=0):
    np.random.seed(seed=seed)
    random.seed(seed)
    
def get_discrete_quantiles(scores, beta, mode=None): # Gets the beta quantile from discrete scores
    cumsum = scores.cumsum(axis=1)
    if not mode:
        quantiles = cumsum[np.arange(cumsum.shape[0]),np.argmax( cumsum >= np.maximum(np.minimum(beta,1-1e-15),0), axis=1)]
    elif mode == 'lower':
        index_of_quantile = np.argmax( cumsum >= np.maximum(np.minimum(beta,1-1e-15),0), axis=1) - 1
        quantiles = cumsum[np.arange(cumsum.shape[0]),np.maximum(index_of_quantile,0)]  
        quantiles[index_of_quantile < 0] = 0.0 
    elif mode == 'upper':
        index_of_quantile = np.argmax( cumsum >= np.maximum(np.minimum(beta,1-1e-15),0), axis=1 ) + 1
        quantiles = cumsum[np.arange(cumsum.shape[0]),np.minimum(index_of_quantile, cumsum.shape[1]-1)] 
        quantiles[index_of_quantile > cumsum.shape[1]-1] = 1.0
    else:
        quantiles = None
    return quantiles

def lac_score_function(cal_scores, cal_labels):
    return 1-cal_scores[np.arange(cal_labels.shape[0]), cal_labels]

def lac_prediction(val_scores,qhat):
    return val_scores > 1-qhat

def cdf_naive_ordinal_score_function(cal_scores, cal_labels):
    cumsum = cal_scores.cumsum(axis=1)
    argmaxes = cal_scores.argmax(axis=1)
    indexer = np.arange(cal_scores.shape[0])
    maxes = cumsum[indexer,argmaxes]
    return np.abs(maxes-cumsum[indexer,cal_labels])

def cdf_naive_ordinal_prediction(val_scores, qhat):
    cumsum = val_scores.cumsum(axis=1)
    argmaxes = val_scores.argmax(axis=1)
    maxes = cumsum[np.arange(val_scores.shape[0]),argmaxes]
    prediction_set = ((cumsum >= (maxes[:,None] - qhat)) & (cumsum <= (maxes[:,None] + qhat)))
    return prediction_set

def ordinal_aps_score_function(cal_scores, cal_labels):
    P = cal_scores == cal_scores.max(axis=1)[:,None]
    idx_construction_incomplete = ~P[np.arange(P.shape[0]),cal_labels] # Places where top-1 isn't correct
    cal_scores[~idx_construction_incomplete] = 0 # Places where top-1 is correct have a score of 0
    while idx_construction_incomplete.sum() > 0:
        P_inc = P[idx_construction_incomplete]
        scores_inc = cal_scores[idx_construction_incomplete]
        set_cumsum = P_inc.cumsum(axis=1)
        lower_edge_idx = (P_inc > 0).argmax(axis=1)
        upper_edge_idx = set_cumsum.argmax(axis=1)
        # Where the lower edge is both valid and also has a higher softmax score than the upper edge
        lower_edge_wins = ((lower_edge_idx - 1) >= 0) & \
                                (  \
                                    (upper_edge_idx + 1 > scores_inc.shape[1]-1) | \
                                    (scores_inc[np.arange(scores_inc.shape[0]),np.maximum(lower_edge_idx - 1,0)] > \
                                     scores_inc[np.arange(scores_inc.shape[0]),np.minimum(upper_edge_idx + 1,scores_inc.shape[1]-1)]) \
                                )
        P_inc[lower_edge_wins,lower_edge_idx[lower_edge_wins]-1] = True
        P_inc[~lower_edge_wins,upper_edge_idx[~lower_edge_wins]+1] = True
        P[idx_construction_incomplete] = P_inc
        idx_construction_incomplete = ~P[np.arange(P.shape[0]),cal_labels] # Where the labels are not included
    return (cal_scores * P.astype(float)).sum(axis=1)

# This is a numerical crutch to fix the error in the ordinal aps score function
def get_qhat_ordinal_aps(prediction_function, cal_scores, cal_labels, alpha):
    n = cal_scores.shape[0]
    grid_size = 10000
    for q in np.linspace(1e-3, 1 - 1e-3, grid_size)[::-1]:
        coverage, _, _ = evaluate_sets(prediction_function, np.copy(cal_scores), np.copy(cal_labels), q, alpha)
        if coverage <= (np.ceil((n + 1)*(1 - alpha))/n):
            # return q + 1/(grid_size - 1)
            return np.minimum(q + 1/(grid_size - 1), 1.0 - 1e-6)  # Clip q to be less than 1.0
    return q

def ordinal_aps_prediction(val_scores, qhat):
    # if qhat > 1:  # bug somewhere?
        # return np.ones_like(val_scores).astype(bool)
    P = val_scores == val_scores.max(axis=1)[:, None]
    idx_construction_incomplete = (val_scores * P.astype(float)).sum(axis=1) <= qhat # Places where top-1 isn't correct
    while idx_construction_incomplete.sum() > 0:
        P_inc = P[idx_construction_incomplete]
        scores_inc = val_scores[idx_construction_incomplete]
        set_cumsum = P_inc.cumsum(axis=1)
        lower_edge_idx = (P_inc > 0).argmax(axis=1)
        upper_edge_idx = set_cumsum.argmax(axis=1)
        
        # Where the lower edge is both valid and also has a higher softmax score than the upper edge
        lower_edge_wins = (
            ((lower_edge_idx - 1) >= 0) &
            (
                (upper_edge_idx + 1 > scores_inc.shape[1] - 1) |
                (
                    scores_inc[np.arange(scores_inc.shape[0]), np.maximum(lower_edge_idx - 1, 0)] > 
                    scores_inc[np.arange(scores_inc.shape[0]), np.minimum(upper_edge_idx + 1, scores_inc.shape[1] - 1)]
                )
            )
        )
        P_inc[lower_edge_wins, lower_edge_idx[lower_edge_wins] - 1] = True
        P_inc[~lower_edge_wins, upper_edge_idx[~lower_edge_wins] + 1] = True  # IndexError here when alpha is too small
        P[idx_construction_incomplete] = P_inc
        idx_construction_incomplete = (val_scores * P.astype(float)).sum(axis=1) <= qhat
    return P

def evaluate_sets(prediction_function, val_scores, val_labels, qhat, alpha, print_bool=False):
    sets = prediction_function(val_scores, qhat) 
    # Check
    sizes = sets.sum(axis=1)
    sizes_distribution = np.array([(sizes == i).mean() for i in range(5)])
    # Evaluate coverage
    covered = sets[np.arange(val_labels.shape[0]), val_labels]
    coverage = covered.mean()
    label_stratified_coverage = [
        covered[val_labels == j].mean() for j in range(np.unique(val_labels).max() + 1)
    ]
    label_distribution = [
        (val_labels == j).mean() for j in range(np.unique(val_labels).max() + 1)
    ]
    if(print_bool):
        print(r'$\alpha$' + f":{alpha}  |  coverage: {coverage}  |  average size: {sizes.mean()}  |  qhat: {qhat}  |  set size distribution: {sizes_distribution} ")
        print(f"label stratified coverage: {label_stratified_coverage}  \nlabel distribution: {label_distribution}")
    return coverage, label_stratified_coverage, sizes_distribution

def class_conditional_eval(sets, labels, classes=['neg', 'mod', 'mil', 'sev']):
    res = collections.defaultdict()
    for i, v in enumerate(classes):
        class_cond_sets = sets[labels == i]
        n = class_cond_sets.shape[0]
        res[v] = {
            'count': n,
            'coverage': (class_cond_sets[np.arange(n), i].sum() / n) if n > 0 else 0,
            'size': class_cond_sets.sum(1).mean(),
        }
    return res

def size_conditional_eval(sets, labels, sizes=[1, 2, 3, 4]):
    res = collections.defaultdict()
    for i, v in enumerate(sizes):
        size_cond_sets = sets[sets.sum(1) == v]
        n = size_cond_sets.shape[0]
        res[v] = {
            'count': n,
            'coverage': (size_cond_sets[np.arange(n), i].sum() / n) if n > 0 else 0,
        }
    return res

if __name__ == "__main__":
    fix_randomness(seed=1000)
    # Experimental parameters
    num_trials = 100
    # Define miscoverage rate
    alpha = 0.1
    # Scores is n_patients X num_locations X num_severity_classes
    scores = np.load('../files/scores.npy')
    # Labels is n_patients X num_locations 
    labels = np.load('../files/labels.npy').astype(int)
    scores = scores.reshape(-1,scores.shape[-1])
    labels = labels.flatten()
    # Check validity
    valid = (scores.sum(axis=1) > 0) & (labels >= 0)
    scores = scores[valid]
    labels = labels[valid]

    # Version of conformal
    #score_function = cdf_naive_ordinal_score_function 
    #prediction_function = cdf_naive_ordinal_prediction
    score_function = ordinal_aps_score_function
    prediction_function = ordinal_aps_prediction

    coverages = []
    for trial in range(num_trials):
        # Permute
        perm = np.random.permutation(scores.shape[0])
        scores = scores[perm]
        labels = labels[perm]
        # Split
        n = scores.shape[0]//2 # 50/50 split
        cal_scores, val_scores = (scores[:n], scores[n:])
        cal_labels, val_labels = (labels[:n], labels[n:])
        # Calculate accuracy
        est_labels = np.argmax(scores,axis=1)
        acc = (labels == est_labels).mean()
        print(f"Model accuracy: {acc}")

        # Calculate quantile
        #qhat = np.quantile(score_function(np.copy(cal_scores),np.copy(cal_labels)),np.ceil((n+1)*(1-alpha))/n,interpolation='higher')
        qhat = get_qhat_ordinal_aps(prediction_function, np.copy(cal_scores), np.copy(cal_labels), alpha)
        # Calculate sets
        coverages = coverages + [evaluate_sets(prediction_function, np.copy(val_scores), np.copy(val_labels), qhat, alpha, print_bool=True)[0],]
    print(np.histogram(coverages))
