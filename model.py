import json
import pandas as pd
import random
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn import feature_extraction
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

import xgboost as xgb
import sys

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import matthews_corrcoef


def load_munging_data():
    with open('path_data.json', 'r') as infile:
        path_data = json.load(infile)
    with open('feature_data.json', 'r') as infile:
        feature_data = json.load(infile)
    with open('data/part_data.json', 'r') as infile:
        part_data = json.load(infile)
    return path_data,feature_data,part_data


def classify_part_types(feature_list,path_data,n_clusters):
    path_sparse_matrix = [[1 if feature in path["path"] else 0 for feature in feature_list] for path in path_data if path["total_count"] > 1000] #only cluster paths ith >1000 parts to reduce noise 
    																																			 #and one-off outlier pathways
    km = KMeans(n_clusters=n_clusters)
    km.fit(path_sparse_matrix)
    return km

def build_model_data(feature_list,type_classifier,part_data):
    part_type_features = {}
    for i,part in enumerate(part_data[0:1000]):
        if i % 10000 == 0:
            print (str(round(i/len(part_data)*100,0)) + "%")
        sparse_matrix = np.array([1 if feature in part["features"] else 0 for feature in feature_list])
        part_type = str(type_classifier.predict(sparse_matrix)[0])
        if part_type in part_type_features.keys():
            part_type_features[part_type] = [max(x) for x in zip(part_type_features[part_type], sparse_matrix)] #take max of old feature matrix and new sample(0 or 1) at each feature index to update part-type feature-set
        else:
            part_type_features[part_type] = sparse_matrix 

    all_part_type_features = {}
    part_type_data = {}
    for part_type in part_type_features.keys():
        all_part_type_features[part_type] = [feature_list[i] for i,feature_bool in enumerate(part_type_features[part_type]) if feature_bool ==1]

    for i,part in enumerate(part_data[0:1000]):
        if i % 10000 == 0:
            print (str(round(i/len(part_data)*100,0)) + "%")
        sparse_matrix = np.array([1 if feature in part["features"] else 0 for feature in feature_list])
        part_type = str(type_classifier.predict(sparse_matrix)[0])
        part_type_feature_list = all_part_type_features[part_type]
        sparse_matrix = [1 if feature in part["features"] else 0 for feature in part_type_feature_list]
        timestamp_matrix =  [part['timestamps'][feature] if feature in part["features"] else None for feature in part_type_feature_list]
        value_matrix =  [part['values'][feature] if feature in part["features"] else None for feature in part_type_feature_list]
        defective = [part['defective']]
        feature_matrix = timestamp_matrix + value_matrix + defective
        if part_type in part_type_data.keys():
            part_type_data[part_type].append(feature_matrix)
        else:
            part_type_data[part_type] = [feature_matrix]
    n_clusters = type_classifier.get_params()["n_clusters"]
    with open('data/' + str(n_clusters) + '_part_data.json', 'w') as outfile:
        json.dump(part_type_data,outfile)

def load_model_data(n_clusters):
    with open('data/' + str(n_clusters) + '_part_data.json', 'r') as infile:
        model_data = json.load(infile)
    return model_data

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)

def eval_mcc(y_true, y_prob, show=False):
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true) # number of positive
    numn = n - nump # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    best_id = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx <= i are predicted negative while others are predicted positive
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
        new_mcc = mcc(tp, tn, fp, fn)
        mccs[i] = new_mcc
        if new_mcc >= best_mcc:
            best_mcc = new_mcc
            best_id = i
    if show:
        best_proba = y_prob[idx[best_id]]
        y_pred = (y_prob > best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        print("best MCC " + str(best_mcc))
        return best_mcc

def mcc_eval(estimator, X, y):
    y_true = y
    y_prob = estimator.predict(X)
    best_mcc = eval_mcc(y_true, y_prob)
    return best_mcc

def build_classifier(X,y):
    
    pipe = Pipeline([("imputer", Imputer(missing_values="NaN")),
                     ("clf",xgb.XGBClassifier(nthread=2))
                   ])
    
    #when in doubt, use xgboost
    
    parameters = {'imputer__strategy':("mean","median","most_frequent"),
                  'clf__objective':['binary:logistic'],
                  'clf__learning_rate': [0.15], #so called `eta` value
                  'clf__max_depth': [6],
                  'clf__min_child_weight': [3,11],
                  'clf__silent': [1],
                  'clf__subsample': [0.9],
                  'clf__colsample_bytree': [0.3],
                  'clf__n_estimators': [100], #number of trees
                  'clf__seed': [1337]}


    #should evaluate by train_eval instead of the full dataset
    clf = GridSearchCV(pipe, parameters, n_jobs=2, 
                       cv=StratifiedKFold(y, n_folds=5, shuffle=True), 
                        scoring=mcc_eval,
                       verbose=99, refit=True)

    clf.fit(X, y)
    return clf

#feature_list = [data["feature"] for data in feature_data]
# part_clf = classify_part_types(path_data,3)
# build_model_data(part_clf,part_data)

if  __name__ == "__main__":
    model_data = np.array(load_model_data(3)["0"])
    X = model_data[:,0:-1]
    y = model_data[:,-1].astype(int)
    build_classifier(X,y)
