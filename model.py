import json
import pickle
import pandas as pd
import random
import numpy as np
import csv
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


def load_json(json_file):
    with open(json_file, 'r') as infile:
        json_data = json.load(infile)
    return json_data

def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)

def eval_mcc(y_true, y_prob, show=False):
    try:
        idx = np.argsort(y_prob)
    except TypeError:
        return 0
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    print("n =" + str(n))
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
    best_proba = y_prob[idx[best_id]]
    if show:
        y_pred = (y_prob > best_proba).astype(int)
        score = matthews_corrcoef(y_true, y_pred)
        print(score, best_mcc)
        plt.plot(mccs)
        return best_proba, best_mcc, y_pred
    else:
        print("best MCC " + str(best_mcc) + " at cuttoff = " + str(best_proba))
        return best_mcc,best_proba

def mcc_eval(estimator, X, y):
    y_true = y
    y_prob = estimator.predict_proba(X)[:,1]
    best_mcc,best_cutoff = eval_mcc(y_true, y_prob)
    return best_mcc

def load_model(model_file):
    with open(model_file, 'rb') as infile:
        model = pickle.load(infile)
    return model

class BoschModel():

    def __init__(self, n_part_types,subset):
        self.n_part_types = n_part_types
        self.subset = subset
        self.feature_list = load_json('feature_list.json')
        self.path_data = load_json('path_data.json')
        self.all_part_type_features = {}
        self.part_type_clf = None
        self.models = {}
        self.mcc_cutoffs = {}
        
    def classify_part_types(self,min_part_count=1000):
        path_sparse_matrix = [[1 if feature in path["path"] else 0 for feature in self.feature_list] for path in self.path_data if path["total_count"] > min_part_count] #only cluster paths with >n parts to reduce noise 
                                                                                                                                                 #and one-off outlier pathways

        km = KMeans(n_clusters=self.n_part_types)
        km.fit(path_sparse_matrix)
        self.part_type_clf = km

    def build_part_type_features(self):
        if self.part_type_clf == None:
            self.classify_part_types()

        part_data = load_json('data/part_data.json')
        part_type_features = {}

        for i,part in enumerate(part_data[0:self.subset]):
            if i % 10000 == 0:
                print (str(round(i/len(part_data)*100,0)) + "%")
            sparse_matrix = np.array([1.0 if feature in part["features"] else 0.0 for feature in self.feature_list])
            part_type = str(self.part_type_clf.predict(sparse_matrix.reshape(1,-1))[0])
            if part_type in part_type_features.keys():
                part_type_features[part_type] = [max(x) for x in zip(part_type_features[part_type], sparse_matrix)] #take max of old feature matrix and new sample(0 or 1) at each feature index to update part-type feature-set
            else:
                part_type_features[part_type] = sparse_matrix 

        self.all_part_type_features = {}
        for part_type in part_type_features.keys():
            self.all_part_type_features[part_type] = [self.feature_list[i] for i,feature_bool in enumerate(part_type_features[part_type]) if feature_bool ==1]
            print ("Part Type " + part_type + " has " + str(len(self.all_part_type_features[part_type])) + " feature")

    def build_model_data(self):
        part_data = load_json('data/part_data.json')
        part_type_data = {}
        for i,part in enumerate(part_data[0:self.subset]):
            if i % 10000 == 0:
                print (str(round(i/len(part_data)*100,0)) + "%")
            sparse_matrix = np.array([1.0 if feature in part["features"] else 0.0 for feature in self.feature_list])
            part_type = str(self.part_type_clf.predict(sparse_matrix.reshape(1,-1))[0])
            part_type_feature_list = self.all_part_type_features[part_type]
            timestamp_matrix =  [part['timestamps'][feature] if feature in part["features"] else None for feature in part_type_feature_list]
            value_matrix =  [part['values'][feature] if feature in part["features"] else None for feature in part_type_feature_list]
            feature_matrix = timestamp_matrix + value_matrix + [part['defective']]
            if part_type in part_type_data.keys():
                part_type_data[part_type].append(feature_matrix)
            else:
                part_type_data[part_type] = [feature_matrix]
        for part_type, data in part_type_data.items():
            with open('data/' + str(self.n_part_types) +  '_' + part_type + '_part_data.json', 'w') as outfile: #clusters_parttype_part_data.json
                json.dump(data,outfile)

    def load_model_data(self):
        with open('data/' + str(self.n_part_types) +  '_part_data.json', 'r') as infile:
            model_data = json.load(infile)
        return model_data

    def build_part_type_models(self):
        model_data = self.load_model_data()
        for part_type,data in model_data.items():
            with open('data/' + str(self.n_part_types) +  '_' + part_type + '_part_data.json', 'r') as infile: #clusters_parttype_part_data.json
                data = json.load(infile)
            data_array = np.array(data)
            X = data_array[:,0:-1]
            y = data_array[:,-1].astype(int)
            clf = self.build_classifier(X,y)
            self.models[part_type] = clf
            y_prob = clf.predict_proba(X)[:,1]
            best_mcc,best_cutoff = eval_mcc(y,y_prob)
            self.mcc_cutoffs[part_type] = best_cutoff

    def build_classifier(self,X,y):
        
        pipe = Pipeline([("imputer", Imputer(missing_values="NaN")),
                         ("clf",xgb.XGBClassifier(nthread=2))
                       ])
        
        #when in doubt, use xgboost
        
        parameters = {'imputer__strategy':("mean","median","most_frequent"),
                      'clf__objective':['binary:logistic'],
                      'clf__learning_rate': [0.05], #so called `eta` value
                      'clf__max_depth': [6],
                      'clf__min_child_weight': [3,11],
                      'clf__subsample': [0.75],
                      'clf__colsample_bytree': [0.5],
                      'clf__n_estimators': [200], #number of trees
                      'clf__seed': [1337]}


        clf = GridSearchCV(pipe, parameters, n_jobs=3, 
                           cv=StratifiedKFold(y, n_folds=5, shuffle=True), 
                            scoring=mcc_eval,
                           verbose=1, refit=True)

        clf.fit(X, y)
        return clf

    def get_top_n_features(self,n):
        top_n_features = {}
        for part_type,model in self.models.items():
            scores = model.best_estimator_.named_steps['clf'].booster().get_fscore()
            sorted_features = sorted(scores.items(), key=lambda x: x[1],reverse=True)
            top_n_features[part_type] = [feature_tuple[0] for feature_tuple in sorted_features[0:min(n,len(sorted_features)-1)]]
        top_n_features['all_top_features'] = list(set([feature for part_type in top_n_features.keys() for feature in top_n_features[part_type]]))
        return top_n_features

    def save_model(self):
        with open('models/' + str(self.subset) + "_subset_" + str(self.n_part_types) +  '_type_model.pkl', 'wb') as outfile:
            pickle.dump(self,outfile,pickle.HIGHEST_PROTOCOL)

    def build_submission(self):
        print ("building submission")
        with open('submissions/' + str(self.n_part_types) + '_part_types_submission.csv', 'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Id","Response"])
        chunksize = 10 ** 4#
        page = 0 #chunk counter
        reader_numeric = pd.read_csv('data/test/test_numeric.csv', chunksize=chunksize)
        reader_categorical = pd.read_csv('data/test/test_categorical.csv', chunksize=chunksize)
        reader_date = pd.read_csv('data/test/test_date.csv', chunksize=chunksize)
        reader = zip(reader_numeric, reader_categorical, reader_date) #combine 3 datasets
        for numeric, categorical, date in reader:
            print ("page " + str(page))
            page = page + 1
            numeric_chunk = pd.DataFrame(numeric)
            categorical_chunk = pd.DataFrame(categorical)
            date_chunk = pd.DataFrame(date)
            cat_columns = list(categorical_chunk.columns.values) #store list of categorical and numeric feature labels
            num_columns = list(numeric_chunk.columns.values)
            for index, part in date_chunk.iterrows():
                feature_list = []
                value_list = {}
                timestamp_list = {}
                part_id = int(part['Id'])
                old_timestamp_indx = 0
                for feature,timestamp in part.items():
                    if feature != "Unnamed: 0" and feature != "Id":
                        if pd.notnull(timestamp):
                            split_feature = feature.split("D")
                            timestamp_indx = int(split_feature[1])#get the index of the timestamp feature to be used as lookup for cat and num features
                            for i in reversed(range(old_timestamp_indx + 1, timestamp_indx)): #cat and num features are timestamped by the date feature index immediately following the cat or num feature index, 
                                                                                              #iterate backwards from current timestamp and look up cat and numeric features until both are null or until the previous timestamp
                                feature_indx = split_feature[0] + "F" + str(i)
                                if feature_indx in cat_columns:
                                    if pd.notnull(categorical_chunk[feature_indx][index]):
                                        feature_val = categorical_chunk[feature_indx][index].split("T")[1] #trim T off categorical value to induce numeric
                                    else:
                                        feature_val = categorical_chunk[feature_indx][index]
                                    feature_type = "categorical"
                                elif feature_indx in num_columns:
                                    feature_val = numeric_chunk[feature_indx][index]
                                    feature_type = "numeric"
                                else:
                                    feature_val = None
                                if pd.notnull(feature_val):
                                    feature_list.append(feature_indx)
                                    value_list[feature_indx] = feature_val
                                    timestamp_list[feature_indx] = timestamp
                                else:
                                    old_timestamp_indx = timestamp_indx
                                    break

                sparse_matrix = np.array([1.0 if feature in feature_list else 0.0 for feature in self.feature_list])
                part_type = str(self.part_type_clf.predict(sparse_matrix.reshape(1,-1))[0])
                part_type_feature_list = self.all_part_type_features[part_type]
                timestamp_matrix =  [timestamp_list[feature] if feature in feature_list else None for feature in part_type_feature_list]
                value_matrix =  [value_list[feature] if feature in feature_list else None for feature in part_type_feature_list]
                feature_matrix = np.array(timestamp_matrix + value_matrix).reshape(1,-1)
                clf = self.models[part_type]
                pred = clf.predict_proba(feature_matrix)[:,1][0]
                if pred >= self.mcc_cutoffs[part_type]:
                    response = 1
                else:
                    response=0
                with open('submissions/' + str(self.n_part_types) + '_part_types_submission.csv', 'a',newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([part_id,response])

#feature_list = [data["feature"] for data in feature_data]
# part_clf = classify_part_types(path_data,3)
# build_model_data(part_clf,part_data)

if  __name__ == "__main__":

    test_model = BoschModel(3,20000)
    test_model.classify_part_types()
    test_model.build_part_type_features()
    test_model.build_model_data()
    test_model.build_part_type_models()
    test_model.save_model()
    #test_model.build_submission()
