import warnings
warnings.filterwarnings('ignore')
import json
import pandas as pd
import random
import numpy as np
import hashlib
import math
import ijson

def parse_data(dataset):
    part_data_chunks = {}
    chunksize = 10 ** 4#
    page = 0 #chunk counter
    reader_numeric = pd.read_csv('data/' + dataset + '/' + dataset + '_numeric.csv', chunksize=chunksize)
    reader_categorical = pd.read_csv('data/' + dataset + '/' + dataset + '_categorical.csv', chunksize=chunksize)
    reader_date = pd.read_csv('data/' + dataset + '/' + dataset + '_date.csv', chunksize=chunksize)
    reader = zip(reader_numeric, reader_categorical, reader_date) #combine 3 datasets
    for numeric, categorical, date in reader:
        page = page + 1
        part_data_chunks[str(page)] = []
        print ("page " + str(page))
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
                            grouped_timestamp = (timestamp/10000) + math.floor((part_id/1000))#timestamps are not continous, but repeated at an unknown interval. grouped_timestamp offsets timestamps into incremental groups
                                                                                        #every 10000 items to capture the feature of groups of defective batches in the process. Timestamp divided by 1000 to ensure
                                                                                        #batching dominates the incremental value
                            if pd.notnull(feature_val):
                                feature_list.append(feature_indx)
                                value_list[feature_indx] = feature_val
                                timestamp_list[feature_indx] = grouped_timestamp
                            else:
                                old_timestamp_indx = timestamp_indx
                                break
            if dataset == 'train':
                part_data_chunks[str(page)].append({'part_id':part_id,'features':feature_list,'values':value_list,'timestamps':timestamp_list,'defective':int(numeric_chunk['Response'][index])})
            elif dataset == 'test':
                part_data_chunks[str(page)].append({'part_id':part_id,'features':feature_list,'values':value_list,'timestamps':timestamp_list})
            else:
                print (dataset + ' ataset does not exist')
        break
    with open('data/' + dataset + '_part_data.json', 'w') as outfile:
        json.dump(part_data_chunks, outfile)

def filter_features(feature_list,dataset,n_chunks):
    part_data = []
    for n in range(1,n_chunks+1):
        with open('data/' + dataset + '_part_data.json') as f:
            json_obj = ijson.items(f,n_chunk)
            for part in json_obj:
                print(part)
                part['features'] = [feature for feature in part['features'] if feature in feature_list]
                part['values'] = {feature:value for feature,value in part['values'] if feature in feature_list}
                part['timestamps'] = {feature:timestamp for feature,timestamp in part['timestamps'] if feature in feature_list}
                part_data.append(part)
    with open('data/' + dataset + '_part_data.json', 'w') as outfile:
        json.dump(part_data, outfile)

parse_data('train')
#parse_test_data()

with open('feature_list.json', 'r') as infile:
    feature_list = set(json.load(infile)) #feature lookup list, set lookups faster then pure lists
filter_features(feature_list,'train',118)
#filter_features('data/test_part_data.json',feature_list,'test',118)