import warnings
warnings.filterwarnings('ignore')
import json
import pandas
import random
import numpy as np
import hashlib

filtered_features = ["Unnamed: 0","Id"]

def parse_train_data(percent):
    chunksize = 10 ** 4#chunk size for pandas to keep data load within memory constraints
    path_data = {}
    edge_data = {}
    feature_data = {}
    part_data = []
    page = 0 #chunk counter
    reader_numeric = pandas.read_csv('data/train/train_numeric.csv', chunksize=chunksize)
    reader_categorical = pandas.read_csv('data/train/train_categorical.csv', chunksize=chunksize)
    reader_date = pandas.read_csv('data/train/train_date.csv', chunksize=chunksize)
    reader = zip(reader_numeric, reader_categorical, reader_date) #combine 3 datasets
    for numeric, categorical, date in reader:
        print ("page " + str(page))
        page = page + 1
        numeric_chunk = pandas.DataFrame(numeric)
        categorical_chunk = pandas.DataFrame(categorical)
        date_chunk = pandas.DataFrame(date)
        cat_columns = list(categorical_chunk.columns.values) #store list of categorical and numeric feature labels
        num_columns = list(numeric_chunk.columns.values)
        for index, part in date_chunk.iterrows():
            feature_list = []
            value_list = {}
            timestamp_list = {}
            part_id = int(part['Id'])
            old_timestamp_indx = 0
            path = []
            defective = int(numeric_chunk['Response'][index])
            #iterate over ever part in the date file, using the date features to lookup the associated numeric and categorical feature belonging to this timestamp
            for feature,timestamp in part.items():
                if feature not in filtered_features:
                    line = feature.split("_")[0]
                    station = feature.split("_")[1]
                    if pandas.notnull(timestamp):
                        split_feature = feature.split("D")
                        timestamp_indx = int(split_feature[1])#get the index of the timestamp feature to be used as lookup for cat and num features
                        for i in reversed(range(old_timestamp_indx + 1, timestamp_indx)): #cat and num features are timestamped by the date feature index immediately following the cat or num feature index, 
                                                                                          #iterate backwards from current timestamp and look up cat and numeric features until both are null or until the previous timestamp
                            feature_indx = split_feature[0] + "F" + str(i)
                            if feature_indx in cat_columns:
                                if pandas.notnull(categorical_chunk[feature_indx][index]):
                                    feature_val = categorical_chunk[feature_indx][index].split("T")[1] #trim T off categorical value to induce numeric
                                else:
                                    feature_val = categorical_chunk[feature_indx][index]
                                feature_type = "categorical"
                            elif feature_indx in num_columns:
                                feature_val = numeric_chunk[feature_indx][index]
                                feature_type = "numeric"
                            else:
                                feature_val = None
                            if pandas.notnull(feature_val):
                                feature_list.append(feature_indx)
                                value_list[feature_indx] = feature_val
                                timestamp_list[feature_indx] = timestamp
                                if feature_indx in feature_data.keys():
                                    feature_data[feature_indx]["total_count"] = feature_data[feature_indx]["total_count"] + 1
                                    feature_data[feature_indx]["defective_count"] =  feature_data[feature_indx]["defective_count"] + defective
                                    feature_data[feature_indx]["defective_rate"] =  feature_data[feature_indx]["defective_count"]/feature_data[feature_indx]["total_count"]
                                else:
                                    feature_data[feature_indx] = {"total_count":1,"defective_count":defective,"defective_rate":defective,
                                                                  "feature_type":feature_type,"station":station,"line":line,"feature":feature_indx,"example_val":feature_val}
                                path.append({"feature":feature_indx,"timestamp":timestamp,"station":station,"value":feature_val,"feature_type":feature_type,"defective":defective})
                            else:
                                old_timestamp_indx = timestamp_indx
                                break
            if random.random() <= percent or defective == 1:
                part_data.append({'features':feature_list,'values':value_list,'timestamps':timestamp_list,'defective':defective})
            sorted_path = sorted(path, key=lambda k: k['timestamp'])
            if sorted_path:
                path_string = "".join([path["feature"] for path in sorted_path])
                path_hash = hashlib.md5(path_string.encode()).hexdigest()
                if path_hash in path_data.keys():
                    path_data[path_hash]["total_count"] =  path_data[path_hash]["total_count"] + 1
                    path_data[path_hash]["defective_count"] =  path_data[path_hash]["defective_count"] + sorted_path[0]["defective"]
                    path_data[path_hash]["defective_rate"] =  path_data[path_hash]["defective_count"]/path_data[path_hash]["total_count"]
                else:
                    path_data[path_hash] = {"total_count":1,"defective_count":sorted_path[0]["defective"],"defective_rate":sorted_path[0]["defective"],"path":[path["feature"] for path in sorted_path]}

                for i in range(len(sorted_path) -1):
                    edge = sorted_path[i]["feature"] + "-" + sorted_path[i+1]["feature"]
                    if edge in edge_data.keys():
                        edge_data[edge]["total_count"] =  edge_data[edge]["total_count"] + 1
                        edge_data[edge]["defective_count"] =  edge_data[edge]["defective_count"] + sorted_path[i]["defective"]
                        edge_data[edge]["defective_rate"] =  edge_data[edge]["defective_count"]/edge_data[edge]["total_count"]
                    else:
                        edge_data[edge] = {"total_count":1,"defective_count":sorted_path[i]["defective"],"defective_rate":sorted_path[i]["defective"],
                                            "start_feature":sorted_path[i]["feature"] ,"end_feature":sorted_path[i+1]["feature"]}
                                                            
    edge_list = [data for data in edge_data.values()] #convert to list of dicts for d3.js consumption
    path_list = sorted([data for data in path_data.values()],key=lambda k:k['total_count'],reverse=True)
    #remove feature values to reduce dataset for visulization
    all_feature_list = sorted([{data_key:feature_data for data_key,feature_data in data.items() if data_key != 'values'} for key,data in feature_data.items()],key=lambda k:k['feature'])

    with open('edge_data.json', 'w') as outfile:
        json.dump(edge_list, outfile)
    with open('path_data.json', 'w') as outfile:
        json.dump(path_list, outfile)
    with open('data/part_data.json', 'w') as outfile:
        json.dump(part_data, outfile)

def parse_test_data():
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
                if feature not in filtered_features:
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

parse_train_data(0.2)