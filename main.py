import warnings
warnings.filterwarnings('ignore')
import json
import pandas
import random
import numpy as np
import hashlib

with open('features_under_1000.json', 'r') as infile: #load list of features having less then 1000 samples
    features_under_1000 = json.load(infile)

chunksize = 10 ** 4#chunk size for pandas to keep data load within memory constraints
path_data = {}
edge_data = {}
feature_data = {}
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
        part_id = int(part['Id'])
        old_timestamp_indx = 0
        path = []
        defective = int(numeric_chunk['Response'][index])
        #iterate over ever part in the date file, using the date features to lookup the associated numeric and categorical feature belonging to this timestamp
        for feature,timestamp in part.items():
            if feature != "Unnamed: 0" and feature != "Id":
                line = feature.split("_")[0]
                station = feature.split("_")[1]
                if pandas.notnull(timestamp):
                    split_feature = feature.split("D")
                    timestamp_indx = int(split_feature[1])#get the index of the timestamp feature to be used as lookup for cat and num features
                    for i in reversed(range(old_timestamp_indx + 1, timestamp_indx)): #cat and num features are timestamped by the date feature index immediately following the cat or num feature index, 
                                                                                      #iterate backwards from current timestamp and look up cat and numeric features until both are null or until the previous timestamp
                        feature_indx = split_feature[0] + "F" + str(i)
                        if feature_indx in cat_columns:
                            feature_val = categorical_chunk[feature_indx][index]
                            feature_type = "categorical"
                        elif feature_indx in num_columns:
                            feature_val = round(numeric_chunk[feature_indx][index],5)#round to 5 decimals to reduce size
                            feature_type = "numeric"
                        else:
                            feature_val = None
                        if pandas.notnull(feature_val):
                            if feature_indx in feature_data.keys():
                                feature_data[feature_indx]["total_count"] = feature_data[feature_indx]["total_count"] + 1
                                feature_data[feature_indx]["defective_count"] =  feature_data[feature_indx]["defective_count"] + defective
                                feature_data[feature_indx]["defective_rate"] =  feature_data[feature_indx]["defective_count"]/feature_data[feature_indx]["total_count"]
                                if defective == 1 or feature_indx in features_under_1000 or random.random() <= 0.25: #store all data for defects and features with < 1000 samples, otherwise downsample to 25%
                                    feature_data[feature_indx]["values"].append([part_id,timestamp,feature_val,defective])
                            else:
                                feature_data[feature_indx] = {"total_count":1,"defective_count":defective,"defective_rate":defective,
                                                              "values":[[part_id,timestamp,feature_val,defective]],"feature_type":feature_type,"station":station,"line":line,"feature":feature_indx,"example_val":feature_val}
                            path.append({"feature":feature_indx,"timestamp":timestamp,"station":station,"value":feature_val,"feature_type":feature_type,"defective":defective})
                        else:
                            old_timestamp_indx = timestamp_indx #store new timestamp to old and break loop if null data found for current timestamp
                            break
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
feature_list = sorted([{data_key:feature_data for data_key,feature_data in data.items() if data_key != 'values'} for key,data in feature_data.items()],key=lambda k:k['feature'])

with open('edge_data.json', 'w') as outfile:
    json.dump(edge_list, outfile)
with open('path_data.json', 'w') as outfile:
    json.dump(path_list, outfile)
with open('feature_data.json', 'w') as outfile:
    json.dump(feature_list, outfile)
with open('data/all_feature_data.json', 'w') as outfile:
    json.dump(feature_data, outfile)