import json
import pandas
import random
import numpy as np

chunksize = 10 ** 4 #chunk size for pandas to keep data load within memory constraints
path_data = {}
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
                            feature_val = numeric_chunk[feature_indx][index]
                            feature_type = "numeric"
                        else:
                            feature_val = None
                        if pandas.notnull(feature_val):
                            if feature_indx in feature_data.keys():
                                feature_data[feature_indx]["total_count"] = feature_data[feature_indx]["total_count"] + 1
                                feature_data[feature_indx]["defective_count"] =  feature_data[feature_indx]["defective_count"] + defective
                                feature_data[feature_indx]["defective_rate"] =  feature_data[feature_indx]["defective_count"]/feature_data[feature_indx]["total_count"]
                                if defective == 1:
                                    feature_data[feature_indx]["defect_values"].append(feature_val)#store every defective value
                                elif random.random() <= 0.05:#randomly sample 5% of all data
                                    feature_data[feature_indx]["values"].append(feature_val)
                            else:
                                feature_data[feature_indx] = {"total_count":1,"defective_count":defective,"defective_rate":defective,
                                                              "values":[],"defect_values":[],"feature_type":feature_type,"station":station,"line":line,"feature":feature_indx}
                            path.append({"feature":feature_indx,"timestamp":timestamp,"station":station,"value":feature_val,"feature_type":feature_type,"defective":defective})
                        else:
                            old_timestamp_indx = timestamp_indx
                            break
        sorted_path = sorted(path, key=lambda k: k['timestamp'])
        for i in range(len(sorted_path) -1):
            path = sorted_path[i]["feature"] + "-" + sorted_path[i+1]["feature"]
            if path in path_data.keys():
                path_data[path]["total_count"] =  path_data[path]["total_count"] + 1
                path_data[path]["defective_count"] =  path_data[path]["defective_count"] + sorted_path[i]["defective"]
                path_data[path]["defective_rate"] =  path_data[path]["defective_count"]/path_data[path]["total_count"]
            else:
                path_data[path] = {"total_count":1,"defective_count":sorted_path[i]["defective"],"defective_rate":sorted_path[i]["defective"],
                                    "start_feature":sorted_path[i]["feature"] ,"end_feature":sorted_path[i+1]["feature"]}

path_list = [data for data in path_data.values()] #convert to list of dicts for d3.js consumption
feature_list = sorted([data for data in feature_data.values()],key=lambda k:k['feature'])

with open('data/path_data.json', 'w') as outfile:
    json.dump(path_list, outfile)
with open('data/feature_data.json', 'w') as outfile:
    json.dump(feature_list, outfile)