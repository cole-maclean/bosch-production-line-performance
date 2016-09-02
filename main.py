import warnings
warnings.filterwarnings('ignore')
import json
import pandas
import random
import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

chunksize = 10 ** 4
path_data = {}
feature_data = {}
page = 0
reader_numeric = pandas.read_csv('data/train/train_numeric.csv', chunksize=chunksize)
reader_categorical = pandas.read_csv('data/train/train_categorical.csv', chunksize=chunksize)
reader_date = pandas.read_csv('data/train/train_date.csv', chunksize=chunksize)
reader = zip(reader_numeric, reader_categorical, reader_date)
for numeric, categorical, date in reader:
    print ("page " + str(page))
    page = page + 1
    numeric_chunk = pandas.DataFrame(numeric)
    categorical_chunk = pandas.DataFrame(categorical)
    date_chunk = pandas.DataFrame(date)
    cat_columns = list(categorical_chunk.columns.values)
    num_columns = list(numeric_chunk.columns.values)
    for index, part in date_chunk.iterrows():
        part_id = int(part['Id'])
        old_timestamp_indx = 0
        path = []
        defective = int(numeric_chunk['Response'][index])
        for feature,timestamp in part.items():
            if feature != "Unnamed: 0" and feature != "Id":
                line = feature.split("_")[0]
                station = feature.split("_")[1]
                if pandas.notnull(timestamp):
                    split_feature = feature.split("D")
                    timestamp_indx = int(split_feature[1])
                    for i in reversed(range(old_timestamp_indx + 1, timestamp_indx)):
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
                                if defective == 1:
                                    feature_data[feature_indx]["defect_values"].append(feature_val)
                                elif random.random() <= 0.99:
                                    feature_data[feature_indx]["values"].append(feature_val)
                            else:
                                feature_data[feature_indx] = {"total_count":1,"defective_count":defective,"values":[],"defect_values":[],"feature_type":feature_type}
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
            else:
                path_data[path] = {"total_count":1,"defective_count":sorted_path[i]["defective"],"start_feature":sorted_path[i]["feature"] ,"end_feature":sorted_path[i+1]["feature"]}
    break

x = np.linspace(-1, 1, 25)
for feature,data in feature_data.items():
    if data["feature_type"] == "numeric" and len(data["defect_values"]) >= 0:
        val_mean = np.mean(data["values"])
        val_std = np.std(data["values"])
        defect_mean = np.mean(data["defect_values"])
        defect_std = np.std(data["defect_values"])
        data["gauss"] = list(gaussian(x,val_mean,val_std))
        data["defect_gauss"] = list(gaussian(x,defect_mean,defect_std))

with open('data/path_data.json', 'w') as outfile:
    json.dump(path_data, outfile)
with open('data/feature_data.json', 'w') as outfile:
    json.dump(feature_data, outfile)