import os
import pandas as pd
from scipy.io import loadmat


def load_data():
    list_of_files = os.listdir("data\\Identification\\MFCC\\")
    cumulative_df = pd.DataFrame()
    for file in list_of_files:
        data_set = loadmat("data\\Identification\\MFCC\\" + file)
        features = data_set['feat']
        labels = data_set['Y']
        features_df = pd.DataFrame(features)
        labels_df = pd.DataFrame(labels, columns=["Subject", "Session"])
        combined_df = pd.concat([features_df, labels_df], axis=1)
        cumulative_df = pd.concat(
            [cumulative_df, combined_df]).sort_values(by="Subject")
    return cumulative_df


def load_file(filename):
    data_set = loadmat("data\\Identification\\MFCC\\" + str(filename))
    features = data_set['feat']
    labels = data_set['Y']
    features_df = pd.DataFrame(features)
    labels_df = pd.DataFrame(labels, columns=["Subject", "Session"])
    combined_df = pd.concat([features_df, labels_df], axis=1)
    return combined_df
