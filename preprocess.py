import os

import imblearn.under_sampling
import pandas as pd
from scipy.io import loadmat
import numpy as np
from pyprep.prep_pipeline import PrepPipeline
import matplotlib.pyplot as plt
import re
import mne
from sklearn.decomposition import PCA
from scipy.signal import butter, filtfilt
import pywt
from sklearn.decomposition import PCA
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours


def get_subject(filename):
    # initializing substrings
    sub1 = "s"
    sub2 = "_"
    print(filename)
    # getting index of substrings
    idx1 = filename.index(sub1) - 1
    idx2 = filename.index(sub2)

    # length of substring 1 is added to
    # get string from next character
    res = filename[idx1 + len(sub1) + 1: idx2]
    return int(res)


def get_session(filename):
    # initializing substrings
    sub1 = "_s"
    sub2 = "."

    # getting index of substrings
    idx1 = filename.index(sub1) - 1
    idx2 = filename.index(sub2)

    # length of substring 1 is added to
    # get string from next character
    res = filename[idx1 + len(sub1) + 1: idx2]
    return int(res)


def Preprocess():
    list_of_files = os.listdir("data\\RAW_PARSED\\")
    cumulative_df = pd.DataFrame()
    for file in list_of_files:
        data_set = loadmat("data\\RAW_PARSED\\" + file)['recording']
        # Transpose to get a signal in each row
        df = pd.DataFrame(data_set).transpose().copy()
        # Apply filter
        b, a = butter(3, [0.02, 0.4], 'band', fs=256.0)
        signal = filtfilt(b, a, df)
        coeffs = pywt.wavedec(signal, 'db1', mode='sym', level=2)
        coeff_df = pd.DataFrame(coeffs[0])
        coeff_df = coeff_df.transpose()
        # Reduce the dataset dimensionality
        pca = PCA(n_components=11)
        coeff_df = pd.DataFrame(pca.fit_transform(coeff_df))
        # Add subject and Session labels
        coeff_df['Subject'] = get_subject(file)
        coeff_df['Session'] = get_session(file)
        cumulative_df = pd.concat([cumulative_df, coeff_df])

    # Apply prototype selection
    print("Starting Undersampling")
    labels = cumulative_df['Subject']
    cumulative_df.columns = cumulative_df.columns.astype(str)
    nm = imblearn.under_sampling.NearMiss()
    ps_feat, ps_lab = nm.fit_resample(cumulative_df.values, labels)
    df = pd.DataFrame(ps_feat)
    columns_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'Subject', 'Session']
    df.columns = columns_names
    df = df.iloc[::8]
    print(df)
    print("saving preprocessed dataset")
    df.to_csv("Preprocessed_dataset.csv", index=False)
    print("Dataset saved")
