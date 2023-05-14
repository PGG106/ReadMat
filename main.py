# This is a sample Python script.
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn import svm
from scipy.io import loadmat
# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from torch import nn


class EegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=168, out_features=512)
        self.layer_2 = nn.Linear(in_features=512, out_features=512)
        self.layer_3 = nn.Linear(in_features=512, out_features=20)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))


def get_model_metrics(test_labels, model_predictions):
    accuracy = accuracy_score(test_labels, model_predictions)
    precision = precision_score(test_labels, model_predictions, average="macro")
    recall = recall_score(test_labels, model_predictions, average="macro")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    # Create the confusion matrix
    cm = confusion_matrix(test_labels, model_predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(11, 11))
    disp.plot(ax=ax, cmap="binary")

    plt.show()


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def SVM(train_featuresSVM, test_featuresSVM, train_labelsSVM, test_labelsSVM):
    # Create a svm Classifier
    clf = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    clf.fit(train_featuresSVM, train_labelsSVM)

    # Predict the response for test dataset
    model_predictions = clf.predict(test_featuresSVM)
    get_model_metrics(test_labelsSVM, model_predictions)


def RF(train_featuresRF, test_featuresRF, train_labelsRF, test_labelsRF):
    # Create a random forest classifier
    rf = RandomForestClassifier()
    rf.fit(train_featuresRF, train_labelsRF)
    model_predictions = rf.predict(test_featuresRF)
    get_model_metrics(test_labelsRF, model_predictions)


if __name__ == '__main__':
    list_of_files = os.listdir("data\\Identification\\MFCC\\")
    cumulative_df = pd.DataFrame()
    for file in list_of_files:
        data_set = loadmat("data\\Identification\\MFCC\\" + file)
        features = data_set['feat']
        labels = data_set['Y']
        features_df = pd.DataFrame(features)
        labels_df = pd.DataFrame(labels, columns=["Subject", "Session"])
        combined_df = pd.concat([features_df, labels_df], axis=1)
        cumulative_df = pd.concat([cumulative_df, combined_df]).sort_values(by="Subject")
    # remove columns with null values
    cumulative_df.dropna(inplace=True)
    cumulative_df.drop("Session", axis=1, inplace=True)
    # split again the features and the labels
    features = cumulative_df.drop('Subject', axis=1)
    labels = cumulative_df['Subject']
    # split in train and test set
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.30)
    train_labels.sort_values().value_counts(sort=False).plot.bar()
    # plt.show()

    # RF(train_features, test_features, train_labels, test_labels)
    # SVM(train_features, test_features, train_labels, test_labels)
    # Create istance of the net
    model_3 = EegNet()
    # Set loss function and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model_3.parameters(),
                                lr=0.1)
    # Tensor-ise data
    train_features_tensor = torch.tensor(train_features.values).to(torch.float32)
    test_features_tensor = torch.tensor(test_features.values).to(torch.float32)
    train_labels_tensor = torch.tensor(train_labels.values).to(torch.float32)
    test_labels_tensor = torch.tensor(test_labels.values).to(torch.float32)
    # Set a manual seed
    torch.manual_seed(42)
    # Loop through data
    epochs = 10

    for epoch in range(epochs):
        # Training
        model_3.train()

    # CNN dataset construction, proof that it's not doable
    # load a file (TODO: do this on all files)
    # data_set = loadmat("data\\Identification\\MFCC\\MFCC_vepc10.mat")
    # features = data_set['feat']
    # labels = data_set['Y']
    # features_df = pd.DataFrame(features)
    # labels_df = pd.DataFrame(labels, columns=["Subject", "Session"])
    # combined_df = pd.concat([features_df, labels_df], axis=1)
    # # Split the file based on subject and session
    # subjects = combined_df.groupby(["Subject","Session"])
    # for group in subjects:
    #     keys=group[1].keys()
    #     signal = group[1][1]
    #     print(signal[:1024])
    #     plt.plot(signal[:1024])
    #     plt.show()
