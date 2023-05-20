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

import utils
from preprocess import Preprocess


class EegNet(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        """Initializes multi-class classification model.

        Args:
          input_features (int): Number of input features to the model
          output_features (int): Number of outputs features (number of output classes)
          hidden_units (int): Number of hidden units between layers, default 8

        Returns:

        Example:
        """
        super().__init__()
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )

    def forward(self, x):
        return self.linear_layer_stack(x)


def get_model_metrics(test_labels, model_predictions):
    accuracy = accuracy_score(test_labels, model_predictions)
    precision = precision_score(
        test_labels,
        model_predictions,
        average="macro",
        zero_division=0)
    recall = recall_score(test_labels,
                          model_predictions,
                          average="macro",
                          zero_division=0)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    # Create the confusion matrix
    cm = confusion_matrix(test_labels, model_predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(15, 15))
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
    print("starting RF prediction")
    model_predictions = rf.predict(test_featuresRF)
    get_model_metrics(test_labelsRF, model_predictions)


def NN(
        train_featuresNN,
        test_featuresNN,
        train_labelsNN,
        test_labelsNN,
        input_features):
    print("Neural network training started")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Create istance of the net
    model_3 = EegNet(input_features=input_features,
                     hidden_units=768,
                     output_features=21).to(device)
    # Set loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model_3.parameters(),
                                  lr=0.0001)

    # Tensor-ise data
    train_features_tensor = torch.tensor(
        train_featuresNN.values).to(
        torch.float32).to(device)
    test_features_tensor = torch.tensor(
        test_featuresNN.values).to(
        torch.float32).to(device)
    train_labels_tensor = torch.tensor(
        train_labelsNN.values).to(
        torch.long).to(device)
    test_labels_tensor = torch.tensor(test_labelsNN.values).to(torch.long).to(device)
    # Correct for labels 1 indexing
    train_labels_tensor = train_labels_tensor - 1
    test_labels_tensor = test_labels_tensor - 1
    # Store accuracy to plot it
    test_accuracy_points = []
    train_accuracy_points = []
    # Store loss to plot it
    test_loss_points = []
    train_loss_points = []

    # Set a manual seed
    torch.manual_seed(42)
    # Loop through data
    epochs = 2000
    for epoch in range(epochs):
        # Training
        y_logits = model_3(train_features_tensor.transpose())
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits, train_labels_tensor)
        acc = accuracy_fn(y_true=train_labels_tensor,
                          y_pred=y_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing
        with torch.inference_mode():
            test_logits = model_3(test_features_tensor)
            test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

            test_loss = loss_fn(test_logits, test_labels_tensor)
            test_acc = accuracy_fn(y_true=test_labels_tensor,
                                   y_pred=test_preds)

        test_accuracy_points.append(test_acc)
        train_accuracy_points.append(acc)
        test_loss_points.append(test_loss)
        train_loss_points.append(loss.detach().cpu().numpy())

        if epoch % 10 == 0:
            print(
                f"Epoch: {epoch} | Loss: {loss:.4f}, Acc: {acc:.2f}% | Test loss: {test_loss:.4f}, Test acc: {test_acc:.2f}%")

    # Get a confusion matrix for the NN
    y_logits = model_3(test_features_tensor)
    model_predictions = torch.softmax(y_logits, dim=1).argmax(dim=1)
    # adjust for equality between 0 and 1 indexing
    model_predictions += 1
    # Get metrics and confusion matrix for the NN
    get_model_metrics(test_labelsNN, model_predictions.cpu())
    # Plot the accuracy
    plt.plot(train_accuracy_points)
    plt.plot(test_accuracy_points)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # Plot the loss
    plt.plot(train_loss_points)
    plt.plot(test_loss_points)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


# Experiment 1: Very naive approach, merge all the data and to a train test split Problem found: Metrics are
# inflated, the net appears able to learn how to identify a person, but it
# actually just learns to distinguish a session
def Experiment1(cumulative_df):
    cumulative_df.drop("Session", axis=1, inplace=True)
    # split again the features and the labels
    features = cumulative_df.drop('Subject', axis=1)
    labels = cumulative_df['Subject']
    # split in train and test set
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.30)
    # RF(train_features, test_features, train_labels, test_labels)
    # SVM(train_features, test_features, train_labels, test_labels)
    # NN(train_features, test_features, train_labels, test_labels,168)


# Experiment 2: Use an entire session as the test set. This shows the
# previous approach is wrong and gives a baseline for the next experiments
def Experiment2(preserve_df):
    # Experiment 2, use an entire session for each subject as the test data
    # Divide the sessions
    session1 = preserve_df[preserve_df.Session == 1]
    session2_3 = preserve_df[preserve_df.Session != 1]
    # Remove the session column from both sets
    train = session2_3.drop("Session", axis=1)
    test = session1.drop("Session", axis=1)
    # Split features and labels
    train_features = train.drop('Subject', axis=1)
    train_labels = train['Subject']
    test_features = test.drop('Subject', axis=1)
    test_labels = test['Subject']
    # RF(train_features, test_features, train_labels, test_labels)
    # SVM(train_features, test_features, train_labels, test_labels)
    # NN(train_features, test_features, train_labels, test_labels,168)


# Experiment 3: like Experiment 2, but we use only the file that gave the best result according to the paper.
# This significantly improves the results of experiment 2
def Experiment3():
    combined_df = utils.load_file("MFCC_rest_closed.mat")
    combined_df.dropna(inplace=True)
    session1 = combined_df[combined_df.Session == 1].copy()
    session2_3 = combined_df[combined_df.Session != 1].copy()
    # Remove the session column from both sets
    train = session2_3.drop("Session", axis=1)
    test = session1.drop("Session", axis=1)
    # Split features and labels
    train_features = train.drop('Subject', axis=1)
    train_labels = train['Subject']
    test_features = test.drop('Subject', axis=1)
    test_labels = test['Subject']
    # RF(train_features, test_features, train_labels, test_labels)
    # SVM(train_features, test_features, train_labels, test_labels)
    # NN(train_features, test_features, train_labels, test_labels,168)


# Experiment 4: Same as experiment 1 but with an original pre-processed dataset
def Experiment4():
    Preprocess()
    exp4_df = pd.read_csv("Preprocessed_dataset.csv")
    exp4_df.drop("Session", axis=1, inplace=True)
    # split again the features and the labels
    features = exp4_df.drop('Subject', axis=1)
    labels = exp4_df['Subject']
    # split in train and test set
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.30)
    print("starting the classification")
    RF(train_features, test_features, train_labels, test_labels)
    # NN(train_features, test_features, train_labels, test_labels, 11)


if __name__ == '__main__':
    # cumulative_df = utils.load_data()
    # # remove columns with null values
    # cumulative_df.dropna(inplace=True)
    # Keep a version of the Dataframe with the Sessions intact for experiment 2
    # preserve_df = cumulative_df.copy()
    # Experiment1(cumulative_df)
    # Experiment2(preserve_df)
    # Experiment3()
    Experiment4()
