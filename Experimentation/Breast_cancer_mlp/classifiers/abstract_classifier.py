import os, sys
from Experimentation.Breast_cancer_mlp.data_loader.data_loader import DataLoader


class AbstractClassifier:

    """
    Parent class of all project classifiers.
    Attributes:
        model : An object that defines the classifier classifiers to implement.
        X_train : The features of the training data
        Y_train : The targets of training data (the ground truth label)
        X_test :  The features of the testing data
        Y_test : The targets of training data (the ground truth label)
    """

    def __init__(self, model, dataset):
        self.model = model
        self.data_loader = DataLoader(dataset=dataset)
        self.X_train = self.data_loader.X_train
        self.X_test = self.data_loader.X_test
        self.Y_train = self.data_loader.Y_train
        self.Y_test = self.data_loader.Y_test

    def train(self):
        self.model = self.model.fit(self.X_train, self.Y_train)

    def evaluate(self, label="Training"):
        if label == 'Training':
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test
        return 1 - self.model.score(x, y)
