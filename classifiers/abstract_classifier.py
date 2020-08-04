import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from cross_validation.cross_validation import CrossValidation


class AbstractClassifier:

    """
    Parent class of all project classifiers.
    Attributes:
        model : An object that defines the classifier classifiers to implement.
        metrics : An object that defines the different metrics that can be used to evaluate a classifiers.
        X_train : The features of the training data
        Y_train : The targets of training data (the ground truth label)
        X_test :  The features of the testing data
        Y_test : The targets of training data (the ground truth label)
    """

    def __init__(self, model=None, hyperparameters=None):
        self.model = model
        self.hyperparameters = hyperparameters
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(load_iris().data, load_iris().target, test_size = 0.25, random_state = 42)

    def train(self):
        self.model = self.model.fit(self.X_train, self.Y_train)

    def predict(self, x):
        return self.model.predict(x)

    def evaluate(self, label="Training"):

        if label == 'Training':
            x, y = self.X_train, self.Y_train
        else:
            x, y = self.X_test, self.Y_test

        print(label + ' accuracy', round(self.model.score(x, y) * 100, 2), " %")

    def tunning_model(self, hyperparameters, kfold):
        cross_validate_model = CrossValidation(self.model, hyperparameters, kfold)
        cross_validate_model.fit_and_predict(self.X_train, self.Y_train, self.X_test, self.Y_test)
        return cross_validate_model.get_score(self.X_test, self.Y_test)
