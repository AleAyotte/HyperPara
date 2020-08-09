from sklearn.svm import SVC
from Experimentation.Iris_svm.classifiers.abstract_classifier import AbstractClassifier


class SVMClassifier(AbstractClassifier):

    def __init__(self, hyperparameter=None, dataset=None):

        if hyperparameter is None:
            hyperparameter = {'C': 1, 'gamma': 1}

        self.hyperparameter = hyperparameter
        super().__init__(SVC(probability=True, gamma=self.hyperparameter['gamma'], C=self.hyperparameter['C']), dataset)

