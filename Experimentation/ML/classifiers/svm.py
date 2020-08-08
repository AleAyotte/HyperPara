import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.svm import SVC
from .abstract_classifier import AbstractClassifier


class SVMClassifier(AbstractClassifier):

    def __init__(self, hyperparameter={'C': 1, 'gamma': 1}, dataset=None):

        self.hyperparameter = hyperparameter
        super().__init__(SVC(probability=True, gamma=self.hyperparameter['gamma'], C=self.hyperparameter['C']), dataset)
