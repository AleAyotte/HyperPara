import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.svm import SVC
from classifiers.abstract_classifier import AbstractClassifier


class SVMClassifier(AbstractClassifier):

    def __init__(self, kernel='rbf', C=1, gamma=1):
        super().__init__(SVC(probability=True, kernel=kernel, gamma=gamma, C=C))


if __name__ == "__main__":
    print(SVMClassifier())