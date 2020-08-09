from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


class DataLoader:

    def __init__(self, test_size=0.25, random_state=None, dataset=None):

        if dataset == 'iris':
            self.data = load_iris()
        elif dataset == 'breast_cancer':
            self.data = load_breast_cancer()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            preprocessing.scale(self.data.data), self.data.target,
            test_size=test_size, random_state=random_state
        )
