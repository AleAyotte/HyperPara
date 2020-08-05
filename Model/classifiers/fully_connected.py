import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from sklearn.neural_network import MLPClassifier
from Model.classifiers.abstract_classifier import AbstractClassifier


class FullyConnectedClassifier(AbstractClassifier):

    def __init__(self, hyperparameter = {
        'hidden_layer_sizes': (10,), 'alpha': 1e-2,
        'learning_rate_init': 1e-3, 'max_iter': 5000,
        'batch_size': 50
    }):

        self.hyperparameter = hyperparameter

        self.model = MLPClassifier(
                hidden_layer_sizes=self.hyperparameter['hidden_layer_sizes'],
                learning_rate_init=self.hyperparameter['learning_rate_init'],
                alpha=self.hyperparameter['alpha'],
                max_iter=self.hyperparameter['max_iter'],
                batch_size=self.hyperparameter['batch_size']
        )

        super().__init__(self.model)
