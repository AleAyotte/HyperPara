import os, sys
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from sklearn.neural_network import MLPClassifier
from classifiers.abstract_classifier import AbstractClassifier


class FullyConnectedClassifier(AbstractClassifier):

    def __init__(self, hidden_layer_sizes=(10,), activation='relu', alpha=1e-2, learning_rate_init=1e-3, solver='adam', mode='0'):
        model = MLPClassifier(
                random_state=None,
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                solver=solver,
                max_iter=5000
                )
        super().__init__(model, mode)