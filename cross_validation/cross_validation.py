import os, sys
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from metrics.metrics import Metrics
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))


class CrossValidation:
    """
    This class does k cross validation
    """

    def __init__(self, model, hyperparameters, kfold):
        self.metrics = Metrics()
        cross_validation = StratifiedKFold(n_splits=kfold, shuffle=True)
        self.clf = GridSearchCV(model, hyperparameters, cv=cross_validation, n_jobs=-1, verbose=1)

    def fit_and_predict(self, x_train, y_train, x_test, y_test):
        prediction = self.clf.fit(x_train, y_train).best_estimator_.predict(x_test)
        self.metrics.accuracy(self.clf, y=y_test, pred=prediction)

    def get_score(self, x_test, y_test):
        return round(self.clf.score(x_test, y_test) * 100, 2)
