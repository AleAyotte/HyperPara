from sklearn import metrics
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class Metrics:

    def accuracy(self, model, y, x=None, label=None, pred=None):

        if pred is None:
            print(label + ' accuracy', round(model.score(x, y) * 100, 2), " %")
        else:
            print('Testing accuracy after cross-validation ', round(metrics.accuracy_score(pred, y) * 100, 2), " %")
