import os, sys
import argparse
sys.path.append(os.path.dirname(os.path.join(os.getcwd())))
from classifiers.svm import SVMClassifier
from classifiers.fully_connected import FullyConnectedClassifier
my_parser = argparse.ArgumentParser()


def main():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model', action='store', type=str, required=True)
    args = my_parser.parse_args()

    if args.model == "svm":
        model = SVMClassifier()
        model.train()
        model.evaluate(label="Training")
        model.evaluate(label="Testing")

    if args.model == "MLP":
        model = FullyConnectedClassifier()
        model.train()
        model.evaluate(label="Training")
        model.evaluate(label="Testing")

if __name__ == "__main__":
    main()