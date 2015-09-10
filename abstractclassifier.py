#!/usr/bin/python2.7

class AbstractClassifier(object):

    def __init__(self):
        raise NotImplementedError()

    def fit(self, train_features, train_labels):
        raise NotImplementedError()

    def predict(self, predict):
        raise NotImplementedError()

