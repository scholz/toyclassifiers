#!/usr/bin/python2.7
import sys
sys.path.append("../tools/")
from time import time
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.datasets import make_classification

# Construct dataset
train_features,train_labels=make_classification(n_informative=4,n_classes=4)
test_features=train_features
test_labels=train_labels




def test_gaussian_naive_bayes(train_features, train_labels, test_features, test_labels):
    from sklearn.naive_bayes import GaussianNB
    from toygaussiannb import ToyGaussianNB

    clf=GaussianNB()
    # train sklearn bayes
    t0=time()
    clf.fit(train_features, train_labels)
    skl=clf.predict(test_features)
    print "training and predict time sklearn:", round(time()-t0, 3), "s"
    #print clf.theta_[0,:10]

    # train my bayes
    t0=time()
    clf=ToyGaussianNB()
    clf.fit(train_features, train_labels)
    myn,__=clf.predict(test_features)

    print "training and predict time toy bayes:", round(time()-t0, 3), "s"
   
    # compare results
    print "acc of sklearn naive bayes:", accuracy_score(test_labels, skl)
    print "acc of my naive bayes:", accuracy_score(test_labels, myn)
    print "matching between sklearn and my naive bayes:", accuracy_score(skl,myn)



if __name__ == "__main__":
    test_gaussian_naive_bayes( train_features, train_labels, test_features, test_labels)

