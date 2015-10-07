#!/usr/bin/python2.7
import numpy as np
from abstractclassifier import AbstractClassifier

class ToyGaussianNB(AbstractClassifier):
    """
    Toy Gaussian Naive Bayes (GaussianNB)
    
    Algorithm
    ---------
    - Training
      - Compute priors based on prevalence of classes in train_data
      - Compute mean and variance per class per feature
    - Classification
      - Compute the probability of an instance belonging to a specific class by:
        - Iterating over all features and multiplying each iteration together,
          where each iteration is a product of:
          - the prior for the investigated class
          - the probability that the instance value comes from a normal distribution (gaussian)
            created using the mean and variance derived during training for this feature
        - To yield a valid probability (x in  {0..1}) to which class this instance belongs
          the sum of all probability products for each must be divided by the individuals products
        - The class with the highest probability is chosen as result for this instance

     Note: In the code mulitplications are replaced by summation since we are working with
           logarithms to avoid problems with small numbers.


    Used Documentation
    ------------------
    - http://www.cs.cmu.edu/~epxing/Class/10701-10s/Lecture/lecture5.pdf (using gaussian for continuous variables)
    - http://scikit-learn.org/stable/modules/naive_bayes.html
    - http://github.com/scikit-learn/scikit-learn/blob/a95203b/sklearn/naive_bayes.py (prevent std==0 by using std+=eps)
    Note: lots of optimization potential, this code is approx. 60x slower than sklearn gaussian NB


    Attributes
    ----------
    class_prior_ : array, shape (n_classes,)
        probability of each class.
    class_count_ : array, shape (n_classes,)
        number of training samples observed in each class.
    theta_ : array, shape (n_classes, n_features)
        mean of each feature per class
    sigma_ : array, shape (n_classes, n_features)
        variance of each feature per class
    
    class_names_  : array, shape (n_classes,)
        name of each class

    class_priors_ : array, shape (n_classes,)
        prior of each class

    class_feature_means_ : array, shape (n_classes, n_features)
        mean of each feature per class

    class_feature_vars_ : array, shape (n_classes, n_features)
        variance of each feature per class
    
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from toygaussiannb import ToyGaussianNB
    >>> clf = ToyGaussianNB()
    >>> clf.fit(X, Y)
    >>> print(clf.predict([[-0.8, -1]]))
    [1]
    """

    eps=1e-9

    def __init__(self):
        self.class_names_=[]
        self.class_priors_=[]
        self.class_feature_means_=[] 
        self.class_feature_vars_=[]  
    
    def gaussian(self, x,mean,var):
        prob_x= np.log( 1./np.sqrt(2*np.pi * var) ) - 0.5* ((x-mean)**2/var)
        return prob_x

    def fit(self, training_features, training_labels):

        classes=np.array(training_labels)
        # ----------------------------------------------- #
        # compute prior probabilities                     #
        # ----------------------------------------------- #
        for c in classes:
            if c not in self.class_names_:
                self.class_names_.append(c)
                self.class_priors_.append(1.)
            else:
                self.class_priors_[self.class_names_.index(c)]+=1.
        self.class_priors_=np.array(self.class_priors_, dtype=float)
        self.class_priors_/=len(classes)

        # ----------------------------------------------- #
        # compute mean and variance per class per feature #
        # ----------------------------------------------- #
        m,n=training_features.shape
        self.class_feature_means_=np.zeros((len(self.class_names_),n), dtype=float)
        self.class_feature_vars_=np.zeros((len(self.class_names_),n), dtype=float)
       
        for f in range(0,n):
            f_vect=training_features[:,f]
            for c in range(len(self.class_names_)):
                self.class_feature_means_[c, f]=np.mean(f_vect[classes==self.class_names_[c]])
                self.class_feature_vars_[c, f]=np.var(f_vect[classes==self.class_names_[c]])+self.eps

    
    def predict(self, predict):    
        # ----------------------------------------------- #
        # predict classes on predict DS                   #
        # ----------------------------------------------- #
        m,n=predict.shape
        res=[]
        res_proba=[]
        # for every row
        for r in range(0,m):
            # result vector for this row will have a log likelihood for each class
            posteriori=np.log(self.class_priors_)
            # for every feature
            for f in range(0,n):
                # for each class
                for c in range(len(self.class_names_)):
                    posteriori[c]+=self.gaussian(predict[r,f], self.class_feature_means_[c,f], self.class_feature_vars_[c,f])

            
            # argmax c (extract name of class with maximal log likelihood) 
            res.append(self.class_names_[np.where(posteriori==max(posteriori))[0]])
            res_proba.append(posteriori)
        # iterate over result to build result array
        return(res, res_proba)
    
    
    def __repr__(self):
        return "class names: %s\nclass priors: %s\nmeans: %s\nvars: %s"%(self.class_names_, self.class_priors_, self.class_feature_means_, self_class_feature_vars_)
        

