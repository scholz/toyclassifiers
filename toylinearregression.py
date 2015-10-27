#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import numpy as np
from abstractclassifier import AbstractClassifier
from sklearn import linear_model
#import linear_model.SGDRegressor as SGDRegressor


lr_raw_data=[ [2,1], [3,2], [4,3] ]
train_data=[r[:-1] for r in lr_raw_data]
train_labels=[r[-1] for r in lr_raw_data]




"""
Linear regression algorithm

1) configure approach to use (gradient descent, normal equation)
2) fit (determine thetas)
3) predict


"""

class ToyLinearRegression(AbstractClassifier):



    def __init__(self, method='ne', gd_iterations=None, alpha=None):
        """
        Attributes
        ---------
        
        method:        either 'gd' for gradient descent or 'ne' for normal equation
        gd_iterations: number of iterations for gradient descent
        alpha:         learning rate when using gd

        TODO: check if for correct mode

        """
        self.method=method
        self.gd_iterations=gd_iterations
        self.alpha=alpha
        self.thetas=[]

    def cost(self, train_data, train_labels):
        """
        Compute cost function J(θ), i.e.
        the mean summed quadradic error (MSE)
        """
        
        h_theta_tl=np.dot(train_data, self.thetas).T - train_labels
        m=train_data.shape[0]
        J=1/(2.*m) * np.sum(  np.square(h_theta_tl) )
        return J

    
    def fit(self, train_data, train_labels):
        train_data=np.array(train_data)
        # extend train_data(x1,..,xn) by x0=1 for all examples
        train_data=np.hstack((np.ones( (train_data.shape[0],1), dtype=float), train_data))
        train_labels=np.array(train_labels)
        self.thetas=np.zeros((train_data.shape[1], 1), dtype=float)

        if self.method=='gd':
            self.fit_gd(train_data, train_labels)
        else:
            self.fit_ne(train_data, train_labels)

    def feature_scaling(self, train_data, train_labels):
        """
        scale and normalize features to range around -std .. +std,
        i.e. apply X=(X-mean(X))/std(X)
        """

    def fit_gd(self, train_data, train_labels):
        """
        Use gradient descent to solve for θ, i.e. 

        repeat self.gd_iterations for all θ_i:
            theta_i = theta_i - alpha * 1/m * sum( h_theta(x)-y )*x_i (apply simultanaeously)
        """
        m=train_data.shape[0]
        for it in range(self.gd_iterations):
            
            tmp_thetas=self.thetas
            
            h_theta_tl=np.dot(train_data, tmp_thetas).T - train_labels
            for i in range(len(tmp_thetas)):
                tmp_thetas[i]=self.thetas[i] - self.alpha * 1./m * np.sum( np.multiply( h_theta_tl, train_data[:,i]) )
            self.thetas=tmp_thetas

            #print "iter:",it,"cost:", self.cost(train_data, train_labels)


    def fit_ne(self, train_data, train_labels):
        """
        Use normal equation to compute thetas, i.e.
         thetas = (X.T * X)^-1 * X.T * y
        """
        intermediate_1=np.linalg.pinv(np.dot(train_data.T, train_data))
        intermediate_2=np.dot(train_data.T, train_labels)
        self.thetas=np.dot(intermediate_1, intermediate_2)

    def get_thetas(self):
        return self.thetas


if __name__=="__main__":
    x=ToyLinearRegression()
    x.fit(train_data, train_labels)
    print x.get_thetas()    
    
    x=ToyLinearRegression(method='gd', gd_iterations=1500, alpha=0.01)
    x.fit(train_data, train_labels)
    print x.get_thetas()    
    
    SGD=linear_model.SGDRegressor(eta0=0.01, penalty='none', learning_rate='constant', loss='squared_loss', shuffle=False, n_iter=1500 )
    SGD.fit(train_data, train_labels)
    print "coef:",SGD.coef_
    print "intercept", SGD.intercept_
    print SGD.score(train_data, train_labels)



    #print x
    #print x.predict(train_data)

