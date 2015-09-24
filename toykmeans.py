#!/usr/bin/python2.7
from abstractclassifier import AbstractClassifier
import numpy as np
import random

class ToyKMeans(AbstractClassifier):
    """ToyKMeans

    Used External Documentation
    ---------------------------
    - Book: Collective Intelligence, Segaran, O'Reilly
    - Sklearn Cython Source: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/cluster/_k_means.pyx


    """

    def __init__(self, n_clusters, n_iterations=2, seed=None, centroids=[]):
        self.n_clusters_=n_clusters
        self.n_iterations_=n_iterations
        self.centroids_=np.array(centroids)
        self.inertia_=0
        random.seed(seed)

    def fit(self, train_data, metric=None):
        """Run the K-means clustering on train_data

        Parameters
        ----------
        train_data : array-like, shape (n_samples, n_features)

        Returns
        -------
        clusters : array-like, shape (n_clusters, n_features)
        """
        
        if metric==None:
            self.metric=self.euclidean_sqr

        if self.centroids_.shape[0]==0:
            centroids=self.random_init(train_data)
        else:
            centroids=self.centroids_

        # remove mean from data
        #train_data_mean=np.mean(train_data,axis=0)
        #train_data=train_data-train_data_mean
        # row norms??
        #train_data_sqr_norms = np.einsum('ij,ij->i', train_data, train_data)


        old_centroids=np.zeros(centroids.shape)

        # iterate until no change in cluster centers or defined number of iterations is reached
        n_iterations=self.n_iterations_
        while n_iterations>0 and np.array_equal(centroids,old_centroids)==False:
            n_iterations-=1
            old_centroids=centroids
            centroids=self.fit_iteration(train_data, centroids)
        
        self.centroids_=centroids
        return centroids
        
    def fit_iteration(self, train_data, centroids):
    
        train_data_centroid_idx=np.zeros( (train_data.shape[0],2) )
        dist=0.
        for i in range(train_data.shape[0]):
            dists=[]
            for c in range(centroids.shape[0]):
                # compute distance between current train_data instance and centroid
                dists.append( self.metric( instance=train_data[i,:], centroid=centroids[c,:]) )

            # assign instance to closest centroid
            train_data_centroid_idx[i,:]=np.array([ dists.index(min(dists)), min(dists)])

            # inertia i.e. total distance
            dist+=min(dists)

        self.inertia_=dist
        # extract instances with largest distances
        distances=train_data_centroid_idx[:,1]
        distances_idx=distances.argsort()[::-1]
        
        # new centroid positions
        new_centroids=np.zeros(centroids.shape)

        # centroids with no assigned points are assigned the farthest points from the other centroids
        # note this allows that this point is attributed to two centroids
        pc=0
        for c in range(centroids.shape[0]):
            if c not in train_data_centroid_idx[:,1] and pc<train_data.shape[0]:
                new_centroids[c,:]=train_data[distances_idx[pc],:]
                pc+=1

        # move clusters such that the distance to all assigned points is minimized
        for c in range(centroids.shape[0]):
            points_of_centroid=train_data[train_data_centroid_idx[:,0]==c,:]
            if points_of_centroid.shape[0]>0:
                new_centroids[c,:]=np.mean(points_of_centroid,axis=0)


        return new_centroids

    def predict(self, test_data):
        """ Predict to which clusters the instances in test_data belong
        """
        if self.centroids_.shape[0]==0:
            raise ValueError("No centroids present. Run KMeans.fit first.")

        print test_data.shape
        part_of_cluster=np.zeros(test_data.shape[0])
        for i in range(test_data.shape[0]):
            dists=[]
            for c in range(self.centroids_.shape[0]):
                # compute distance between current train_data instance and each cluster
                dists.append( self.metric( instance=test_data[i,:], centroid=self.centroids_[c,:]) )
            
            # assign point to cluster with minimal distance
            part_of_cluster[i]=dists.index(min(dists))

        return part_of_cluster

    def random_init(self, train_data):
        """Provides random initialization of clusters using dimension of train_data

        Intializes self.n_clusters_ for each dimension randomly using the ranges
        used by the features in the data set

        Parameters
        ----------
        train_data : array-like, shape (n_samples, n_features)

        Returns
        -------
        centroids: array-like, shape (n_clusters, n_features)
        """

        centroids=np.zeros((self.n_clusters_, train_data.shape[1]))
        for c in range(self.n_clusters_):
            for f in range(train_data.shape[1]):
                centroids[c,f]=random.uniform(min(train_data[:,f]), max(train_data[:,f]))

        return centroids


    def euclidean_sqr(self, instance, centroid):
        """ calculate euclidean distance between instance and cluster
        """
        return np.linalg.norm(instance-centroid)**2
        

if __name__=="__main__":
    data=np.array([ [1,1], [1.5,1.5], [1,1.5],
                    [4,4], [4.5,4.5], [1,1.7]], dtype=np.float)

    centroids=[[1,1], [2,2],[3,3]]

    #data=np.array([ [1], [1.5], [1.7],
    #                [4], [4.5], [1.2]], dtype=np.float)

    #centroids=[[1], [2],[200]]
    km=ToyKMeans(n_clusters=3, centroids=centroids)
    print "toykmeans"
    print km.fit(data)
    #print km.predict(np.array([data[3]]))

    from sklearn.cluster import KMeans
    skm=KMeans(n_clusters=3,init=np.array(centroids), max_iter=2, n_init=1, verbose=True)
    skm.fit(data)
    print "skm"
    print skm.cluster_centers_
    print skm.labels_
    skm.predict(np.array([data[3]]))

