#!/usr/bin/python2.7
import numpy as np

raw_data=[['slashdot', 'USA', 'yes', 18, 'None'], ['google', 'France', 'yes', 23, 'Premium'], ['digg', 'USA', 'yes', 24, 'Basic'], ['kiwitobes', 'France', 'yes', 23, 'Basic'], ['google', 'UK', 'no', 21, 'Premium'], ['(direct)', 'New Zealand', 'no', 12, 'None'], ['(direct)', 'UK', 'no', 21, 'Basic'], ['google', 'USA', 'no', 24, 'Premium'], ['slashdot', 'France', 'yes', 19, 'None'], ['digg', 'USA', 'no', 18, 'None'], ['google', 'UK', 'no', 18, 'None'], ['kiwitobes', 'UK', 'no', 19, 'None'], ['digg', 'New Zealand', 'yes', 12, 'Basic'], ['slashdot', 'UK', 'no', 21, 'None'], ['google', 'UK', 'yes', 18, 'Basic'], ['kiwitobes', 'France', 'yes', 19, 'Basic']]

train_data=[r[:-1] for r in raw_data]
train_labels=[r[-1] for r in raw_data]


"""

Decision Tree Computation Steps
1) Compute Entropy for each possible split on all attributes for all available rows
2) Select the split with the best information gain
3) Add split as node to tree
4) Goto 1 using this new branch

"""


class DTNode():
    def __init__(self, tb_node=None, fb_node=None, attr_idx=-1, val=None, results=None):
        self.true_branch=tb_node
        self.false_branch=fb_node
        self.attr_idx=attr_idx
        self.val=val
        self.results=results

    def __repr__(self):
        print self.true_branch==None, self.false_branch==None, self.attr_idx, self,val, self.results

class ToyDecisionTree():
    """

    Input to predict must be a list of values

    Note: categorial variables/attributes must be provided as text
    else they will treated as continuous i.e. their numerical order will be of importance

    Documentation used:
    - Toby Segaran: Collective Intelligence
    """

    def __init__(self):
        self.tree=None

    def fit(self, train_labels, train_data, score=None):
        train_data=np.array(train_data)
        train_labels=np.array(train_labels)
        self.tree=self.generatetree(train_labels, train_data)

    def predict(self, test_data):
        test_data=np.array(test_data)
        # stay compatible with return value of DT
        return [self.predict_instance(r, self.tree) for r in test_data],[]
   
    def predict_instance(self, instance, node):

        if node.results!=None:
            return node.results[0][0] #,node.results[1]

        # if attr is numerical
        try:
            if instance[node.attr_idx] >= float(node.val):
                r=self.predict_instance( instance, node.true_branch )
            else:
                r=self.predict_instance( instance, node.false_branch )
        # if attr is a string attribute
        except:
            if instance[node.attr_idx] == node.val:
                r=self.predict_instance( instance, node.true_branch )
            else:
                r=self.predict_instance( instance, node.false_branch )
        
        return r

    def entropy(self, train_labels):
        """ lower bound of bits necessary to encode each class based on its probability """
        unique_vals, unique_counts=np.unique(train_labels, return_counts=True)
        p=unique_counts/float(len(train_labels))
        intermediate=-p*np.log2(p)
        return(np.sum(intermediate))

    def split_data(self, train_labels, train_data, attr_idx, val):
        """ split data by comparing attribute keyed by attribute index to value """

        # if attr is numerical
        try:
            train_attr=np.array(train_data[:, attr_idx], dtype=float)
            selector=(train_attr>=float(val))
        # if attr is a string attribute
        except:
            selector=(train_data[:,attr_idx] == val)

        tb_labels=train_labels[selector]
        tb_data=train_data[selector,:]
        fb_labels=train_labels[-selector]
        fb_data=train_data[-selector,:]
        return(tb_labels, tb_data, fb_labels, fb_data)
        
    def generatetree(self, train_labels, train_data, score=None):
        if score==None:
            score=self.entropy

        if train_data.shape[0]==0:
            return DTNode()

        current_score=score(train_labels)

        best_gain=0.0
        best_attr_idx=None
        best_val=None

        # find best split accross all attributes and values
        for attr_idx in range(train_data.shape[1]):

            # unique values of this attribute in training data
            attrib_vals=np.unique(train_data[:,attr_idx])

            for val in attrib_vals:
                (tb_labels, tb_data, fb_labels, fb_data)=self.split_data(train_labels, train_data, attr_idx, val)

                # compute information gain
                p=float(tb_labels.shape[0])/train_data.shape[0]
                gain=current_score - p* score(tb_labels) - (1-p)*score(fb_labels)

                if gain>best_gain and tb_labels.shape[0]>0 and fb_labels.shape[0]>0:
                    best_gain=gain
                    best_attr_idx=attr_idx
                    best_val=val
                    best_tb=(tb_labels, tb_data)
                    best_fb=(fb_labels, fb_data)

        if best_gain>0.0:
            tb_node=self.generatetree(best_tb[0], best_tb[1])
            fb_node=self.generatetree(best_fb[0], best_fb[1])
            return DTNode(tb_node=tb_node, fb_node=fb_node, attr_idx=best_attr_idx, val=best_val)

        else:
            ra,rb=np.unique(train_labels, return_counts=True)
            res= np.vstack((ra,rb)).tolist()
            return DTNode(results=res)
    
    
    def printtree(self,tree,indent):
        #print tree
        if tree.results!=None:
            print str(tree.results)
        else:
            print tree.attr_idx, ":", tree.val , "?"
            indent+=3
            print " "*indent+'T:',
            self.printtree(tree.true_branch,indent)
            print " "*indent+'F:',
            self.printtree(tree.false_branch,indent)
            
    
    def __repr__(self):
        self.printtree(self.tree, 0)
        return ""
   


if __name__=="__main__":
    x=ToyDecisionTree()
    x.fit(train_labels, train_data)
    print x
    print x.predict(train_data)
