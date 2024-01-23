import numpy as np
from sklearn.utils import resample

from dtree import *

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.trees=[]

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.y=y
        self.X=X
        oob_errors=[]
        scores=[]
        in_indeces=[]
        for i in range(self.n_estimators):
            scores.append([])
        for i in range(self.n_estimators):
            index=resample(range(len(X)))
            in_indeces.append(index)
            oob_index=np.array(list(set(range(len(X)))-set(index)))        
            boot_x=X[index]
            boot_y=y[index]
            #how to check class/self type???
            if self.__class__.__name__ == 'RandomForestClassifier621':
                tree=ClassifierTree621(min_samples_leaf=self.min_samples_leaf)
                tree.fit(boot_x, boot_y)
            if self.__class__.__name__ == 'RandomForestRegressor621':
                tree=RegressionTree621(min_samples_leaf=self.min_samples_leaf)
                tree.fit(boot_x, boot_y)
            self.trees.append(tree)
            oob_errors.append(oob_index)
        if self.oob_score:
            for x in range(len(oob_errors)):
                new_trees=[]
                for i in range(len(in_indeces)):
                    if np.isin(in_indeces[i], oob_errors[x], invert=True).all():
                        new_trees.append(self.trees[i])
                for t in new_trees:
                    oob_score = self.score(X[oob_errors[x]], y[oob_errors[x]])
                    print((X[oob_errors[x]], y[oob_errors[x]]))
                    scores[x].append(oob_score)
            if self.__class__.__name__ == 'RandomForestRegressor621':
                self.oob_score_=[np.mean(scores)/150]
            else:
                self.oob_score_ = [np.mean(scores)]

            
class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf=min_samples_leaf
        self.loss=np.var
        self.type=RegressionTree621(min_samples_leaf=self.min_samples_leaf)


    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of observations in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        predictions=[]
        i=0
        for t in self.trees:
            p=t.predict(X_test)
            for pred in p:
                predictions.append(np.mean(pred)*len(pred))
                i+=len(pred)
        return np.array(predictions)/i
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        pred=self.predict(X_test)
        y_bar=y_test.mean()
        return np.sum((pred-y_bar)**2)/np.sum((y_test-y_bar)**2)
        
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, 
    max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.loss=gini
        self.type=ClassifierTree621(min_samples_leaf=self.min_samples_leaf)

    def predict(self, X_test) -> np.ndarray:
        empty=[]
        for x in X_test:
            empty.append([])
        for t in self.trees:
            p=t.predict(X_test)
            for x in range(len(p)):
                for x_i in p[x]:
                    empty[x].append(x_i)
        class_per_ob=[]
        for x in empty:
            class_counts=dict()        
            for x_i in x:
                if x_i in class_counts:
                    class_counts[x_i]+=1
                else:
                    class_counts[x_i]=1
            sort_=sorted(class_counts.items(), key=lambda x:x[1], reverse=True)
            class_per_ob.append(list(sort_)[0][0])
        return np.array(class_per_ob)
            
            

        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        self.y_test=y_test
        pred=self.predict(X_test)
        accuracy=1-((y_test-pred)**2).sum()/len(y_test)
        return accuracy