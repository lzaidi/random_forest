import numpy as np
from scipy import stats
from sklearn.metrics import r2_score, accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] < self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Creates leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction
        return self.prediction


def gini(y):
    """
    Return the gini impurity score for values in y
    Assume y = {0,1}
    Gini = 1 - sum_i p_i^2 where p_i is the proportion of class i in y
    """
    total = 0
    set_y = set(y)
    for p in set_y:
        prop = ((y == p).sum() / len(y)) ** 2
        total += prop
    return 1 - total

    
def find_best_split(X, y, loss, min_samples_leaf):
    best_feature = -1
    best_split =- 1
    k = 11
    best_loss = loss(y)
    for i in range(1,len(X.T)):
        candidates = (np.random.choice(X[:, i], size=k))
        for split in candidates:
            yl = y[X[:, i]<split]
            yr = y[X[:, i]>=split]
            if len(yl) < min_samples_leaf or len(yr) < min_samples_leaf:
                continue
            l = ((len(yl) * loss(yl)) + (len(yr) * loss(yr))) / (len(yl) + len(yr))
            if l == 0:
                return i, split 
            if l < best_loss:
                best_loss = l
                best_feature = i
                best_split = split
    return best_feature, best_split 

class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss 
        
    def fit(self, X, y):
        """
        Creates a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for  either a classifier or regression.
        """
        self.root = self.fit_(X, y)


    def fit_(self, X, y):
        """
        Recursively creates and returns a decision tree fit to (X,y) for
        either a classification or regression. 
        """
        if len(X) <= self.min_samples_leaf or len(np.unique(X))==1:
            return self.create_leaf(y)
        col, split = find_best_split(X, y, self.loss, self.min_samples_leaf)
        if col == -1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:, col]<split], y[X[:, col]<split])
        rchild = self.fit_(X[X[:, col]>=split], y[X[:, col]>=split])
        return DecisionNode(col, split, lchild, rchild)
    

    def predict(self, X_test):
        """
        Makes a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621
        """
        tree = self.root
        predictions = [tree.predict(x) for x in X_test]
        return predictions



class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.var)

    def score(self, X_test, y_test):
        "Returns the R^2 of y_test vs predictions for each record in X_test"
        pred = self.predict(X_test)
        return r2_score(y_test, pred)


    def create_leaf(self, y):
        """
        Returns a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))

class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)

    def score(self, X_test, y_test):
        "Returns the accuracy_score() of y_test vs predictions for each record in X_test"
        pred = self.predict(X_test)
        return accuracy_score(y_test, pred)

    def create_leaf(self, y):
        """
        Returns a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor. 
        """
        return LeafNode(y, stats.mode(y)[0])
