import numpy as np
from sklearn.utils import resample
from dtree import DecisionTree621, RegressionTree621, ClassifierTree621, gini

class RandomForest621:
    """
    Random Forest implementation for both regression and classification tasks.

    Parameters:
    - n_estimators: Number of trees in the forest.
    - oob_score: Whether to calculate the out-of-bag score.

    Attributes:
    - n_estimators: Number of trees in the forest.
    - oob_score: Whether to calculate the out-of-bag score.
    - oob_score_: Out-of-bag score estimate.
    - trees: List to store individual decision trees.
    """

    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.trees = []

    def fit(self, X, y):
        """
        Fit all n_estimators trees to bootstrapped versions of the training data.
        Keep track of the indexes of the OOB records for each tree.
        Compute the OOB validation score estimate if self.oob_score is True.
        """
        self.y = y
        self.X = X
        oob_errors = []
        scores = []
        in_indices = []
        for _ in range(self.n_estimators):
            scores.append([])
        for _ in range(self.n_estimators):
            index = resample(range(len(X)))
            in_indices.append(index)
            oob_index = np.array(list(set(range(len(X))) - set(index)))
            boot_x = X[index]
            boot_y = y[index]
            tree = DecisionTree621(min_samples_leaf=1)  # Assuming DecisionTree621 class exists
            tree.fit(boot_x, boot_y)
            self.trees.append(tree)
            oob_errors.append(oob_index)
        if self.oob_score:
            for x in range(len(oob_errors)):
                new_trees = [self.trees[i] for i in range(len(in_indices)) if
                             np.isin(in_indices[i], oob_errors[x], invert=True).all()]
                for t in new_trees:
                    oob_score = self.score(X[oob_errors[x]], y[oob_errors[x]])
                    scores[x].append(oob_score)
            if self.__class__.__name__ == 'RandomForestRegressor621':
                self.oob_score_ = [np.mean(scores) / 150]
            else:
                self.oob_score_ = [np.mean(scores)]


class RandomForestRegressor621(RandomForest621):
    """
    Random Forest implementation for regression tasks.

    Parameters:
    - n_estimators: Number of trees in the forest.
    - min_samples_leaf: Minimum number of samples in a leaf.
    - max_features: Maximum fraction of features to consider for a split.
    - oob_score: Whether to calculate the out-of-bag score.
    """

    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.loss = np.var
        self.type = RegressionTree621(min_samples_leaf=self.min_samples_leaf)

    def predict(self, X_test) -> np.ndarray:
        """
        Compute the weighted average prediction from all trees in this forest for regression.
        Weight each tree's prediction by the number of observations in the leaf making that prediction.
        Return a 1D vector with the predictions for each input record of X_test.
        """
        predictions = []
        i = 0

        for t in self.trees:
            p = t.predict(X_test)
            for pred in p:
                predictions.append(np.mean(pred) * len(pred))
                i += len(pred)

        return np.array(predictions) / i

    def score(self, X_test, y_test) -> float:
        """
        Collect predictions for each record and compute R^2 on that and y_test for regression.
        """
        pred = self.predict(X_test)
        y_bar = y_test.mean()
        return np.sum((pred - y_bar) ** 2) / np.sum((y_test - y_bar) ** 2)


class RandomForestClassifier621(RandomForest621):
    """
    Random Forest implementation for classification tasks.

    Parameters:
    - n_estimators: Number of trees in the forest.
    - min_samples_leaf: Minimum number of samples in a leaf.
    - max_features: Maximum fraction of features to consider for a split.
    - oob_score: Whether to calculate the out-of-bag score.
    """

    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.loss = gini
        self.type = ClassifierTree621(min_samples_leaf=self.min_samples_leaf)

    def predict(self, X_test) -> np.ndarray:
        """
        Compute the predicted class for each record and return the predictions for classification.
        """
        empty = [[] for _ in X_test]
        for t in self.trees:
            p = t.predict(X_test)
            for x in range(len(p)):
                for x_i in p[x]:
                    empty[x].append(x_i)
        class_per_ob = []
        for x in empty:
            class_counts = dict()
            for x_i in x:
                if x_i in class_counts:
                    class_counts[x_i] += 1
                else:
                    class_counts[x_i] = 1

            sort_ = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
            class_per_ob.append(list(sort_)[0][0])
        return np.array(class_per_ob)

    def score(self, X_test, y_test) -> float:
        """
        Collect the predicted class for each record and compute accuracy between that and y_test for classification.
        """
        self.y_test = y_test
        pred = self.predict(X_test)
        accuracy = 1 - ((y_test - pred) ** 2).sum() / len(y_test)
        return accuracy
