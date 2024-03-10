# Random Forest Implementation

## Overview

RandomForest621 is a Python implementation of a Random Forest for both regression and classification tasks. It is built on top of decision tree models, providing an ensemble learning approach for improved performance and robustness.

## Features

- **RandomForest621 Class:**
  - Parameters:
    - `n_estimators`: Number of trees in the forest.
    - `oob_score`: Whether to calculate the out-of-bag score.
  - Attributes:
    - `n_estimators`: Number of trees in the forest.
    - `oob_score`: Whether to calculate the out-of-bag score.
    - `oob_score_`: Out-of-bag score estimate.
    - `trees`: List to store individual decision trees.

- **RandomForestRegressor621 Class:**
  - Inherits from RandomForest621.
  - Additional Parameters:
    - `min_samples_leaf`: Minimum number of samples in a leaf.
    - `max_features`: Maximum fraction of features to consider for a split.
  - Methods:
    - `predict(X_test)`: Compute the weighted average prediction for regression.
    - `score(X_test, y_test)`: Compute R^2 for regression.

- **RandomForestClassifier621 Class:**
  - Inherits from RandomForest621.
  - Additional Parameters:
    - `min_samples_leaf`: Minimum number of samples in a leaf.
    - `max_features`: Maximum fraction of features to consider for a split.
  - Methods:
    - `predict(X_test)`: Compute the predicted class for each record.
    - `score(X_test, y_test)`: Compute accuracy for classification.

## Usage

1. Import the required libraries and Decision Tree classes.
   ```python
   import numpy as np
   from sklearn.utils import resample
   from dtree import DecisionTree621, RegressionTree621, ClassifierTree621, gini
2. Create an instance of RandomForestRegressor621 or RandomForestClassifier621, depending on the task.
```
     # Example for regression
    rf_regressor = RandomForestRegressor621(n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=True)
    rf_regressor.fit(X_train, y_train)
    predictions = rf_regressor.predict(X_test)
    r2_score = rf_regressor.score(X_test, y_test)
    
    # Example for classification
    rf_classifier = RandomForestClassifier621(n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=True)
    rf_classifier.fit(X_train, y_train)
    predictions = rf_classifier.predict(X_test)
    accuracy = rf_classifier.score(X_test, y_test)
```
3. Access additional information:
```
oob_score_estimate = rf_regressor.oob_score_
```
### Dependencies:
- numpy
- scikit-learn
