# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import mode  

def test_decision_tree(X_train, y_train, X_test, y_test, criterion, splitter, min_samples_split, min_samples_leaf, max_features, min_impurity_decrease, class_weight, ccp_alpha, n_trees, max_leaf_nodes=2):
    """
    Create a decision tree digital twin model to determine whether a patient is diabetic or not.
    Output:
        (accuracy, timetaken) - performance metrics of the decision tree given the inputed hyperparams
    NOTE
    ----
    max_leaf_nodes is set to 2 since the data used has 2 potential classifications: diabetic or not diabetic
    """
    # create the decision tree model
    # n_trees decision trees will be created to deal with the randomness and the average performance will be selected
    decision_trees = np.empty(n_trees, dtype=object)
    predictions = np.empty(n_trees)
    max_depths = np.zeros(n_trees, dtype=int)

    # create and fit the tree models and use them to predict the values of y for X_test
    for i in range(n_trees):
        decision_trees[i] = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                                      min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha)
        decision_trees[i].fit(X_train, y_train)
        max_depths[i] = decision_trees[i].get_depth()
        predictions[i] = decision_trees[i].predict(X_test)

    # apply majority voting (column-wise) to get the final predicted labels
    y_pred, _ = mode(predictions, axis=0)
    # flatten the 2D array returned by mode into a prediction 1D array
    y_pred = y_pred.flatten()
    
    # accuracy score
    accuracy = accuracy_score(y_test, y_pred)

    # average max_depth as a proxy for complexity/time taken
    average_max_depth = np.mean(max_depths)
    
    return (accuracy, average_max_depth)


def test_random_forest(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, n_jobs, max_samples, class_weight, oob_score):
    return
