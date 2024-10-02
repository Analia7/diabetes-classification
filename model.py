# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.stats import mode
from ucimlrepo import fetch_ucirepo # fetch diabetes dataset
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def test_random_forest(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, n_jobs, max_samples, class_weight, oob_score):
    return

def test_decision_tree(X_train, y_train, X_test, y_test, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, min_impurity_decrease, class_weight, ccp_alpha, n_trees, max_leaf_nodes=2):
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
    predictions = np.zeros(n_trees, dtype=int)

    # create and fit the tree models and use them to predict the values of y for X_test
    for i in range(n_trees):
        decision_trees[i] = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_split=min_samples_split,
                                                      min_samples_leaf=min_samples_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                                                      min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, ccp_alpha=ccp_alpha)
        decision_trees[i].fit(X_train, y_train)
        predictions[i] = decision_trees[i].predict(X_test)

    # apply majority voting (column-wise) to get the final predicted labels
    y_pred, _ = mode(predictions, axis=0)
    

    #decision_trees = np.array([ for _ in range(n_trees)], dtype=object)
    
    return