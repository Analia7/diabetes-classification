# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo # fetch diabetes dataset
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def test_random_forest(n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, n_jobs, max_samples, class_weight, oob_score, ):
    return

def test_decision_tree(criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, max_leaf_nodes=2, min_impurity_decrease, class_weight, ccp_alpha):
    """
    NOTE
    ----
    max_leaf_nodes is set to 2 since the data used has 2 potential classifications: diabetic or not diabetic

    """
    return