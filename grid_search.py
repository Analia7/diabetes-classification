import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer
from model import CustomDecisionTreeModel, custom_scorer
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
print("X", X)
print("y", y)

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# load .yml file containing hyperparameter values to be tested
def load_yml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
hyperparams = load_yml('hyperparameters.yml')

# create grid
hparam_grid = {hparam: values for hparam, values in hyperparams['hyperparameters'].items()}

# create custom scorer
def custom_scorer_func(estimator, X, y, max_possible_depth=100):
    # Get accuracy and max depth from the estimator
    accuracy, max_depth = estimator.score(X, y)
    # Use the custom scorer function to combine accuracy and max depth
    return custom_scorer(accuracy, max_depth, max_possible_depth)

performance_scorer = make_scorer(custom_scorer_func, greater_is_better=True)

decision_tree_testing = CustomDecisionTreeModel()

grid_search = GridSearchCV(estimator=decision_tree_testing, param_grid=hparam_grid, scoring=performance_scorer, cv=5)
grid_search.fit(X_train, y_train)

# export to .csv file

results_df = pd.DataFrame(grid_search.cv_results_)

# select relevant columns
results_df = results_df[['param_criterion', 'param_splitter', 'param_min_samples_split',
                         'param_min_samples_leaf', 'param_min_weight_fraction_leaf', 
                         'param_max_features', 'param_min_impurity_decrease',
                         'param_class_weight', 'param_ccp_alpha', 'mean_test_score', 'rank_test_score']]

# Assuming accuracy and max_depth were saved in the score() method of CustomDecisionTreeModel
results_df['mean_accuracy'] = [estimator.score(X_test, y_test)[0] for estimator in grid_search.cv_results_['estimator']]
results_df['mean_max_depth'] = [estimator.score(X_test, y_test)[1] for estimator in grid_search.cv_results_['estimator']]


results_df.to_csv('grid_search_results.csv', index=False)