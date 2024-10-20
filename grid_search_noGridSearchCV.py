import yaml
import csv
import numpy as np
from sklearn.model_selection import GridSearchCV
from model import test_decision_tree, custom_scorer
import itertools

# load .yml file containing hyperparameter values to be tested
def load_yml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
hyperparams = load_yml('hyperparameters.yml')

# create grid
hparam_grid = {hparam: values for hparam, values in hyperparams['hyperparameters'].items()}

# create the grid search function
def grid_search(X_train, y_train, X_test, y_test, hparam_combinations, output_file):
    with open(output_file, mode='w', newline=''):
        writer.writerow([
            'criterion', 'splitter', 'min_samples_split', 'min_samples_leaf', 
            'min_weight_fraction_leaf', 'max_features', 'min_impurity_decrease', 
            'class_weight', 'ccp_alpha', 'n_trees', 'accuracy', 'max_depth'
        ])
        
        # track best performance
        best_accuracy = 0
        min_maxdepth = 1000
        best_performance = 0
        current_performance = 0
        best_hparams = None
        # performance will be measured as combination of high accuracy and small max_depth

        # iterate through all hparam combinations
        for hparams in hparam_combinations:
            accuracy, average_max_depth = test_decision_tree(
                X_train, 
                y_train, 
                X_test, 
                y_test, 
                criterion=hparams['criterion'],
                splitter=hparams['splitter'],
                min_samples_split=hparams['min_samples_split'],
                min_samples_leaf=hparams['min_samples_leaf'],
                min_weight_fraction_leaf=hparams['min_weight_fraction_leaf'],
                max_features=hparams['max_features'],
                min_impurity_decrease=hparams['min_impurity_decrease'],
                class_weight=hparams['class_weight'],
                ccp_alpha=hparams['ccp_alpha'],
                n_trees=hparams['n_trees']
            )
            
            # write to .csv file
            writer.writerow([
                hparams['criterion'], hparams['splitter'], hparams['min_samples_split'], 
                hparams['min_samples_leaf'], hparams['min_weight_fraction_leaf'], 
                hparams['max_features'], hparams['min_impurity_decrease'], 
                hparams['class_weight'], hparams['ccp_alpha'], hparams['n_trees'], 
                accuracy, average_max_depth
            ])
            current_performance = custom_scorer(accuracy, average_max_depth)
            if current_performance > best_performance:
                best_performance = current_performance
                best_hparams = hparams

            if accuracy > best_accuracy:
                best_accuracy = accuracy

            if average_max_depth < min_maxdepth:
                min_maxdepth = average_max_depth

    print(f"Grid search completed. Results saved to {output_file}")
    return best_hparams, best_performance, best_accuracy, min_maxdepth
            


# OLD GRIDSEARCH
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
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
hparam_combinations = list(ParameterGrid(hparam_grid))

# create custom scorer
def custom_scorer_func(estimator, X, y, max_possible_depth=100):
    # Get accuracy and max depth from the estimator
    accuracy, max_depth = estimator.score(X, y)
    # Use the custom scorer function to combine accuracy and max depth
    return custom_scorer(accuracy, max_depth, max_possible_depth)

performance_scorer = make_scorer(custom_scorer_func, greater_is_better=True)

decision_tree_testing = CustomDecisionTreeModel()

grid_search = GridSearchCV(estimator=decision_tree_testing, param_grid=hparam_grid, scoring=performance_scorer, cv=1, refit=False)
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