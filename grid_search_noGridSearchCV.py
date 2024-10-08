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
            
