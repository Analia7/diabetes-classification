import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
from sklearn.metrics import make_scorer
from model import CustomDecisionTreeModel, custom_scorer
from joblib import Parallel, delayed
from ucimlrepo import fetch_ucirepo 

def load_yml(file_path):
    """
    Load hyperparameters from a YAML file.
    """
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def evaluate_model(hparams, X_train, y_train, X_test, y_test):
    """
    Initialize and evaluate the CustomDecisionTreeModel with given hyperparameters.
    
    Parameters:
    - params: The hyperparameters to be used for model initialization.
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    
    Returns:
    - A dictionary containing the hyperparameters, accuracy, and max depth.
    """
    # Initialize the model with the current set of hyperparameters
    decision_tree_testing = CustomDecisionTreeModel(
        criterion=hparams['criterion'],
        splitter=hparams['splitter'],
        min_samples_split=hparams['min_samples_split'],
        min_samples_leaf=hparams['min_samples_leaf'],
        min_weight_fraction_leaf=hparams['min_weight_fraction_leaf'],
        max_features=hparams['max_features'],
        min_impurity_decrease=hparams['min_impurity_decrease'],
        class_weight=hparams['class_weight'],
        ccp_alpha=hparams['ccp_alpha'],
        n_trees=hparams['n_trees'],
        max_leaf_nodes=hparams['max_leaf_nodes']
    )
    
    # Fit the model
    decision_tree_testing.fit(X_train, y_train)
    
    # Evaluate the model on the test set
    accuracy, max_depth = decision_tree_testing.score(X_test, y_test)
    
    # Return the results including hyperparameters, accuracy, and max depth
    return {
        'criterion': hparams['criterion'],
        'splitter': hparams['splitter'],
        'min_samples_split': hparams['min_samples_split'],
        'min_samples_leaf': hparams['min_samples_leaf'],
        'min_weight_fraction_leaf': hparams['min_weight_fraction_leaf'],
        'max_features': hparams['max_features'],
        'min_impurity_decrease': hparams['min_impurity_decrease'],
        'class_weight': hparams['class_weight'],
        'ccp_alpha': hparams['ccp_alpha'],
        'n_trees': hparams['n_trees'],
        'max_leaf_nodes': hparams['max_leaf_nodes'],
        'accuracy': accuracy,
        'max_depth': max_depth
    }

def manual_grid_search(X_train, y_train, X_test, y_test, param_grid, output_csv, n_jobs=-1):
    """
    Perform parallelized grid search with a custom decision tree model.
    
    Parameters:
    - X_train, y_train: Training data and labels.
    - X_test, y_test: Testing data and labels.
    - param_grid: Dictionary of hyperparameter values (use ParameterGrid to iterate).
    - output_csv: The file path where to save the CSV file with the results.
    - n_jobs: Number of CPU cores to use (-1 means use all available cores).
    
    Outputs:
    - Writes the grid search results into the CSV file specified by `output_csv`.
    """
    # Perform parallelized evaluation of all hyperparameter combinations
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_model)(params, X_train, y_train, X_test, y_test) 
                                      for params in ParameterGrid(param_grid))
    
    # Convert the results to a DataFrame and write to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    return

def run_grid_search():
    """
    Run the grid search and write results to a CSV file.
    """
    # Load dataset
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891)
    X = cdc_diabetes_health_indicators.data.features
    y = cdc_diabetes_health_indicators.data.targets
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Load hyperparameters from YAML file
    hyperparams = load_yml('hyperparameters.yml')
    
    # Create a hyperparameter grid
    hparam_grid = {hparam: values for hparam, values in hyperparams['hyperparameters'].items()}
    
    # Run the manual grid search
    output_csv = 'grid_search_results.csv'
    manual_grid_search(X_train, y_train, X_test, y_test, hparam_grid, output_csv, n_jobs=-1)
    return

if __name__ == "__main__":
    run_grid_search()
