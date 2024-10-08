# Importing necessary libraries
import numpy as np
from sklearn.ensemble import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode  
from sklearn.base import BaseEstimator, ClassifierMixin

class CustomDecisionTreeModel(BaseEstimator, ClassifierMixin):
    """
    Create a decision tree digital twin model to determine whether a patient is diabetic or not.
    Output:
        (accuracy, timetaken) - performance metrics of the decision tree given the inputed hyperparams
    NOTE
    ----
    max_leaf_nodes is set to 2 since the data used has 2 potential classifications: diabetic or not diabetic
    """
    def __init__(self, criterion='gini', splitter='best', min_samples_split=2, min_samples_leaf=1, 
                 min_weight_fraction_leaf=0.0, max_features=None, min_impurity_decrease=0.0, 
                 class_weight=None, ccp_alpha=0.0, n_trees=5, max_leaf_nodes=2, max_possible_depth=1000, alpha=0.7, beta=0.3):
        self.criterion = criterion
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.n_trees = n_trees
        self.max_leaf_nodes = max_leaf_nodes
        self.max_possible_depth = max_possible_depth
        self.alpha = alpha
        self.beta = beta

    def fit(self, X, y):
        self.decision_trees_ = []
        self.max_depths_ = []
        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(
                criterion=self.criterion, splitter=self.splitter, min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf, min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_features=self.max_features, min_impurity_decrease=self.min_impurity_decrease,
                class_weight=self.class_weight, ccp_alpha=self.ccp_alpha, max_leaf_nodes=self.max_leaf_nodes
            )
            tree.fit(X, y)
            self.decision_trees_.append(tree)
            self.max_depths_.append(tree.get_depth())
        return self

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.decision_trees_])
        y_pred, _ = mode(predictions, axis=0)
        return y_pred.flatten()

    def score(self, X, y):
        # Get accuracy
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        # Get average max depth
        avg_max_depth = np.mean(self.max_depths_)
        
        return accuracy, avg_max_depth

# in case I need the custom scorer
def custom_scorer(accuracy, max_depth, max_possible_depth=1000, alpha=0.7, beta=0.3):
    """
    Custom scoring function that combines accuracy and max_depth.
    
    Parameters:
    - accuracy: The accuracy of the model
    - max_depth: The max depth of the decision tree
    - max_possible_depth: The maximum possible depth of a tree (for scaling)
    - alpha: Weight for accuracy
    - beta: Weight for max depth penalty
    
    Returns:
    - score: Combined score based on accuracy and max depth
    """
    normalized_max_depth = max_depth / max_possible_depth
    score = alpha * accuracy - beta * normalized_max_depth
    return score
