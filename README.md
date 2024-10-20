# Diabetes classification prediction
Finding causal structure in the hyperparameters for training a digital twin that predicts whether a patient is diabetic or not, given a set of measurements.

## Data generated
Using the data provided by CDC Diabetes Indicators (https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset or https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) we will aim to train an _optimally chosen_ `sklearn` `DecisionTreeClassifier` model to predict whether a patient is diabetic or not so that the model is a) fast and b) accurate.<sup>1</sup>

The `DecisionTreeClassifier` model implementation used will be the one provided in `sklearn`.

We will aim to use causal discovery to recover the causal relationships between the input arguments a user may give to the `DecisionTreeClassifier` and how these affect the performance of the `DecisionTreeClassifier` (performance including accuracy and speed).

### Hyperparameter combinations
For this purpose we will individually train a `DecisionTreeClassifier` model for each combination of the potential hyperparameters that a user may input using grid search and collect the performance of training the corresponding model with the CDC Diabetes Indicator data.<sup>2,3</sup>

The input arguments we are testing for the `DecisionTreeClassifier` are the following:
| Argument                   | Values Allowed                               | Values Being Tested                                       |
|----------------------------|---------------------------------------------|----------------------------------------------------------|
| `criterion`                  | 'gini' or 'entropy'                        | 'gini' and 'entropy' (2 values)                         |
| `min_samples_split`          | int or % (float value 0.0-1.0)             | 10 values evenly spaced between 0.0 and 1.0              |
| `min_samples_leaf`           | int or % (float value 0.0-1.0)             | 10 values evenly spaced between 0.0 and 1.0              |
| `min_weight_fraction_leaf`      | int or % (float value 0.0-1.0)             | 10 values evenly spaced between 0.0 and 1.0              |
| `max_features`               | % (float value 0.0-1.0), 'auto', 'sqrt' or 'log2' | 10 values evenly spaced between 0.0 and 1.0, 'sqrt' and 'log2' (12 values) |
| `min_impurity_decrease`      | float                                       | 10 values                                                |
| `class_weight`               | None or 'balanced'                         | None and 'balanced' (2 values)                          |
| `splitter`                   | 'best' or 'random'                         | 'best' and 'random' (2 values)                          |


Which leaves us with a total of 960,000 combinations to try.

### Performance of each model
Each of this combinations will be used to create a `DecisionTreeClassifier` instance and this model will be trained on the CDC Diabetes Indicator dataset.<sup>2<sup>

The performance of the tree will be measured taking into account the values below:
- `accuracy`: The accuracy of the model.
- `precision`: Precision for the positive class.
- `recall`: Recall for the positive class.
- `f1_score`: Harmonic mean of precision and recall.
- `roc_auc`: Area Under the ROC Curve (only for binary classification).
- `avg_max_depth`: The average depth of the trees in the model, as a proxy of time taken and model complexity.<sup>4</sup>

So the dataset we will generate will contain a total of 14 columns.

### A note on randomness
Every time an instance of a `DecisionTreeClassifier` is created there is some inherent randomness in how it is created (unless a `random_state` is specified which will not be the case in this repository). Thus, in order to deal with this randomness so that the data produced is reproducible to some extent every combination will create and train a total of `n_trees` `DecisionTreeClassifier` models and the there will be a majority voting procedure to pick the mode prediction as the final prediction of our combination of parameters. The `n_trees` will be fixed to 5 in this repository, but the code allows for anyone interested to choose a different number of `DecisionTreeClassifier` instances to be created per hyperparameter combination.

### A note on the imbalance in the data
The data visualization process showed that the data is heavily skewed towards class 0 (not diabetic), as seen below, and hence it is important to deal with this issue.
![Data distribution visualization](class_distribution.png)

The first attempt at running `grid_search.py` showed that the `class_weights` parameter of the `DecisionTree` doesn't work very well. In particular, setting this value to `None` favors the majority class to much (giving us an accuracy of around 86% and setting the value to `balanced` favors the minority class too much (giving us an accuracy of 14%). Thus, `SMOTE` will be use to create synthetic samples of the minority class when resampling data before feeding this data to the `DecisionTree`.

In light of this discovery, new performance measures will be added to obtain more information on the correctness of the model.

# Footnotes
1. By optimally chosen we mean the following. Causal discovery will attempt to recover the causal relationships between the input arguments a user may give to the `DecisionTreeClassifier` and how these affect the performance of the `DecisionTreeClassifier` (performance including accuracy and speed). Given this, we will be able to choose a narrower subset of hyperparameter values to try to find the optimal combination of hyperparameters that contributes to peak performance.
2. Note that we will actually train several `DecisionTreeClassifier` models per combination to account for the randomness that follows the creating of a `DecisionTreeClassifier` instance. More on this on the next few paragraphs.
3. This data collection will not be done over the whole set of the hyperparameter space as this would be computational prohibitive, so a subset of it will be chosen to get a broad picture of how the values of each argument affect the outcome.
4. See *A note on randomness* for a discussion of why an average max depth is used.
