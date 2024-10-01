import os
import argparse

import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import FunctionTransformer

from utils import load_train, load_test, proc_series, label
from algos.castle import PC
from node_classifier_model.node_classifier import NodeClassifier

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data', type=str)
    parser.add_argument('--tr_it', type=int, default=-1) # -1 means use all
    parser.add_argument('--te_it', type=int, default=-1)
    parser.add_argument('-o', type=str, dest='output_path', default='./')
    parser.add_argument('--n_jobs', type=int, default=1)

    parser.add_argument('--alg', dest='algorithm', type=str, choices=['pc', 'nc'])
    parser.add_argument('--thr', dest='threshold', type=float, default=0.5)
    parser.add_argument('--pt', dest='pair_test', type=str, choices=['anm', 'bivariate_fit', 'cds', 'gnn', 'igci', 'reci'])

    return parser

def _predict(X, y, ft, alg):
    true = ft.transform(label(y)['label'])
    G = alg.learn(X)
    pred = ft.transform(label(G)['label'])

    return balanced_accuracy_score(true, pred)

# Must follow the common interface of
# fun(X, thr)
# and return an adjacency matrix
def get_alg(opt):
    if opt.algorithm == 'pc':
        pc = PC(opt.threshold)
        return pc
    elif opt.algorithm == 'nc':
        nc = NodeClassifier(opt.pair_test, opt.threshold)
        return nc
    else:
        raise ValueError(f"Unrecognised algorithm type. Passed '{opt.algorithm}'.")

def test_loop(data, n_iter, opt):
    x_train, y_train = data
    ft = FunctionTransformer(proc_series)
    alg = get_alg(opt)

    parallel = Parallel(opt.n_jobs)
    if n_iter > 0:
        scores = parallel(
            delayed(_predict)(
                x_train[d_id], y_train[d_id], ft, alg
            )
            for d_id in list(x_train)[:n_iter]
        )
    else:
        scores = parallel(
            delayed(_predict)(
                x_train[d_id], y_train[d_id], ft, alg
            )
            for d_id in x_train
        )

    return np.mean(scores)

if __name__ == "__main__":
    parser = get_parser()
    options = parser.parse_args()

    # Check if output folder exists and create if necessary.
    if not os.path.isdir(options.output_path):
        os.mkdir(options.output_path)

    data_tr = load_train(options.data)
    acc_tr = test_loop(data_tr, options.tr_it, options)

    data_te = load_test(options.data)
    acc_te = test_loop(data_te, options.te_it, options)

    df = pd.DataFrame([[options.algorithm, options.threshold, acc_tr, acc_te]], columns=['algorithm', 'threshold', 'acc_train', 'acc_test'])

    filepath = os.path.join(options.output_path, 'results.csv')
    if os.path.exists(filepath):
        df_existing = pd.read_csv(filepath)
        df = pd.concat([df_existing, df], axis=0)

    df.to_csv(filepath, index=False)