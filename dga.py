from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from params import csv_params, algo_dicts
from udfs import transform_domain_len_udf, transform_domain_num_len_udf, transform_domain_sym_len_udf, \
    transform_domain_vow_len_udf, transform_domain_uniq_count_udf, transform_domain_norm_ent_udf, \
    transform_domain_gini_idx_udf, transform_domain_class_err_udf

from pickle import dump, load


def ingest(params):
    """
        Ingest data (CSV) into a Pandas dataframe.
    :param params:      Parameters for data ingestion.
    :return:            raw dataframe.
    """
    df = pd.read_csv(**params)

    return df


def describe(df):
    """
        Provide descriptive statistics for the given dataframe.
    :param df:          raw dataframe
    """
    print("---------------------------- Descriptive statistics ----------------------------")
    print(" Total number of rows: {}".format(len(df)))
    print(" Total number of columns: {}".format(len(df.columns)))

    for column in df.columns:
        if column not in ['class', 'subclass']:
            print(" Column '{column}' has {n} unique values.".format(column=column, n=df[column].nunique()))
        else:
            print(" Column '{column}'s distribution: \n".format(column=column), df[column].value_counts().to_dict())
    print("--------------------------------------------------------------------------------\n")


def transform(df):
    """
        Transform the raw dataframe, extracting features.
    :param df:          raw dataframe
    :return:            transform dataframe
    """
    # TODO: Due to the time constraint, we primarily extract the features from a string perspective. This may need
    #  more research in dealing with a practical application.                           MX 06/11/2019

    df = df.assign(t_x1=df['domain'].apply(transform_domain_len_udf))
    df = df.assign(t_x2=df['domain'].apply(transform_domain_num_len_udf))
    df = df.assign(t_x3=df['domain'].apply(transform_domain_sym_len_udf))
    df = df.assign(t_x4=df['domain'].apply(transform_domain_vow_len_udf))
    df = df.assign(t_x5=df['domain'].apply(transform_domain_uniq_count_udf))
    df = df.assign(t_x6=df['domain'].apply(transform_domain_norm_ent_udf))
    df = df.assign(t_x7=df['domain'].apply(transform_domain_gini_idx_udf))
    df = df.assign(t_x8=df['domain'].apply(transform_domain_class_err_udf))

    # Transform target column 'class' into numeric using label encoder.
    le = preprocessing.LabelEncoder()
    le.fit(df['class'])
    df = df.assign(t_y=le.transform(df['class']))

    return df, le


def cross_val_algo(columns_to_test, k_fold=3):
    for clf in algo_dict:
        score = cross_val_score(estimator=algo_dict[clf], X=X[columns_to_test], y=y, cv=k_fold).mean()
        # print('columns': columns_to_test, 'clf': clf, 'score': score)
        return {'columns': columns_to_test, 'clf': clf, 'score': score}


def cross_val(df, columns_X, column_y, filename=None):
    """
        See the best setting (algorithm/classifier and feature set) via brute-forced cross-validation (k=3 by default).
    :param df:          transform dataframe
    :param columns_X:   columns of candidate feature set
    :param column_y:    column of target
    :param filename:    output CSV that records the details of the cross-validation.
    :return:            best setting
    """

    # TODO: It is not good practice to use global variables. However, they are made global for running
    #  multi-processing. This may be fixed later.           MX 06/11/2019
    global X, y, algo_dict
    X, y = df[columns_X], df[column_y]

    # Test the subsets of the X columns with size ranging from 2 to n.
    n = len(columns_X)
    columns_X = np.array(columns_X)
    columns_tests = [columns_X[c] for c in [list(c) for k in range(2, n + 1) for c in combinations(list(range(n)), k)]]

    score_dfs = list()
    for algo in algo_dicts:
        algo_dict = algo_dicts[algo]

        pool = Pool(processes=12)
        scores = pool.map(cross_val_algo, columns_tests)
        pool.close()

        score_dfs.append(pd.DataFrame(scores).assign(algo=algo))

    score_df = pd.concat(score_dfs, ignore_index=True)
    best = score_df.iloc[score_df['score'].idxmax()]

    print("--------------------------------- Best setting ---------------------------------")
    print(" Algorithm: ", best['algo'])
    print(" Classifier: ", best['clf'])
    print(" Feature set: ", best['columns'])
    print(" Score: ", best['score'])
    print("--------------------------------------------------------------------------------")

    if filename:
        score_df.to_csv(filename, header=True, index=False)
    return best


def train(df, columns_X, column_y, clf, filename=None):

    X, y = df[columns_X], df[column_y]
    clf.fit(X, y)

    if filename:
        with open(filename, 'wb') as f:
            dump(clf, f)

    return clf


def predict():
    pass


def main():
    # Ingest raw data.
    raw_df = ingest(csv_params)

    # Print descriptive statistics.
    describe(raw_df)

    # Transform raw data.
    transform_df, le = transform(raw_df)

    # Select best algorithm and classifier via brute-forced cross-validation.
    # The algorithms can be "NB", "LR" or "SVM". Each algorithm can construct different classifiers by using
    # different methods or parameters.

    # TODO: It is not the best practice to select classifier/feature set using a brute-forced cross-validation.
    #  Instead, this can be done by measuring class/within-class variance, fisher information and/or PCA. Due to the
    #  time constraint, this may be fixed later.            MX 06/11/2019
    columns_X, column_y = ['t_x1', 't_x2', 't_x3', 't_x4', 't_x5', 't_x6', 't_x7', 't_x8'], ['t_y']

    # best = cross_val(transform_df, columns_X, column_y, filename="data/scores.csv")

    best = {'algo': 'lr',
            'clf': 'newton-cg',
            'columns': ['t_x1', 't_x2', 't_x3', 't_x4', 't_x5', 't_x6', 't_x7'],
            'score': 0.9122425817242358}

    columns_X = best['columns']
    clf = algo_dicts[best['algo']][best['clf']]

    train(transform_df, columns_X, column_y, clf, filename="data/clf.pickle")

if __name__ == "__main__":
    main()
