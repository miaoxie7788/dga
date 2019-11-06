from itertools import combinations
from multiprocessing import Pool
from pickle import dump, load

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from params import csv_params, algo_dicts
from udfs import transform_domain_len_udf, transform_domain_num_len_udf, transform_domain_sym_len_udf, \
    transform_domain_vow_len_udf, transform_domain_uniq_count_udf, transform_domain_norm_ent_udf, \
    transform_domain_gini_idx_udf, transform_domain_class_err_udf


def ingest(params):
    """
        Ingest data (CSV) into a Pandas dataframe.
    :param params:      parameters for data ingestion.
    :return:            raw dataframe.
    """
    raw_df = pd.read_csv(**params)

    return raw_df


def describe(raw_df):
    """
        Provide descriptive statistics for the given dataframe.
    :param raw_df:      raw dataframe
    """
    print("---------------------------- Descriptive statistics ----------------------------")
    print(" Total number of rows: {}".format(len(raw_df)))
    print(" Total number of columns: {}".format(len(raw_df.columns)))

    for column in raw_df.columns:
        if column not in ['class', 'subclass']:
            print(" Column '{column}' has {n} unique values.".format(column=column, n=raw_df[column].nunique()))
        else:
            print(" Column '{column}'s distribution: \n".format(column=column), raw_df[column].value_counts().to_dict())
    print("--------------------------------------------------------------------------------\n")


def transform(raw_df, target_column='class'):
    """
        Transform the raw dataframe, extracting features. The extracted features will be named as 't_x1', 't_x2', ....
    :param raw_df:          raw dataframe
    :param target_column:   target column, 'class' or 'subclass'
    :return:                transform dataframe
    """

    # TODO: Due to the time constraint, we primarily extract the features from a string perspective. This may need
    #  more research while dealing with a practical application.                           MX 06/11/2019

    transform_df = raw_df
    transform_df = transform_df.assign(t_x1=transform_df['domain'].apply(transform_domain_len_udf))
    transform_df = transform_df.assign(t_x2=transform_df['domain'].apply(transform_domain_num_len_udf))
    transform_df = transform_df.assign(t_x3=transform_df['domain'].apply(transform_domain_sym_len_udf))
    transform_df = transform_df.assign(t_x4=transform_df['domain'].apply(transform_domain_vow_len_udf))
    transform_df = transform_df.assign(t_x5=transform_df['domain'].apply(transform_domain_uniq_count_udf))
    transform_df = transform_df.assign(t_x6=transform_df['domain'].apply(transform_domain_norm_ent_udf))
    transform_df = transform_df.assign(t_x7=transform_df['domain'].apply(transform_domain_gini_idx_udf))
    transform_df = transform_df.assign(t_x8=transform_df['domain'].apply(transform_domain_class_err_udf))

    # Transform target column 'class'/'subclass' into numeric using label encoder.
    le = preprocessing.LabelEncoder()
    le.fit(transform_df[target_column])
    transform_df = transform_df.assign(t_y=le.transform(transform_df[target_column]))

    return transform_df, le


def cross_val_algo(train_columns, k_fold=3):
    for clf in algo_dict:
        score = cross_val_score(estimator=algo_dict[clf], X=train_df[train_columns], y=targets, cv=k_fold).mean()
        # print('columns': columns_to_test, 'clf': clf, 'score': score)
        return {'columns': train_columns, 'clf': clf, 'score': score}


def cross_val(transform_df, train_columns, target_column):
    """
        Seek the best setting (algorithm/classifier and feature set) via brute-forced cross-validation (k=3 by default).
    :param transform_df:                transform dataframe
    :param train_columns:               training columns
    :param target_column:               target column
    :return:                            best setting
    """

    # TODO: It is not good practice to use global variables. However, they are made global for running
    #  multi-processing. This may be fixed later.           MX 06/11/2019
    global train_df, targets, algo_dict
    train_df, targets = transform_df[train_columns], transform_df[target_column]

    # Generate the subsets of the training columns with size ranging from 2 to n.
    n = len(train_columns)
    train_columns = np.array(train_columns)
    train_columns_sets = [train_columns[c] for c in
                          [list(c) for k in range(2, n + 1) for c in combinations(list(range(n)), k)]]

    setting_dfs = list()
    for algo in algo_dicts:
        algo_dict = algo_dicts[algo]

        # Run cross_val_algo in a multi-processing manner, where each process works with a set of training columns.
        pool = Pool(processes=12)
        scores = pool.map(cross_val_algo, train_columns_sets)
        pool.close()

        # Merge multiple setting dataframes while adding a new column algo into each dataframe.
        setting_dfs.append(pd.DataFrame(scores).assign(algo=algo))

    setting_df = pd.concat(setting_dfs, ignore_index=True)
    setting_df.to_csv("data/settings.csv", header=True, index=False)

    # Select the setting that comes with highest 'score', as best.
    best = setting_df.iloc[setting_df['score'].idxmax()].to_dict()

    print("--------------------------------- Best setting ---------------------------------")
    print("""
        Algorithm: {algo} 
        Classifier: {clf} 
        Feature set: {columns} 
        Score: {score} 
          """.format(algo=best['algo'], clf=best['clf'], columns=best['columns'], score=best['score']))
    print("--------------------------------------------------------------------------------")

    with open('data/best.pickle', 'wb') as f:
        dump(best, f)
    print("The best setting is dumped into data/best.pickle.")

    return best


def pipeline_cross_val():
    # Ingest raw data.
    raw_df = ingest(csv_params)

    # Transform raw data.
    transform_df, le = transform(raw_df)

    # Select best algorithm and classifier via brute-forced cross-validation.
    # The algorithms can be "NB", "LR" or "SVM". Each algorithm can construct different classifiers by using
    # different methods or parameters.

    # TODO: It is not the best practice to select classifier/feature set using a brute-forced cross-validation.
    #  Instead, this can be done by measuring class/within-class variance, fisher information and/or PCA. Due to the
    #  time constraint, this may be fixed later.            MX 06/11/2019

    train_columns = [column for column in transform_df.columns if 't_x' in column]
    target_column = ['t_y']

    cross_val(transform_df, train_columns, target_column)


def pipeline_supervised_train():
    # Ingest raw data.
    raw_df = ingest(csv_params)

    # Print descriptive statistics.
    describe(raw_df)

    # Transform raw data.
    transform_df, le = transform(raw_df)

    # Load best setting.
    with open("data/best.pickle", 'rb') as f:
        best = load(f)

    print("--------------------------------- Best setting ---------------------------------")
    print("""
        Algorithm: {algo} 
        Classifier: {clf} 
        Feature set: {columns} 
        Score: {score} 
          """.format(algo=best['algo'], clf=best['clf'], columns=best['columns'], score=best['score']))
    print("--------------------------------------------------------------------------------")

    train_columns = best['columns']
    target_column = ['t_y']

    X, y = transform_df[train_columns], transform_df[target_column]
    clf = algo_dicts[best['algo']][best['clf']]

    clf.fit(X, y)

    # Dumps the trained classifier and label encoder both.
    with open("data/clf.pickle", 'wb') as f:
        dump([clf, le], f)

    print("The trained classifier is dumped into data/clf.pickle.")


def pipeline_supervised_predict():
    with open("data/clf.pickle", 'rb') as f:
        clf, le = load(f)

    while True:
        domain = input("Give a domain to be predicted? \n")

        # TODO: The feature extraction below can be automated with loading best.pickle later. MX 06/11/2019

        if domain:
            x = [transform_domain_len_udf(domain), transform_domain_num_len_udf(domain),
                 transform_domain_sym_len_udf(domain), transform_domain_vow_len_udf(domain),
                 transform_domain_uniq_count_udf(domain), transform_domain_norm_ent_udf(domain),
                 transform_domain_gini_idx_udf(domain), ]

            y = clf.predict([x])
            print(le.inverse_transform(y)[0])


def main():
    action_code = input(""" 
    --------------------------------- DGA detector ---------------------------------
    Choose an action to take 
    'a': automatically select classifier and feature set by cross-validation.
    't': train a classifier. 
    'p': predict a domain with an existing classifier.  
    'e': exit. 
    --------------------------------------------------------------------------------    
        """)

    if action_code == 'a':
        pipeline_cross_val()
        main()
    elif action_code == 't':
        pipeline_supervised_train()
        main()
    elif action_code == 'p':
        pipeline_supervised_predict()
    elif action_code == 'e':
        print("bye.")
    else:
        print("Choose 'a', 't', 'p', or 'e'.")
        main()


if __name__ == "__main__":
    main()
