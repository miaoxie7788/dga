from itertools import combinations
from multiprocessing import Pool
from pickle import dump, load

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from markov import markov_fit, markov_apply
from params import csv_params, algo_dicts
from udfs import transform_domain_len_udf, transform_domain_num_len_udf, transform_domain_sym_len_udf, \
    transform_domain_vow_len_udf, transform_domain_uniq_count_udf, transform_domain_norm_ent_udf, \
    transform_domain_gini_idx_udf, transform_domain_class_err_udf


def ingest(params, csv_filename="data/dga_domains.csv"):
    """
        Ingest data (CSV) into a Pandas dataframe.
    :param params:          parameters for data ingestion.
    :param csv_filename     filename of csv to be ingested.
    :return:                raw dataframe.
    """
    params['filepath_or_buffer'] = csv_filename
    raw_df = pd.read_csv(**params)

    return raw_df


def describe(raw_df, target_column='class'):
    """
        Provide descriptive statistics for the raw dataframe.
    :param raw_df:      raw dataframe
    :param target_column:   target column, 'class' or 'subclass'
    """
    print("---------------------------- Descriptive statistics ----------------------------")
    print(" Total number of rows: {}".format(len(raw_df)))
    print(" Total number of columns: {}".format(len(raw_df.columns)))

    for column in raw_df.columns:
        if column == target_column:
            # Print distribution of class.
            print(" Column '{column}'s distribution: \n".format(column=column), raw_df[column].value_counts().to_dict())
        else:
            # Print number of unique values.
            print(" Column '{column}' has {n} unique values.".format(column=column, n=raw_df[column].nunique()))

    print("--------------------------------------------------------------------------------\n")


def transform(raw_df, target_column='class'):
    """
        Transform the raw dataframe and extract the features which are named as 't_x1', 't_x2', ....
    :param raw_df:          raw dataframe
    :param target_column:   target column, 'class' or 'subclass'
    :return:                transform dataframe, label encoder fitted to target column
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
    target_le = preprocessing.LabelEncoder()
    target_le.fit(transform_df[target_column])
    transform_df = transform_df.assign(t_y=target_le.transform(transform_df[target_column]))

    return transform_df, target_le


def cross_val_algo(train_columns, k_fold=3):
    for clf in algo_dict:
        score = cross_val_score(estimator=algo_dict[clf], X=train_df[train_columns], y=targets, cv=k_fold).mean()
        # print('columns': columns_to_test, 'clf': clf, 'score': score)
        return {'columns': train_columns, 'clf': clf, 'score': score}


def cross_val(transform_df, train_columns, target_column, settings_filename="data/settings.csv",
              best_filename="data/best.pickle"):
    """
        Seek the best setting (algorithm/classifier and feature set) via brute-forced cross-validation (k=3 by default).
    :param transform_df:                transform dataframe
    :param train_columns:               training columns
    :param target_column:               target column
    :param settings_filename:           filename of csv that keeps settings generated from cross validation
    :param best_filename:               filename of pickle that keeps the best setting
    :return:                            best setting (dict)
    """

    # TODO: It is not a good practice to use global variables. However, they are made global for running
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

        # Run cross_val_algo() in a multi-processing manner, where each process works with a set of training columns.
        pool = Pool(processes=12)
        scores = pool.map(cross_val_algo, train_columns_sets)
        pool.close()

        # Merge multiple setting dataframes while adding a new column algo into each dataframe.
        setting_dfs.append(pd.DataFrame(scores).assign(algo=algo))

    setting_df = pd.concat(setting_dfs, ignore_index=True)
    setting_df.to_csv(settings_filename, header=True, index=False)

    # Select the setting that yields highest 'score', as best.
    best = setting_df.iloc[setting_df['score'].idxmax()].to_dict()

    print("--------------------------------- Best setting ---------------------------------")
    print("""
        Algorithm: {algo} 
        Classifier: {clf} 
        Feature set: {columns} 
        Score: {score} 
          """.format(algo=best['algo'], clf=best['clf'], columns=best['columns'], score=best['score']))
    print("--------------------------------------------------------------------------------")

    with open(best_filename, 'wb') as f:
        dump(best, f)
    print("The best setting is dumped into {name}.".format(name=best_filename))

    return best


def pipeline_cross_val(csv_filename="data/dga_domains.csv", settings_filename="data/settings.csv",
                       best_filename="data/best.pickle"):
    # Ingest raw data.
    raw_df = ingest(csv_params, csv_filename)

    # Transform raw data.
    transform_df, _ = transform(raw_df)

    train_columns = [column for column in transform_df.columns if 't_x' in column]
    target_column = ['t_y']

    # TODO: It is not the best practice to select classifier/feature set using a brute-forced cross-validation.
    #  Instead, this can be done by measuring class/within-class variance, fisher information and/or PCA. Due to the
    #  time constraint, this may be fixed later.            MX 06/11/2019

    # Select best algorithm/classifier and feature set via brute-forced cross-validation.
    # The candidate algorithms are ["NB", "LR", "SVM"]. The candidate features are presented in transform().
    # In addition, each algorithm can construct multiple classifiers by using different methods or parameters.

    best = cross_val(transform_df, train_columns, target_column, settings_filename, best_filename)
    return best


def supervised_predict(x, clf, le):
    """
        Predict a feature vector using the given classifier and fitted label encoder.
    :param x:           feature vector extracted from a given domain
    :param clf:         classifier
    :param le:          label encoder
    :return:            label (inverse transformed by label encoder)
    """

    y = clf.predict([x])
    label = le.inverse_transform(y)[0]

    return label


def pipeline_supervised_train(csv_filename="data/dga_domains.csv", best_filename="data/best.pickle",
                              clf_filename="data/clf.pickle"):
    # Load best setting.
    try:
        with open(best_filename, 'rb') as f:
            best = load(f)
    except FileNotFoundError:
        print("{name} does not exist. Try action 'a'.".format(name=best_filename))
        return None

    # Ingest raw data.
    raw_df = ingest(csv_params, csv_filename)

    # Print descriptive statistics.
    describe(raw_df)

    # Transform raw data.
    transform_df, target_le = transform(raw_df)

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

    # Dumps the trained classifier and the fitted label encoder both.
    with open(clf_filename, 'wb') as f:
        dump([clf, target_le], f)

    print("The trained classifier is dumped into {name}.".format(name=clf_filename))

    return None


def pipeline_supervised_predict(best_filename="data/best.pickle", clf_filename="data/clf.pickle"):
    # Load best setting.
    try:
        with open(best_filename, 'rb') as f:
            best = load(f)
    except FileNotFoundError:
        print("{name} does not exist. Try action 'a'.".format(name=best_filename))
        return None

    # Load classifier.
    try:
        with open(clf_filename, 'rb') as f:
            clf, target_le = load(f)
    except FileNotFoundError:
        print("{name} does not exist. Try action 'c'.".format(name=clf_filename))
        return None

    transform_functions = {'t_x1': transform_domain_len_udf, 't_x2': transform_domain_num_len_udf,
                           't_x3': transform_domain_sym_len_udf, 't_x4': transform_domain_vow_len_udf,
                           't_x5': transform_domain_uniq_count_udf, 't_x6': transform_domain_norm_ent_udf,
                           't_x7': transform_domain_gini_idx_udf, 't_x8': transform_domain_class_err_udf, }

    while True:
        domain = input("Give a domain to be predicted? \n")
        if domain:
            x = [transform_functions[column](domain) for column in best['columns']]
            label = supervised_predict(x, clf, target_le)
        else:
            label = 'Try again.'
        print(label)


def pipeline_unsupervised_train(csv_filename="data/dga_domains.csv", markov_filename="data/markov.pickle"):
    # Ingest raw data.
    raw_df = ingest(csv_params, csv_filename)

    # Print descriptive statistics.
    describe(raw_df)

    # In case we don't have labels, we have to assume all domains are `legit'.
    # However we are interested to test how the unsupervised detector works with the labelled 'dga' domains.
    legit_domains = raw_df[raw_df['class'] == 'legit']['domain'].values
    dga_domains = raw_df[raw_df['class'] == 'dga']['domain'].values

    # Turn each domain (string) into a list of chars (iterables).
    legit_seqs = list(map(list, legit_domains))
    dga_seqs = list(map(list, dga_domains))

    # Train a markov model using only legit domains.
    # In practice, we are unable to obtain a set of legit domains (assuming no label). However, it is reasonable to
    # assume most of domains are legit. That is, in practice, the model will be contaminated with a very small number
    # of bad domains. According to our experience, this will not impact on the performance.
    markov_model = markov_fit(legit_seqs)

    # Compute the sequence log probability with averaged 3-grams.
    legit_seq_prs = [markov_apply(seq, markov_model, is_log=True) for seq in legit_seqs]
    dga_seq_prs = [markov_apply(seq, markov_model, is_log=True) for seq in dga_seqs]

    # Test a percentile x ranging from 0.5 - 5 with a step size of 0.5.
    # We use such a percentile as threshold (th), indicating that only x% legit domains will result in a sequence
    # probability larger than th.

    th90 = 0
    for k in range(1, 10):
        th = np.percentile(legit_seq_prs, k / 2)
        acc = sum(dga_seq_prs < th) / len(dga_seq_prs)

        # Select the first one that achieves a detection accuracy of higher than 90% as the threshold.
        if acc > 0.9 and th90 == 0:
            th90 = th

        # If last percentile is still not able to yield an accuracy of higher than 90%, use the last one as the
        # threshold anyway.
        if k == 10 and th90 == 0:
            th90 = th
        print("Test the threshold {th}: theoretical ACC={acc}, FPR={fpr}%".format(th=th, acc=acc, fpr=k / 2))

    print("The selected threshold is {th}".format(th=th90))

    # Dump the trained Markov model.
    with open(markov_filename, 'wb') as f:
        dump([markov_model, th90], f)

    print("The trained Markov model is dumped into {name}.".format(name=markov_filename))

    return None


def unsupervised_predict(domain, markov_model, th):
    if domain:
        seq_pr = markov_apply(list(domain), markov_model, is_log=True)

        if seq_pr < th:
            label = 'dga'
        else:
            label = 'legit'
    else:
        label = "Try again."

    return label


def pipeline_unsupervised_predict():
    with open("data/markov.pickle", 'rb') as f:
        markov_model, th = load(f)

    while True:
        domain = input("Give a domain to be predicted? \n")
        label = unsupervised_predict(domain, markov_model, th)
        print(label)


def main():
    action_code = input(""" 
    --------------------------------- DGA detector ---------------------------------
    Choose an action to take 
    'a': automatically select classifier and feature set by cross-validation.
    'c': train a classifier (supervised). 
    's': predict a domain with the classifier. 
    'm': train a Markov model (unsupervised).
    'u': predict a domain with the Markov model. 
    'e': exit. 
    --------------------------------------------------------------------------------    
        """)

    if action_code == 'e':
        print("bye.")
    elif action_code == 'a':
        pipeline_cross_val()
        main()
    elif action_code == 'c':
        pipeline_supervised_train()
        main()
    elif action_code == 's':
        pipeline_supervised_predict()
    elif action_code == 'm':
        pipeline_unsupervised_train()
        main()
    elif action_code == 'u':
        pipeline_unsupervised_predict()
    else:
        print("Choose 'a', 'c', 's', 'm', 'u', or 'e'.")
        main()


if __name__ == "__main__":
    main()
