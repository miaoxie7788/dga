from itertools import combinations
from multiprocessing import Pool

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score

from params import csv_params, algos
from udfs import transform_domain_len_udf, transform_domain_num_len_udf, transform_domain_sym_len_udf, \
    transform_domain_vow_len_udf, transform_domain_uniq_count_udf, transform_domain_norm_ent_udf, \
    transform_domain_gini_idx_udf, transform_domain_class_err_udf


def cross_val_by_columns(columns_to_test, k_fold=3):
    for model in models:
        score = cross_val_score(estimator=models[model], X=X[columns_to_test], y=y, cv=k_fold).mean()
        # print('columns:', columns_to_test, 'model:', model, 'score:', score)
        return {'columns': columns_to_test, 'model': model, 'score': score}


def cross_val(filename=None):
    """

    """

    columns = X.columns
    n = len(columns)

    # Test the subsets of training columns with size ranging from 2 to n.
    columns_to_tests = [columns[c].to_list() for c in [list(c) for k in range(2, n + 1)
                                                       for c in combinations(list(range(n)), k)]]

    pool = Pool(processes=12)
    scores = pool.map(cross_val_by_columns, columns_to_tests)
    pool.close()

    score_df = pd.DataFrame(scores)

    best = score_df.iloc[score_df['score'].idxmax()]
    print("Best setting: \n", best)

    if filename:
        pd.DataFrame(scores).to_csv(filename, header=True, index=False)

    return best


def ingest(params):
    """
        Ingest data (CSV) into a Pandas dataframe.
    :param params:      Parameters for data ingestion.
    :return:            Pandas dataframe.
    """
    df = pd.read_csv(**params)

    return df


def describe(df):
    """
        Provide descriptive statistics for the given df.
    :param df:          Pandas dataframe
    """
    print("---------------------------- Descriptive statistics ----------------------------")
    print(" Total number of rows: {}".format(len(df)))
    print(" Total number of columns: {}".format(len(df.columns)))

    for column in df.columns:
        if column not in ['class', 'subclass']:
            print(" Column '{column}' has {n} unique values.".format(column=column, n=df['class'].value_counts().to_dict()))
        else:
            print(" Column '{column}'s distribution: \n".format(column=column), df[column].value_counts().to_dict())
    print("--------------------------------------------------------------------------------")

def transform(df):

    # Feature extraction.
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


def train():
    pass


def predict():
    pass


def main():
    # Ingest raw data.
    raw_df = ingest(csv_params)

    # Print descriptive statistics.
    describe(raw_df)

    # Transform raw data.
    # transform_df, _ = transform(raw_df)
    #
    # # These variables are made global for multi-processing.
    # global X, y, models
    #
    # X = transform_df[['t_x1', 't_x2', 't_x3', 't_x4', 't_x5', 't_x6', 't_x7', 't_x8']]
    # y = transform_df['t_y']
    #
    # for algo in algos:
    #     models = algos[algo]
    #     cross_val(filename="data/{algo}_scores.csv".format(algo=algo))

    # print(transform_df)



if __name__ == "__main__":
    main()
