import pandas as pd
from params import csv_params
from udfs import transform_domain_len_udf, transform_domain_num_len_udf, \
    transform_domain_sym_len_udf, transform_domain_vow_len_udf, transform_domain_uniq_count_udf, \
    transform_domain_norm_ent_udf, transform_domain_gini_inx_udf, transform_domain_class_err_udf


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
    :param df:          Panda dataframe
    """

    print("Total number of rows:{}".format(len(df)))
    print("Total number of columns:{}".format(len(df.columns)))

    print("\n")

    for column in df.columns:
        print("{column} has {n} unique values.".format(column=column, n=df[column].nunique()))


def transform(df):

    df = df.assign(t_f1=df['domain'].apply(transform_domain_len_udf))
    df = df.assign(t_f2=df['domain'].apply(transform_domain_num_len_udf))
    df = df.assign(t_f3=df['domain'].apply(transform_domain_sym_len_udf))
    df = df.assign(t_f4=df['domain'].apply(transform_domain_vow_len_udf))
    df = df.assign(t_f5=df['domain'].apply(transform_domain_uniq_count_udf))
    df = df.assign(t_f6=df['domain'].apply(transform_domain_norm_ent_udf))
    df = df.assign(t_f7=df['domain'].apply(transform_domain_gini_inx_udf))
    df = df.assign(t_f8=df['domain'].apply(transform_domain_class_err_udf))

    return df


def train():
    pass


def predict():
    pass


def main():
    df = ingest(csv_params)
    describe(df)

    df = transform(df)

    print(df)



if __name__ == "__main__":
    main()
