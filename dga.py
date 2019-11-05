import pandas as pd
from params import csv_params


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



def transform():
    pass


def train():
    pass


def predict():
    pass


def main():
    df = ingest(csv_params)
    describe(df)


if __name__ == "__main__":
    main()
