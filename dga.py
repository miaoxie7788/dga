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


def transform():
    pass


def train():
    pass


def predict():
    pass


def main():
    df = ingest(csv_params)
    print(df)


if __name__ == "__main__":
    main()
