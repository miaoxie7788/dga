"""
    Keeps various parameters.
"""

# Parameters for data ingestion.
csv_params = {'filepath_or_buffer': "data/dga_domains.csv",
              'header': 0,
              'index_col': False,
              'dtype': {"host": str, ' '"domain": str, ' '"class": str, "subclass": str},
              'engine': 'c'}
