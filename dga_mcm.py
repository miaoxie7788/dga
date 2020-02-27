from datetime import datetime

from analytics.base import PandasAnalytics, PandasLog
from analytics.ml.customised import mcm_fit, mcm_apply
from analytics.util.base_ml_util import verify_options
from numpy import percentile


class DGALogMCM(PandasLog):
    log_type = "csv"

    ingest_params = {
        'header': 0,
        'index_col': False,
        'dtype': {"host": str, "domain": str, "class": str, "subclass": str},
        'engine': 'c'}

    transform_funcs = None


# Predict DGA using Markov chain model (MCM, unsupervised).
class DGAPredictMCM(PandasAnalytics):
    """
        Predict a DGA host using unsupervised learning (Markov).
    """
    estimator = None

    def __init__(self, pandas_log, options):
        super().__init__(dataset=pandas_log)

        self.options = options

    def enrich(self):
        ...

    def publish(self):
        options = verify_options(self.df, self.options)
        X_columns = options['X_columns']
        y_column = options['y_column']

        columns = X_columns + [y_column]

        self.df = self.df[columns]
        self.options = options

    def fit(self):

        X0 = list(map(lambda x: list(x[0]),
                      self.df[self.df[self.options['y_column']] == 'legit'][self.options['X_columns']].values))

        markov = mcm_fit(X0)
        self.estimator = markov

    def apply(self):
        if not self.estimator:
            raise Exception("The estimator has not set.")

        markov = self.estimator
        X = list(map(lambda x: list(x[0]), self.df[self.options['X_columns']].values))
        X0 = list(map(lambda x: list(x[0]),
                      self.df[self.df[self.options['y_column']] == 'legit'][self.options['X_columns']].values))
        y = self.df[self.options['y_column']]

        # Learn the threshold.
        ps = [mcm_apply(x0, markov, is_log=True) for x0 in X0]
        th = percentile(ps, 1)

        ps = [mcm_apply(x, markov, is_log=True) for x in X]

        y0 = list()
        for p in ps:
            if p < th:
                y0.append('dga')
            else:
                y0.append('legit')

        print("accuracy:", (y == y0).sum() / len(y))


if __name__ == "__main__":
    start = datetime.now()

    dga_log = DGALogMCM(filename="data/dga_domains.csv")
    mcm_options = {
        'name': 'MCM',
        'X_columns': ['domain'],
        'y_column': 'class'
    }

    dga_predict_mcm = DGAPredictMCM(dga_log, mcm_options)
    dga_predict_mcm.main()

    end = datetime.now() - start
    print("time consumption: {t} seconds".format(t=end))
