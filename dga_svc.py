from datetime import datetime

from analytics.base import PandasAnalytics, PandasLog
from analytics.util.base_ml_util import verify_options
from analytics.util.udfs import transform_domain_len_udf, transform_domain_num_len_udf, \
    transform_domain_sym_len_udf, transform_domain_vow_len_udf, transform_domain_uniq_count_udf, \
    transform_domain_norm_ent_udf, transform_domain_gini_idx_udf, transform_domain_class_err_udf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class DGALogSVC(PandasLog):
    log_type = "csv"

    ingest_params = {
        'header': 0,
        'index_col': False,
        'dtype': {"host": str, "domain": str, "class": str, "subclass": str},
        'engine': 'c'}

    transform_funcs = {
        't_x1': (transform_domain_len_udf, 'domain'),
        't_x2': (transform_domain_num_len_udf, 'domain'),
        't_x3': (transform_domain_sym_len_udf, 'domain'),
        't_x4': (transform_domain_vow_len_udf, 'domain'),
        't_x5': (transform_domain_uniq_count_udf, 'domain'),
        't_x6': (transform_domain_norm_ent_udf, 'domain'),
        't_x7': (transform_domain_gini_idx_udf, 'domain'),
        't_x8': (transform_domain_class_err_udf, 'domain'),
    }


# Predict DGA using SVC (supervised).
class DGAPredictSVC(PandasAnalytics):
    estimator = SVC

    def __init__(self, pandas_log, options):
        super().__init__(dataset=pandas_log)

        self.options = options

        self.grid_search_cv = None
        self.pipeline = None

    def enrich(self):
        """
             - feature selection
             - model selection
        """

        # Feature selection.
        # The algo is tree classifier but other algos can be applied.
        fs = SelectFromModel(ExtraTreesClassifier())
        ss = StandardScaler()

        # Model selection.
        # Other classification algos such as LogisticRegression(), GaussianNB() and BernoulliNB() can be added into
        # the pipeline instead of SVC() to tune out the best algo.
        svc = self.estimator(max_iter=1000)

        param_grid = {
            'fs__threshold': ['mean', 'median'],
            'svc__C': [1.0, 0.1, 10],
            'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'svc__degree': [3],
            'svc__gamma': ['scale'],
        }

        steps = [('fs', fs), ('ss', ss), ('svc', svc)]
        pipeline = Pipeline(steps=steps)

        self.grid_search_cv = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1, cv=3)

    def publish(self):
        options = verify_options(self.df, self.options)

        X_columns = options['X_columns']
        y_column = options['y_column']
        old_columns = [old_column for _, (_, old_column) in self.log.transform_funcs.items()]
        new_columns = list(self.log.transform_funcs.keys())
        X_columns = list(set(X_columns).difference(set(old_columns)).union(new_columns))

        X = self.df[X_columns]
        y = self.df[y_column]

        columns = X_columns + [y_column]
        self.grid_search_cv = self.grid_search_cv.fit(X, y)
        self.df = self.df[columns]
        self.options['X_columns'] = X_columns

    def fit(self):
        X = self.df[self.options['X_columns']]
        y = self.df[self.options['y_column']]

        estimator = self.grid_search_cv.best_estimator_
        self.estimator = estimator.fit(X, y)

    def apply(self):
        if not self.estimator:
            raise Exception("The estimator has not set.")

        X = self.df[self.options['X_columns']]
        y = self.df[self.options['y_column']]

        y0 = self.estimator.predict(X)
        print("accuracy:", (y == y0).sum() / len(y))
        return y0


if __name__ == "__main__":
    start = datetime.now()

    dga_log = DGALogSVC(filename="data/dga_domains.csv")
    svc_options = {
        'name': 'MCM',
        'X_columns': ['domain'],
        'y_column': 'class'
    }

    dga_predict_svc = DGAPredictSVC(dga_log, svc_options)
    dga_predict_svc.main()

    end = datetime.now() - start
    print("time consumption: {t} seconds".format(t=end))
