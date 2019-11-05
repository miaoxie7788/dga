"""
    Keeps various parameters.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn.svm import SVC

# Parameters for data ingestion.
csv_params = {'filepath_or_buffer': "data/dga_domains.csv",
              'header': 0,
              'index_col': False,
              'dtype': {"host": str, ' '"domain": str, ' '"class": str, "subclass": str},
              'engine': 'c'}

# Algorithms for classification: Naive Bayes (NB), Logistic Regression (LR) and Support Vector Machine (SVM)
algo_dicts = {

    'nb': {
        'Gaussian':     GaussianNB(),
        'Multinomial':  MultinomialNB(),
        'Complement':   ComplementNB(),
        'Bernoulli':    BernoulliNB()
    },

    'lr': {
        'newton-cg':    LogisticRegression(penalty='l2', solver='newton-cg', max_iter=50, n_jobs=4, multi_class='ovr'),
        'lbfgs':      LogisticRegression(penalty='l2', solver='lbfgs', max_iter=50, n_jobs=4, multi_class='ovr'),
        'sag':        LogisticRegression(penalty='l2', solver='sag', max_iter=50, n_jobs=4, multi_class='ovr'),
        'saga':       LogisticRegression(penalty='l2', solver='saga', max_iter=50, n_jobs=4, multi_class='ovr')
    },

    'svm': {

        'linear': SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                          kernel='linear', max_iter=50, probability=False, random_state=None,
                          shrinking=True, tol=0.001, verbose=False),
        'poly': SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                        kernel='poly', max_iter=50, probability=False, random_state=None, shrinking=True,
                        tol=0.001, verbose=False),
        'sigmoid': SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                           kernel='sigmoid', max_iter=50, probability=False, random_state=None,
                           shrinking=True, tol=0.001, verbose=False),
        'rbf': SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                       kernel='rbf', max_iter=50, probability=False, random_state=None, shrinking=True,
                       tol=0.001, verbose=False),
    }

}
