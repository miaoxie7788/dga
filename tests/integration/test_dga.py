import unittest

from dga import ingest, describe, pipeline_cross_val, pipeline_unsupervised_train, pipeline_supervised_train
from params import csv_params


class TestDGA(unittest.TestCase):

    def setUp(self):
        # Ingest test data
        self.raw_df = ingest(csv_params, csv_filename="fixtures/dga_domains_sample.csv")

    # Test function describe() and whether the raw dataframe is properly ingested.
    def test_describe(self):
        describe(self.raw_df)
        self.assertEqual(len(self.raw_df), 100)

    # Test the pipelines.
    def test_pipeline_cross_val(self):
        output = pipeline_cross_val(csv_filename="fixtures/dga_domains_sample.csv",
                                    settings_filename="fixtures/settings.csv", best_filename="fixtures/best.pickle")
        self.assertIsInstance(output, dict)

    def test_pipeline_supervised_train(self):
        output = pipeline_supervised_train(csv_filename="fixtures/dga_domains_sample.csv",
                                           best_filename="fixtures/best.pickle", clf_filename="fixtures/clf.pickle")
        self.assertIsNone(output)

    def test_pipeline_unsupervised_train(self):
        output = pipeline_unsupervised_train(csv_filename="fixtures/dga_domains_sample.csv",
                                             markov_filename="fixtures/markov.pickle")
        self.assertIsNone(output)


if __name__ == '__main__':
    unittest.main()
