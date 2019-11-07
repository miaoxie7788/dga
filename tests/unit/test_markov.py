import unittest

from markov import markov_fit, markov_apply

class TestMarkov(unittest.TestCase):

    def test_markov_fit_empty(self):
        # tests markov_fit with data=[[]]
        data = [[]]
        output = markov_fit(data)

        self.assertEqual(list(output['s']), [])
        self.assertEqual(list(output['p0']), [])
        self.assertEqual(list(output['p1']), [])

    def test_markov_fit_single_str(self):
        # tests markov_fit with data=[['a']]
        data = [['a']]
        output = markov_fit(data)

        self.assertEqual(list(output['s']), ['a'])
        self.assertEqual(list(output['p0']), [1])
        self.assertEqual(list(output['p1']), [0])

    def test_markov_fit_single_num(self):
        # tests markov_fit with data=[[1]]
        data = [[1]]
        output = markov_fit(data)

        self.assertEqual(list(output['s']), ['1'])
        self.assertEqual(list(output['p0']), [1])
        self.assertEqual(list(output['p1']), [0])

    def test_markov_apply_empty(self):
        # tests markov_fit with data=[]

        data = []
        markov_model = {'s': ['a', 'b'], 'p0': [0.5, 0.5], 'p1': [[0, 1], [0, 0]]}
        output = markov_apply(data, markov_model)
        self.assertAlmostEqual(output, 0)

    def test_markov_apply_new_state(self):
        # tests markov_fit with data containing new states

        data = ['c', 'a', 'b']
        markov_model = {'s': ['a', 'b'], 'p0': [0.5, 0.5], 'p1': [[0, 1], [0, 0]]}
        output = markov_apply(data, markov_model)
        self.assertAlmostEqual(output, 0)

    def test_markov_apply_log_zero(self):
        # tests markov_fit with data resulting in log(0)

        data = ['b', 'a']
        markov_model = {'s': ['a', 'b'], 'p0': [0.5, 0.5], 'p1': [[0, 1], [0, 0]]}
        output = markov_apply(data, markov_model, is_log=True)
        self.assertAlmostEqual(output, -230.26)


if __name__ == '__main__':
    unittest.main()
