import adaboost as ada
import unittest
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier

# here is documentation for unittest
# https://docs.python.org/2/library/unittest.html

class TestADAMethods(unittest.TestCase):

    def setUp(self):
        self.y = np.array([1., 0., 1., 1., 0., 0., 1., 1., 1., 0.])
        self.y2 = np.array([1., -1., 1., 1., -1., -1., 1., 1., 1., -1.])
         
    def test_parse_spambase_data(self):
	X, Y = ada.parse_spambase_data("tiny.spam.train")
        for i in range(len(self.y)):
            self.assertEqual(self.y[i], Y[i])
        n, m = X.shape
        self.assertEqual(n, 10)
        self.assertEqual(m, 57)

    def test_new_label(self):
        Y2 = ada.new_label(self.y)
        for i in range(len(self.y2)):
            self.assertEqual(self.y2[i], Y2[i])

    def test_adaboost(self):
        X, Y = ada.parse_spambase_data("tiny.spam.train")
        Y2 = ada.new_label(self.y)
        trees, weights = ada.adaboost(X, Y2, 2)
        self.assertEqual(len(trees), 2)
        self.assertEqual(len(weights), 2)
        self.assertTrue(isinstance(trees[0], sklearn.tree.tree.DecisionTreeClassifier))
        x = np.array([[0, -1], [1, 0], [-1, 0]])
        y = np.array([-1, 1, 1])
        trees, weights = ada.adaboost(x, y, 1)
        h = trees[0]
        pred = h.predict(x)
        for i in range(len(y)):
            self.assertEqual(pred[i], y[i])

    def test_adaboost_predict(self):
        x = np.array([[0, -1], [1, 0], [-1, 0]])
        y = np.array([-1, 1, 1])
        trees, weights = ada.adaboost(x, y, 1)
        pred = ada.adaboost_predict(x, trees, weights)
        for i in range(len(y)):
            self.assertEqual(pred[i], y[i])

if __name__ == '__main__':
    unittest.main()
