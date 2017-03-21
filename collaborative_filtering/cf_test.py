import cf
import unittest

# here is documentation for unittest
# https://docs.python.org/2/library/unittest.html

class TestCFMethods(unittest.TestCase):

    def setUp(self):
        self.user_ratings = {
                1: {1: 4.0, 3: 4.0, 4: 5.0, 5: 1.0},
                2: {1: 5.0, 2: 5.0, 3: 4.0, 7: 1.0},
                3: {4: 2.0, 5: 4.0, 6: 5.0, 7: 3.0},
                4: {2: 3.0}}

        self.movie_ratings = {
                1: {1: 4.0, 2: 5.0},
                2: {2: 5.0, 4: 3.0},
                3: {1: 4.0, 2: 4.0},
                4: {1: 5.0, 3: 2.0},
                5: {1: 1.0, 3: 4.0},
                6: {3: 5.0},
                7: {2: 1.0, 3: 3.0}}

        self.ave_rating = {1: 3.5, 2: 3.75, 3: 3.5, 4: 3.0} 

    def test_parse_file(self):
	user_ratings, movie_ratings = cf.parse_file("tinyTraining.txt")
        self.assertEqual(user_ratings[2][1], self.user_ratings[2][1])
        self.assertEqual(movie_ratings[7][3], self.movie_ratings[7][3])

    def test_compute_average_user_ratings(self):
        ave_ratings = cf.compute_average_user_ratings(self.user_ratings)
        self.assertEqual(ave_ratings[4], 3.0)
        self.assertEqual(ave_ratings[1], 3.5)
        print(ave_ratings)

    def test_compute_user_similarity(self):
        sim = cf.compute_user_similarity(
                self.user_ratings[4], self.user_ratings[1], 3.0, 3.5)
        self.assertEqual(sim, 0.0)
        sim = cf.compute_user_similarity(
                self.user_ratings[1], self.user_ratings[3], 3.5, 3.5)
        self.assertAlmostEqual(sim, -0.759256602365, 5)

if __name__ == '__main__':
    unittest.main()
