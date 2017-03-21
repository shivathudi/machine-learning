import argparse
import re
import os
import csv
import math
import collections as coll
import numpy as np


def parse_argument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('--train', nargs=1, required=True)
    parser.add_argument('--test', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args


def parse_file(filename):
    """
    Given a filename outputs user_ratings and movie_ratings dictionaries

    Input: filename

    Output: user_ratings, movie_ratings
        where:
            user_ratings[user_id] = {movie_id: rating}
            movie_ratings[movie_id] = {user_id: rating}
    """
    user_ratings = {}
    movie_ratings = {}
    # Your code here
    with open(filename, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            movie_id = int(row[0])
            user_id = int(row[1])
            rating = float(row[2])
            if user_id not in user_ratings:
                user_ratings[user_id] = {}
                user_ratings[user_id][movie_id] = rating
            else:
                user_ratings[user_id][movie_id] = rating
            if movie_id not in movie_ratings:
                movie_ratings[movie_id] = {}
                movie_ratings[movie_id][user_id] = rating
            else:
                movie_ratings[movie_id][user_id] = rating

    return user_ratings, movie_ratings


def compute_average_user_ratings(user_ratings):
    """ Given the user_rating dict compute average user ratings

    Input: user_ratings (dictionary of user, movies, ratings)
    Output: ave_ratings (dictionary of user and ave_ratings)
    """
    ave_ratings = coll.defaultdict(float)
    # Your code here
    for each_user in user_ratings.keys():

        ratings_list = user_ratings[each_user].values()
        avg_rating = math.fsum(ratings_list)/math.floor(len(ratings_list))

        ave_ratings[each_user] = avg_rating

    return ave_ratings


def compute_user_similarity(d1, d2, ave_rat1, ave_rat2):
    """ Computes similarity between two users

        Input: d1, d2, (dictionary of user ratings per user) 
            ave_rat1, ave_rat2 average rating per user (float)
        Output: user similarity (float)
    """
    # Your code here
    intersection_movies = d1.viewkeys() & d2.viewkeys()

    if len(intersection_movies) == 0:
        return 0.0

    numerator = 0.0
    denominator_1 = 0.0
    denominator_2 = 0.0
    for each_movie in intersection_movies:
        numerator += (d1[each_movie] - ave_rat1)*(d2[each_movie] - ave_rat2)
        denominator_1 += math.pow((d1[each_movie] - ave_rat1), 2)
        denominator_2 += math.pow((d2[each_movie] - ave_rat2), 2)

    try:
        similarity_score = numerator/(math.sqrt(denominator_1*denominator_2))
        return similarity_score
    except:
        return 0.0

def main():
    """
    This function is called from the command line via
    
    python cf.py --train [path to filename] --test [path to filename]
    """
    args = parse_argument()
    train_file = args['train'][0]
    test_file = args['test'][0]
    print train_file, test_file
    # your code here

    user_ratings, movie_ratings = parse_file(train_file)
    ave_ratings = compute_average_user_ratings(user_ratings)

    actual_rating_list = []
    predicted_rating_list = []

    with open(test_file) as f1, open('predictions.txt', 'w') as f2:
        reader = csv.reader(f1)
        for row in reader:

            testing_movie_id = int(row[0])
            testing_user_id = int(row[1])
            actual_rating = float(row[2])

            numerator = 0.0
            denominator = 0.0

            try:
                ave_rat_i = ave_ratings[testing_user_id]
            except:
                ave_rat_i = 0.0

            for each_user in movie_ratings[testing_movie_id].keys():
                # print movie_ratings[testing_movie_id].keys()
                ave_rat_j = ave_ratings[each_user]
                sim_score = compute_user_similarity(user_ratings[each_user], user_ratings[testing_user_id],
                                                     ave_rat_j, ave_rat_i)
                numerator += sim_score * (movie_ratings[testing_movie_id][each_user] - ave_rat_j)
                denominator += math.fabs(sim_score)

            try:
                predicted_rating = ave_rat_i + (numerator/denominator)
            except:
                predicted_rating = ave_rat_i

            f2.write(','.join([str(each) for each in (row + [round(predicted_rating,1)])]) + '\n')

            actual_rating_list.append(actual_rating)
            predicted_rating_list.append(predicted_rating)


    RMSE = math.sqrt(np.mean((np.array(actual_rating_list) - np.array(predicted_rating_list))**2))
    MAE = np.mean(np.absolute(np.array(actual_rating_list) - np.array(predicted_rating_list)))

    print "RMSE", round(RMSE,4)
    print "MAE", round(MAE,4)



if __name__ == '__main__':
    main()

