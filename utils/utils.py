import random, sqlite3
import numpy as np
from opening_feat import load_features
import itertools as it
import operator
import pandas as pd
from sklearn import preprocessing


def evaluate_average(sum, count):
    try:
        Result = (sum / float(count))
    except ZeroDivisionError:
        return 0
    return Result


def read_user_general_baseline():

    conn = sqlite3.connect('content/database.db')

    _ratings = pd.read_sql("SELECT userid, SUM(rating)/COUNT(rating) AS average "
                           "FROM movielens_rating GROUP BY userid ORDER BY userid", conn, columns=['userID', 'average'])
    conn.close()

    return _ratings, _ratings['average'].mean()


def read_movie_general_baseline():

    conn = sqlite3.connect('content/database.db')

    _ratings_by_movie = pd.read_sql("SELECT movielensid, SUM(rating)/COUNT(rating) AS average "
                                    "FROM movielens_rating GROUP BY movielensid ORDER BY movielensid", conn,
                                    columns=['average'], index_col='movielensID')

    conn.close()

    return _ratings_by_movie


def select_random_users(conn, limit1, limit2):

    query_users = "SELECT u.userid, u.avgrating " \
                 "FROM movielens_user_trailer u " \
                  "ORDER BY u.userid "
                 # "LIMIT ?, ?"

    c = conn.cursor()
    # c.execute(query_users, (limit1, limit2,))
    c.execute(query_users)
    all_users = c.fetchall()
    return all_users


# Randomly select items unrated by the user
def get_random_movie_set(user):

    conn = sqlite3.connect('content/database.db')

    sql = "SELECT t.id, 1, mm.movielensId " \
          "FROM movies m " \
          "JOIN movielens_movie mm ON mm.imdbidtt = m.imdbid " \
          "JOIN trailers t ON t.imdbID = m.imdbID AND t.best_file = 1 " \
          "WHERE EXISTS (SELECT movielensid FROM movielens_rating r WHERE r.movielensid = mm.movielensid) " \
          "EXCEPT " \
          "SELECT t.id, 1, mm.movielensId " \
          "FROM movies m " \
          "JOIN movielens_movie mm ON mm.imdbidtt = m.imdbid " \
          "JOIN trailers t ON t.imdbID = m.imdbID AND t.best_file = 1 " \
          "JOIN movielens_rating r ON r.movielensid = mm.movielensid " \
          "WHERE r.userid = ? " \
          "ORDER BY t.id "

    limit = 200
    c = conn.cursor()
    c.execute(sql, (user,))
    all_movies = c.fetchall()
    if len(all_movies) == 0:
        return 0

    movies = random.sample(all_movies, limit)
    conn.close()

    return movies


def get_user_baseline(userid, _ratings, _global_average):

    return _ratings.iloc[userid]['average'] - _global_average


def get_item_baseline(user_baseline, movieid, _ratings_by_movie, _global_average):

    return _ratings_by_movie.loc[movieid]['average'] - user_baseline - _global_average


# Return a percentage of the user's rated movies
def get_user_training_test_movies(user):

    conn = sqlite3.connect('content/database.db')

    sql = "SELECT t.id, r.rating, m.movielensid " \
          "FROM trailers t " \
          "JOIN movielens_movie m ON m.imdbidtt = t.imdbid " \
          "JOIN movielens_rating r ON r.movielensid = m.movielensid " \
          "WHERE t.best_file = 1 " \
          "AND r.userid = ? " \
          "ORDER BY t.id "
    print sql

    c = conn.cursor()
    c.execute(sql, (user,))
    all_movies = c.fetchall()
    relevant_test_set = filter((lambda x: x[1] >= 4), all_movies)  # good movies for each user
    irrelevant_test_set = filter((lambda x: x[1] < 3), all_movies)  # bad movies for each user
    conn.close()

    all_movies_ids = [movielensid for trailerid, rating, movielensid in all_movies]

    return relevant_test_set, all_movies, irrelevant_test_set, all_movies_ids


def get_similarity_matrices():

    similarities_deep = load_features('movie_cosine_similarities_deep.bin')

    return similarities_deep


# Read the 128x1 feature vector from each trailer
# Normalize the matrix
def extract_features(deep_feautures='resnet_152_lstm_128.dct'):

    DEEP_FEATURES = load_features(deep_feautures)
    arr = np.array([x[1] for x in DEEP_FEATURES.iteritems()])
    scaler = preprocessing.StandardScaler().fit(arr)
    std = scaler.transform(arr)
    DEEP_FEATURES = {k: v for k, v in it.izip(DEEP_FEATURES.keys(), std)}

    return DEEP_FEATURES


def sort_desc(list_to_sort, index=1, desc=True):

    list_to_sort.sort(key=operator.itemgetter(index), reverse=desc)
    return list_to_sort
