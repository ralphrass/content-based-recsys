# Simplicity is the final achievement. After one has played a vast quantity of notes and more notes, it is
# simplicity that emerges as the crowning reward of art. Frederic Chopin.

import operator
import sqlite3
import numpy as np
import time
from utils.utils import get_item_baseline, sort_desc, read_user_general_baseline, read_movie_general_baseline
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process, Manager

_avg_ratings = 3.51611876907599
_std_deviation = 1.098732183


def cosine(movieI, movieJ, feature_vector):

    traileri = movieI[0]
    trailerj = movieJ[0]

    try:
        featuresI = feature_vector[traileri]
        featuresJ = feature_vector[trailerj]
    except KeyError:
        return 0

    return cosine_similarity([featuresI], [featuresJ])


def predict_user_rating(user_baseline, movieid, all_similarities, _ratings_by_movie, _global_average):

    global _avg_ratings

    item_baseline = get_item_baseline(user_baseline, movieid, _ratings_by_movie, _global_average)
    user_item_baseline = (_avg_ratings + user_baseline + item_baseline)

    numerator = sum((rating - user_item_baseline) * sim if sim > 0 else 0 for rating, sim in all_similarities)
    denominator = reduce(operator.add, [abs(x[1]) for x in all_similarities])
    try:
        prediction = (numerator / denominator) + user_item_baseline
    except ZeroDivisionError:
        prediction = 0

    return prediction


def get_content_based_predictions(user_baseline, movies, all_movies, sim_matrix, _ratings_by_movie, _global_average):

    predictions = [(movie[2], predict_user_rating(user_baseline, movie[2],
                                                  [(movieJ[1], sim_matrix[movieJ[0]][movie[0]])
                                                   for movieJ in all_movies], _ratings_by_movie, _global_average),
                    movie[0])
                   for movie in movies]

    return sort_desc(predictions)


def get_user_item_baseline(item_ratings, user_baseline, _global_average):

    # item_baseline = sum([_rating - user_baseline - _global_average
    #                      for _rating in item_ratings['rating']]) / len(item_ratings)
    item_baseline = np.array([_rating - user_baseline - _global_average for _rating in item_ratings['rating']]).mean()
    user_item_baseline = _global_average + user_baseline + item_baseline

    return user_item_baseline


def get_item_top_similarities(_all_ratings_by_user, target_movielens_id, _all_ratings, item_ratings):

    _limit_neighbourhood_to = 30
    similarities = []

    for key, movie in _all_ratings_by_user[_all_ratings_by_user['movielensID'] != target_movielens_id].iterrows():

        join_ratings = pd.merge(_all_ratings[_all_ratings['movielensID'] == movie['movielensID']],
                                item_ratings, on='userID')

        try:
            ratings_x, ratings_y = (np.array(join_ratings['rating_x']), np.array(join_ratings['rating_y']))
            sim = cosine_similarity(ratings_x.reshape(1, -1), ratings_y.reshape(1, -1))[0][0]
            similarities.append((movie['movielensID'], sim))
        except ValueError:
            continue

    return sort_desc(similarities)[:_limit_neighbourhood_to]


def predict_user_item_rating(_all_ratings_by_user, user_item_baseline, top_neighbours):

    try:
        p_ui_numerator = sum([s_ij * (_all_ratings_by_user[_all_ratings_by_user['movielensID'] == movie_id]
                                      ['rating'].iloc[0] - user_item_baseline)
                              for movie_id, s_ij in top_neighbours])
        p_ui_denominator = sum([abs(s_ij) for n, s_ij in top_neighbours])
        p_ui = p_ui_numerator / p_ui_denominator + user_item_baseline
    except ZeroDivisionError:
        return 0

    return p_ui


def get_item_collaborative_predictions(target_user_id, user_baseline, movies_to_predict, _all_ratings, _global_average):

    predictions = []
    _all_ratings_by_user = _all_ratings[_all_ratings['userID'] == target_user_id]

    for trailer_id, rating, target_movielens_id in movies_to_predict:

        item_ratings = _all_ratings[_all_ratings['movielensID'] == target_movielens_id]

        user_item_baseline = get_user_item_baseline(item_ratings, user_baseline, _global_average)
        top_neighbours = get_item_top_similarities(_all_ratings_by_user, target_movielens_id, _all_ratings, item_ratings)

        p_ui = predict_user_item_rating(_all_ratings_by_user, user_item_baseline, top_neighbours)

        predictions.append((target_movielens_id, p_ui, trailer_id))
        # break
    return predictions


def calculate_user_rating_predictions(_start, _end, _user_profiles, new_user_profiles, _convnet_similarity_matrix,
                                      _all_ratings):

    _general_baseline, _global_average = read_user_general_baseline()
    _ratings_by_movie = read_movie_general_baseline()

    for userid, profile in _user_profiles.iloc[_start:_end].iterrows():
        print userid, "userid"
        movies_set = profile['relevant_set'] + profile['irrelevant_set'] + profile['random_set']

        if type(movies_set) == float:
            continue

        predictions_item_collaborative = get_item_collaborative_predictions(userid, profile['user_baseline'].iloc[0],
                                                                            movies_set, _all_ratings, _global_average)
        predictions_content_based = get_content_based_predictions(profile['user_baseline'].iloc[0], movies_set, profile['all_movies'],
                                                                  _convnet_similarity_matrix, _ratings_by_movie, _global_average)

        new_user_profiles[userid] = {'datasets': {'relevant_movies': profile['relevant_set'],
                                                  'irrelevant_movies': profile['irrelevant_set']},
                                     # 'userid': userid,
                                     'predictions': {'deep': predictions_content_based,
                                                     'collaborative': predictions_item_collaborative}
                                     }


def read_all_ratings():

    start = time.time()
    conn = sqlite3.connect('content/database.db')
    _all_ratings = pd.read_sql('select userID, movielensID, rating from movielens_rating order by userid', conn)
    conn.close()
    print "all ratings read in", time.time() - start, "seconds"
    return _all_ratings


# Predict ratings
def build_user_profile(_user_profiles, _convnet_similarity_matrix):

    # _general_baseline, _global_average = read_user_general_baseline()
    # _ratings_by_movie = read_movie_general_baseline()
    _all_ratings = read_all_ratings()

    manager = Manager()
    new_user_profiles = manager.dict()
    # new_user_profiles = {}
    jobs = []
    _max = 4  # multiples of 4
    _process_at_time = 4
    _step = _max/_process_at_time

    for idx in range(0, _max, _step):
        p = Process(target=calculate_user_rating_predictions, args=(idx, idx + _step, _user_profiles, new_user_profiles,
                                                                    _convnet_similarity_matrix, _all_ratings))
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    # for userid, profile in user_profiles.iloc[:1].iterrows():
    #     print userid, "userid"
    #     movies_set = profile['relevant_set'] + profile['irrelevant_set'] + profile['random_set']
    #
    #     if type(movies_set) == float:
    #         continue
    #
    #     predictions_item_collaborative = get_item_collaborative_predictions(userid, profile['user_baseline'].iloc[0],
    #                                                                         movies_set, _all_ratings, _global_average)
    #     predictions_content_based = get_content_based_predictions(profile['user_baseline'].iloc[0], movies_set, profile['all_movies'],
    #                                                               convnet_similarity_matrix, _ratings_by_movie, _global_average)
    #
    #     new_user_profiles[userid] = {'datasets': {'relevant_movies': profile['relevant_set'],
    #                                               'irrelevant_movies': profile['irrelevant_set']},
    #                                  'userid': userid,
    #                                  'predictions': {'deep': predictions_content_based,
    #                                                  'collaborative': predictions_item_collaborative}
    #                                  }

    return dict(new_user_profiles)
