# Simplicity is the final achievement. After one has played a vast quantity of notes and more notes, it is
# simplicity that emerges as the crowning reward of art. Frederic Chopin.

import operator
import sqlite3
import numpy as np
from utils.utils import get_item_baseline, sort_desc, read_user_general_baseline, read_movie_general_baseline
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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


# Each movie to predict contains: trailer id [0], user rating [1] and movielens id [2]
def get_user_collaborative_predictions(userid, user_average, movies_to_predict, all_ratings):

    predictions = []
    current_user_ratings = all_ratings.loc[userid]

    for movie in movies_to_predict:

        # limit to 5% neighbours that also rated the current movie
        neighbours = all_ratings[all_ratings['movielensID'] == movie[2]].sample(frac=0.01).index
        numerator, denominator = 0, 0

        for neighbour_id in neighbours:

            if neighbour_id == userid:
                continue

            neighbour_movies = all_ratings.loc[neighbour_id]
            # intersection = pd.merge(neighbour_movies, current_user_ratings, on='movielensID')
            ratings_user_x = current_user_ratings[current_user_ratings['movielensID'].isin(
                neighbour_movies['movielensID'].tolist())]['rating']
            ratings_user_y = neighbour_movies[neighbour_movies['movielensID'].isin(
                current_user_ratings['movielensID'].tolist())]['rating']

            try:
                # similarity = cosine_similarity(intersection['rating_x'].reshape(1, -1),
                #                                intersection['rating_y'].reshape(1, -1))[0][0]
                similarity = cosine_similarity(ratings_user_x.reshape(1, -1), ratings_user_y.reshape(1, -1))[0][0]
            except ValueError:
                similarity = 0

            denominator += abs(similarity)
            neighbour_movie_rating = neighbour_movies[neighbour_movies['movielensID'] == movie[2]]['rating'].iloc[0]
            numerator += similarity * (neighbour_movie_rating - neighbour_movies['rating'].mean())

        try:
            prediction_user_item = user_average + numerator / denominator
        except ZeroDivisionError:
            continue

        predictions.append((movie[2], prediction_user_item, movie[0]))

    return predictions


# Return the top 30 most similar movies based on item-item collaborative filtering
def get_top_similar_collaborative_items(_all_movies, _all_ratings, target_movie_movielens_id):

    key = _all_movies[_all_movies['movielensID'] == target_movie_movielens_id].index[0]
    other_items = _all_movies.iloc[key + 1:]

    users_that_rated_target_movie = _all_ratings[_all_ratings['movielensID'] == target_movie_movielens_id]

    similarities = []

    for sub_key, sub_value in other_items.iterrows():

        neighbour_trailer_id, neighbour_movie_id = sub_value
        # print neighbour_movie_id

        users_that_rated_both_movies = _all_ratings[(_all_ratings['movielensID'] == neighbour_movie_id) &
                                                    (_all_ratings['userID'].isin(
                                                        users_that_rated_target_movie['userID'].tolist()))][
            'userID'].tolist()

        # print users_that_rated_both_movies

        user_ratings = [(_all_ratings[(_all_ratings['movielensID'] == target_movie_movielens_id)
                                      & (_all_ratings['userID'] == user)]['rating'].iloc[0],
                         _all_ratings[(_all_ratings['movielensID'] == neighbour_movie_id)
                                      & (_all_ratings['userID'] == user)]['rating'].iloc[0])
                        for user in users_that_rated_both_movies]

        if len(user_ratings) > 0:
            unpacked_ratings = zip(*user_ratings)
            ratings_x, ratings_y = np.array(unpacked_ratings[0]), np.array(unpacked_ratings[1])
            sim = cosine_similarity(ratings_y.reshape(1, -1), ratings_x.reshape(1, -1))[0][0]

        similarities.append((neighbour_movie_id, sim))

    return sort_desc(similarities)[:30]


def get_item_collaborative_predictions(userid, user_baseline, movies_to_predict, all_ratings, _ratings_by_movie, _global_average):

    predictions = []
    ratings_current_user = all_ratings.loc[userid]

    conn = sqlite3.connect('content/database.db')
    _all_ratings = pd.read_sql('select userID, movielensID, rating from movielens_rating order by userid', conn)
    _all_movies = pd.read_sql('SELECT t.id, m.movielensid FROM trailers t '
                              'JOIN movielens_movie m ON m.imdbidtt = t.imdbid WHERE t.best_file = 1 ORDER BY t.id',
                              conn)
    conn.close()

    for trailer_id, user_id, target_movielens_id in movies_to_predict:

        top_neighbours = get_top_similar_collaborative_items(_all_movies, _all_ratings, target_movielens_id)

        item_baseline = get_item_baseline(user_baseline, target_movielens_id, _ratings_by_movie, _global_average)
        user_item_baseline = (_avg_ratings + user_baseline + item_baseline)

        numerator, denominator = 0, 0
        for top_neighbour_id, sim in top_neighbours:

            numerator += sim * (ratings_current_user[ratings_current_user['movielensID'] == top_neighbour_id]['rating'].iloc[0] - user_item_baseline)
            denominator += abs(sim)

        prediction = numerator / denominator + user_item_baseline
        predictions.append((target_movielens_id, prediction))

    del _all_movies, _all_ratings
    return predictions


# Predict ratings
def build_user_profile(user_profiles, convnet_similarity_matrix):

    _general_baseline, _global_average = read_user_general_baseline()
    _ratings_by_movie = read_movie_general_baseline()

    conn = sqlite3.connect('content/database.db')
    all_ratings = pd.read_sql('select userID, movielensID, rating from movielens_rating', conn, index_col='userID')
    conn.close()

    print "all ratings read"
    # start = time.time()

    new_user_profiles = {}

    # todo paralelize this
    for userid, profile in user_profiles.iloc[:6000].iterrows():
        print userid, "userid"
        movies_set = profile['relevant_set'] + profile['irrelevant_set'] + profile['random_set']

        if type(movies_set) == float:
            continue

        # predictions_item_collaborative = get_item_collaborative_predictions(userid, profile['user_baseline'].iloc[0],
        #                                                                     movies_set, all_ratings, _ratings_by_movie,
        #                                                                     _global_average)

        predictions_collaborative = get_user_collaborative_predictions(userid, profile['avg'], movies_set, all_ratings)

        predictions_content_based = get_content_based_predictions(profile['user_baseline'].iloc[0], movies_set, profile['all_movies'],
                                                                  convnet_similarity_matrix, _ratings_by_movie, _global_average)

        new_user_profiles[userid] = {'datasets': {'relevant_movies': profile['relevant_set'],
                                                  'irrelevant_movies': profile['irrelevant_set']},
                                     'userid': userid,
                                     'predictions': {'deep': predictions_content_based,
                                                     'collaborative': predictions_collaborative}
                                     }

    return new_user_profiles


# def get_all_predictions(store_result, userid, profile, all_ratings, convnet_similarity_matrix, _ratings_by_movie, _global_average):
#
#     print userid, "userid"
#
#     movies_set = profile['relevant_set'] + profile['irrelevant_set'] + profile['random_set']
#
#     if type(movies_set) == float:
#         return
#
#     predictions_item_collaborative = get_item_collaborative_predictions(userid, 0, movies_set, all_ratings)
#
#     predictions_user_collaborative = get_user_collaborative_predictions(userid, profile['avg'], movies_set, all_ratings)
#
#     predictions_content_based = get_content_based_predictions(profile['user_baseline'].iloc[0], movies_set, profile['all_movies'],
#                                                               convnet_similarity_matrix, _ratings_by_movie, _global_average)
#
#     store_result[userid] = {'datasets': {'relevant_movies': profile['relevant_set'],
#                                          'irrelevant_movies': profile['irrelevant_set']},
#                                  'userid': userid,
#                                  'predictions': {'deep': predictions_content_based,
#                                                  'collaborative': predictions_user_collaborative}
#                                  }

