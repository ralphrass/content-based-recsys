import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import sort_desc, get_item_baseline
import pandas as pd
import time

_avg_ratings = 3.51611876907599


def get_user_collaborative_predictions_precomputed_similarities(movies_to_predict, _user_profiles, _all_ratings,
                                                                _target_user_id, _user_avg, _user_user_sim_matrix):
    global _avg_ratings

    predictions = []
    _limit_top_neighbours_to = 50

    for trailer_id, rating in movies_to_predict:

        # all neighbours
        rating_neighbors = set(_all_ratings[_all_ratings['id'] == trailer_id]['userID'])
        # print len(rating_neighbors), "is the current neighbourhood size"
        # break

        # find top neighbours
        top_neighbors = [(neighbor, sim) for neighbor, sim in _user_user_sim_matrix[_target_user_id]
                         if neighbor in rating_neighbors]

        top_n = sort_desc(top_neighbors)[:_limit_top_neighbours_to]

        # print "Top N", top_n

        # predict rating
        numerator, denominator = (0, 0)
        for neighbour, sim in top_n:

            neighbour_rating = _all_ratings[(_all_ratings['userID'] == neighbour)
                                            & (_all_ratings['id'] == trailer_id)]['rating'].iloc[0]
            numerator += sim * (neighbour_rating - _user_profiles.loc[neighbour]['avg'])
            denominator += abs(sim)
        try:
            p_ui = _user_avg + numerator / denominator
        except ZeroDivisionError:
            p_ui = 0

        predictions.append((trailer_id, p_ui))

    return sort_desc(predictions)


# def get_user_collaborative_predictions(movies_to_predict, _user_profiles, _all_ratings, _target_user_id, _user_avg):
#
#     predictions = []
#     # _limit_all_neighbours_to = 1000
#     _limit_top_neighbours_to = 20
#
#     target_user_ratings = _all_ratings[_all_ratings['userID'] == _target_user_id]
#
#     for trailer_id, rating in movies_to_predict:
#
#         # all neighbours
#         neighbours = set(_all_ratings[_all_ratings['id'] == trailer_id]['userID'])
#         # print len(neighbours), "is the current neighbourhood size"
#
#         # find top neighbours
#         top_neighbours = []
#         for neighbour in neighbours:
#             intersect = pd.merge(_all_ratings[_all_ratings['userID'] == neighbour], target_user_ratings, on='id')
#             try:  # sometimes there is no intersection
#                 sim = cosine_similarity(intersect['rating_x'].reshape(1, -1), intersect['rating_y'].reshape(1, -1))
#                 top_neighbours.append((neighbour, sim[0][0], intersect))
#             except ValueError:
#                 continue
#
#         top_n = sort_desc(top_neighbours)[:_limit_top_neighbours_to]
#
#         # print "Top N", top_n
#
#         # predict rating
#         numerator, denominator = (0, 0)
#         for neighbour, sim, intersect in top_n:
#             neighbour_rating = _all_ratings[(_all_ratings['userID'] == neighbour) & (_all_ratings['id'] == trailer_id)]['rating'].iloc[0]
#             numerator += sim * (neighbour_rating - _user_profiles.loc[neighbour]['avg'])
#             denominator += abs(sim)
#         try:
#             p_ui = _user_avg + numerator / denominator
#         except ZeroDivisionError:
#             p_ui = 0
#
#         predictions.append((trailer_id, p_ui))
#
#     return sort_desc(predictions)


def get_item_collaborative_predictions_precomputed_similarities(movies_to_predict, _all_ratings, _target_user_id,
                                                                _item_item_sim_matrix):
    predictions = []
    _limit_top_neighbours_to = 50
    # target_user_ratings = _all_ratings[_all_ratings['userID'] == _target_user_id]

    for trailer_id, rating in movies_to_predict:

        # print "Trailer id is", trailer_id
        try:
            _all_sim_items = _item_item_sim_matrix[trailer_id]

            # print "All sims are", _all_sim_items
            # break
            # _allowed_sim_items = _all_sim_items[:_limit_top_neighbours_to]
            allowed_sim_items = []

            for item in _all_sim_items:

                rating = _all_ratings[(_all_ratings['userID'] == _target_user_id) &
                                      (_all_ratings['id'] == item[0])]['rating']
                try:  # the current user rated this item
                    rating = float(rating)
                    allowed_sim_items.append((item[1], rating))
                except TypeError:
                    continue

                if len(allowed_sim_items) == _limit_top_neighbours_to:
                    break

            # print "Allowed:", allowed_sim_items
            # b_ui = get_item_baseline(user_baseline, trailer_id, _ratings_by_movie, _global_average)

            try:
                p_ui = (sum([sim * rating for sim, rating in allowed_sim_items]) /
                       sum([abs(sim) for sim, rating in allowed_sim_items]))
            except ZeroDivisionError:
                p_ui = 0
        except KeyError:
            p_ui = 0

        predictions.append((trailer_id, p_ui))

    return sort_desc(predictions)


def get_item_collaborative_predictions(movies_to_predict, _all_ratings, _target_user_id):

    predictions = []
    _limit_top_neighbours_to = 20

    target_user_ratings = _all_ratings[_all_ratings['userID'] == _target_user_id]

    for trailer_id, rating in movies_to_predict:

        top_neighbours = []
        # find most similar movies
        for rated_movie in target_user_ratings['id']:

            intersect = pd.merge(_all_ratings[_all_ratings['id'] == rated_movie],
                                 _all_ratings[_all_ratings['id'] == trailer_id], on='userID')
            # print intersect
            try:
                sim = cosine_similarity(intersect['rating_x'].reshape(1, -1), intersect['rating_y'].reshape(1, -1))
                top_neighbours.append((rated_movie, sim[0][0], intersect))
            except ValueError:
                continue
        top_n = sort_desc(top_neighbours)[:_limit_top_neighbours_to]

        numerator, denominator = (0, 0)
        for neighbour, sim, intersect in top_n:
            user_rating = _all_ratings[(_all_ratings['id'] == neighbour) & (_all_ratings['userID'] == _target_user_id)]['rating'].iloc[0]
            numerator += sim * user_rating
            denominator += abs(sim)

        try:
            p_ui = numerator / denominator
        except ZeroDivisionError:
            p_ui = 0

        predictions.append((trailer_id, p_ui))

    return sort_desc(predictions)

