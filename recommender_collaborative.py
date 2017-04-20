import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import sort_desc
import pandas as pd
import time


def get_user_collaborative_predictions(movies_to_predict, _user_profiles, _all_ratings, _target_user_id, _user_avg):

    predictions = []
    # _limit_all_neighbours_to = 1000
    _limit_top_neighbours_to = 20

    target_user_ratings = _all_ratings[_all_ratings['userID'] == _target_user_id]

    for trailer_id, rating in movies_to_predict:

        # all neighbours
        neighbours = set(_all_ratings[_all_ratings['id'] == trailer_id]['userID'])
        # print len(neighbours), "is the current neighbourhood size"

        # find top neighbours
        top_neighbours = []
        for neighbour in neighbours:
            intersect = pd.merge(_all_ratings[_all_ratings['userID'] == neighbour], target_user_ratings, on='id')
            try:  # sometimes there is no intersection
                sim = cosine_similarity(intersect['rating_x'].reshape(1, -1), intersect['rating_y'].reshape(1, -1))
                top_neighbours.append((neighbour, sim[0][0], intersect))
            except ValueError:
                continue

        top_n = sort_desc(top_neighbours)[:_limit_top_neighbours_to]

        # print "Top N", top_n

        # predict rating
        numerator, denominator = (0, 0)
        for neighbour, sim, intersect in top_n:
            neighbour_rating = _all_ratings[(_all_ratings['userID'] == neighbour) & (_all_ratings['id'] == trailer_id)]['rating'].iloc[0]
            numerator += sim * (neighbour_rating - _user_profiles.loc[neighbour]['avg'])
            denominator += abs(sim)
        try:
            p_ui = _user_avg + numerator / denominator
        except ZeroDivisionError:
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

