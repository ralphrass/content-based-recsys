import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import sort_desc
import pandas as pd


def get_user_collaborative_predictions(movies_to_predict, _all_ratings, _target_user_id, _user_avg):

    user_ratings = _all_ratings[_all_ratings['userid'] == _target_user_id]

    for trailer_id, rating in movies_to_predict:

        similarities = []

        for key, movie in user_ratings[user_ratings['userid'] != _target_user_id][:1000].iterrows():  # todo limited to
            join_ratings = pd.merge(_all_ratings[_all_ratings['userid'] == movie['userid']], user_ratings, on='id')
            ratings_x, ratings_y = (np.array(join_ratings['rating_x']), np.array(join_ratings['rating_y']))
            sim = cosine_similarity(ratings_x.reshape(1, -1), ratings_y.reshape(1, -1))
            similarities.append((movie['userid'], sim))
        top_neighbours = sort_desc(similarities)[:100]


def get_item_collaborative_predictions(target_user_id, user_baseline, movies_to_predict, _all_ratings, _global_average):

    predictions = []
    _all_ratings_by_user = _all_ratings[_all_ratings['userID'] == target_user_id]

    for trailer_id, rating in movies_to_predict:

        item_ratings = _all_ratings[_all_ratings['id'] == trailer_id]

        user_item_baseline = get_user_item_baseline(item_ratings, user_baseline, _global_average)
        top_neighbours = get_item_top_similarities(_all_ratings_by_user, trailer_id, _all_ratings,
                                                   item_ratings)

        p_ui = predict_user_item_rating(_all_ratings_by_user, user_item_baseline, top_neighbours)

        predictions.append((trailer_id, p_ui))
        # break
    return sort_desc(predictions)


def get_user_item_baseline(item_ratings, user_baseline, _global_average):

    item_baseline = np.array([_rating - user_baseline - _global_average for _rating in item_ratings['rating']]).mean()
    user_item_baseline = _global_average + user_baseline + item_baseline

    return user_item_baseline


def get_item_top_similarities(_all_ratings_by_user, trailer_id, _all_ratings, item_ratings):

    _limit_neighbourhood_to = 30
    similarities = []

    for key, movie in _all_ratings_by_user[_all_ratings_by_user['id'] != trailer_id][:1000].iterrows():  # todo limited to

        join_ratings = pd.merge(_all_ratings[_all_ratings['id'] == movie['id']],
                                item_ratings, on='userID')
        try:
            ratings_x, ratings_y = (np.array(join_ratings['rating_x']), np.array(join_ratings['rating_y']))
            sim = cosine_similarity(ratings_x.reshape(1, -1), ratings_y.reshape(1, -1))[0][0]
            similarities.append((movie['id'], sim))
        except ValueError:
            continue

    return sort_desc(similarities)[:_limit_neighbourhood_to]


def predict_user_item_rating(_all_ratings_by_user, user_item_baseline, top_neighbours):

    try:
        p_ui_numerator = sum([s_ij * (_all_ratings_by_user[_all_ratings_by_user['id'] == movie_id]
                                      ['rating'].iloc[0] - user_item_baseline)
                              for movie_id, s_ij in top_neighbours])
        p_ui_denominator = sum([abs(s_ij) for n, s_ij in top_neighbours])
        p_ui = p_ui_numerator / p_ui_denominator + user_item_baseline
    except ZeroDivisionError:
        return 0

    return p_ui

