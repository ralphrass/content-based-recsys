import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import sort_desc
import pandas as pd


def get_item_collaborative_predictions(target_user_id, user_baseline, movies_to_predict, _all_ratings, _global_average,
                                       global_item_to_item_similarity):

    predictions = []
    _all_ratings_by_user = _all_ratings[_all_ratings['userID'] == target_user_id]

    for trailer_id, rating, target_movielens_id in movies_to_predict:

        item_ratings = _all_ratings[_all_ratings['movielensID'] == target_movielens_id]

        user_item_baseline = get_user_item_baseline(item_ratings, user_baseline, _global_average)
        top_neighbours = get_item_top_similarities(_all_ratings_by_user, target_movielens_id, _all_ratings,
                                                   item_ratings, global_item_to_item_similarity)

        p_ui = predict_user_item_rating(_all_ratings_by_user, user_item_baseline, top_neighbours)

        predictions.append((target_movielens_id, p_ui, trailer_id))
        # break
    return predictions


def get_user_item_baseline(item_ratings, user_baseline, _global_average):

    item_baseline = np.array([_rating - user_baseline - _global_average for _rating in item_ratings['rating']]).mean()
    user_item_baseline = _global_average + user_baseline + item_baseline

    return user_item_baseline


def get_item_top_similarities(_all_ratings_by_user, target_movielens_id, _all_ratings, item_ratings,
                              global_item_to_item_similarity):

    _limit_neighbourhood_to = 30
    similarities = []

    for key, movie in _all_ratings_by_user[_all_ratings_by_user['movielensID'] != target_movielens_id].iterrows():

        # items can repeat for each user
        if int(movie['movielensID']) in global_item_to_item_similarity and \
                        target_movielens_id in global_item_to_item_similarity[int(movie['movielensID'])]:

            similarities.append((movie['movielensID'],
                                 global_item_to_item_similarity[movie['movielensID']][target_movielens_id]))
        else:
            join_ratings = pd.merge(_all_ratings[_all_ratings['movielensID'] == movie['movielensID']],
                                    item_ratings, on='userID')

            try:
                ratings_x, ratings_y = (np.array(join_ratings['rating_x']), np.array(join_ratings['rating_y']))
                sim = cosine_similarity(ratings_x.reshape(1, -1), ratings_y.reshape(1, -1))[0][0]
                similarities.append((movie['movielensID'], sim))

                if int(movie['movielensID']) not in global_item_to_item_similarity:
                    global_item_to_item_similarity[int(movie['movielensID'])] = dict()

                global_item_to_item_similarity[int(movie['movielensID'])][target_movielens_id] = sim
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

