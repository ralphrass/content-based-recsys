# Simplicity is the final achievement. After one has played a vast quantity of notes and more notes, it is
# simplicity that emerges as the crowning reward of art. Frederic Chopin.

import sqlite3
import time
from utils.utils import read_user_general_baseline, read_movie_general_baseline
import pandas as pd
from multiprocessing import Process, Manager
from recommender_content_based import get_content_based_predictions, get_content_based_user_centroid_predictions
from recommender_collaborative import get_item_collaborative_predictions


def calculate_user_rating_predictions(_start, _end, _user_profiles, new_user_profiles, _convnet_similarity_matrix,
                                      _all_ratings):

    _general_baseline, _global_average = read_user_general_baseline()
    _ratings_by_movie = read_movie_general_baseline()
    global_item_to_item_similarity = dict()

    for userid, profile in _user_profiles.iloc[_start:_end].iterrows():
        print userid, "userid"
        movies_set = profile['relevant_set'] + profile['irrelevant_set'] + profile['random_set']

        if type(movies_set) == float:
            continue

        predictions_content_based_user_centroid = get_content_based_user_centroid_predictions(
            movies_set, _user_profiles, profile['relevant_centroid'], profile['irrelevant_centroid'],
            profile['avg'], _all_ratings)

        predictions_item_collaborative = get_item_collaborative_predictions(userid, profile['user_baseline'].iloc[0],
                                                                            movies_set, _all_ratings, _global_average,
                                                                            global_item_to_item_similarity)

        predictions_content_based = get_content_based_predictions(profile['user_baseline'].iloc[0], movies_set,
                                                                  profile['all_movies'], _convnet_similarity_matrix,
                                                                  _ratings_by_movie, _global_average)

        _base_weight = 0.25

        weight_content_based = _base_weight * (len(profile['relevant_centroid']) > 0) + \
                               _base_weight * (len(profile['irrelevant_centroid']) > 0)

        weight_collaborative_based = 1 - weight_content_based

        # set weight to content-based filtering: [0, 0.25, 0.5]. collaborative filtering is the complement
        predictions_weighted_hybrid = [(item[0][0], (item[0][1] * weight_content_based +
                                                    item[1][1] * weight_collaborative_based), item[0][2])
                                       for item in zip(predictions_content_based_user_centroid,
                                                      predictions_item_collaborative)]

        # get top 5 recommendations from collaborative filtering and add the top 5 from content-based
        predictions_mixing_hybrid = predictions_item_collaborative[:5]
        cp_mixing_hybrid = predictions_mixing_hybrid[:]

        for item in predictions_content_based_user_centroid:

            if len(predictions_mixing_hybrid) == 10:
                break

            if item[0] not in [x[0] for x in cp_mixing_hybrid]:
                predictions_mixing_hybrid.append(item)

        new_user_profiles[userid] = {'datasets': {'relevant_movies': profile['relevant_set'],
                                                  'irrelevant_movies': profile['irrelevant_set']},
                                     'predictions': {'deep': predictions_content_based,
                                                     'collaborative': predictions_item_collaborative,
                                                     'user-centroid': predictions_content_based_user_centroid,
                                                     'weighted-hybrid': predictions_weighted_hybrid,
                                                     'mixing-hybrid': predictions_mixing_hybrid
                                                     }
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

    _all_ratings = read_all_ratings()

    manager = Manager()
    new_user_profiles = manager.dict()
    # jobs = []
    _max = 10
    _step = 1

    for idx in range(0, _max, _step):
        calculate_user_rating_predictions(idx, idx + _step, _user_profiles, new_user_profiles,
                                          _convnet_similarity_matrix, _all_ratings)
    #     p = Process(target=calculate_user_rating_predictions, args=(idx, idx + _step, _user_profiles, new_user_profiles,
    #                                                                 _convnet_similarity_matrix, _all_ratings))
    #     jobs.append(p)
    #     p.start()
    #
    # for p in jobs:
    #     p.join()

    return dict(new_user_profiles)
