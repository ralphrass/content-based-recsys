# Simplicity is the final achievement. After one has played a vast quantity of notes and more notes, it is
# simplicity that emerges as the crowning reward of art. Frederic Chopin.

import sqlite3
import time
from utils.utils import read_user_general_baseline, read_movie_general_baseline, extract_features
import pandas as pd
from multiprocessing import Process, Manager
from recommender_content_based import get_content_based_predictions, get_content_based_user_centroid_predictions, \
    get_user_item_interaction
from recommender_collaborative import get_item_collaborative_predictions
from recommender_hybrid import get_weighted_hybrid_recommendations, get_mixing_hybrid_recommendations


def calculate_user_rating_predictions(_start, _end, _user_profiles, new_user_profiles, _convnet_similarity_matrix,
                                      _all_ratings):

    _general_baseline, _global_average = read_user_general_baseline()
    _ratings_by_movie = read_movie_general_baseline()
    DEEP_FEATURES_BOF = extract_features('content/bof_128.bin')

    for userid, profile in _user_profiles.iloc[_start:_end].iterrows():

        print userid, "userid"

        movies_set = profile['relevant_set'] + profile['irrelevant_set'] + profile['random_set']

        if type(movies_set) == float:
            continue
        # exit()

        # predictions_user_centroid_item_interaction = \
        #     get_user_item_interaction(profile['relevant_centroid'], movies_set, DEEP_FEATURES_BOF)
        # print predictions_user_centroid_item_interaction
        # exit()

        predictions_content_based_user_centroid = \
            get_content_based_user_centroid_predictions(movies_set, _user_profiles, profile['relevant_centroid'],
                                                        profile['irrelevant_centroid'], profile['avg'], _all_ratings)
        print predictions_content_based_user_centroid
        predictions_item_collaborative = get_item_collaborative_predictions(userid, profile['user_baseline'].iloc[0],
                                                                            movies_set, _all_ratings, _global_average)
        print predictions_item_collaborative
        # exit()
        predictions_content_based = get_content_based_predictions(profile['user_baseline'].iloc[0], movies_set,
                                                                  profile['all_movies'], _convnet_similarity_matrix,
                                                                  _ratings_by_movie, _global_average)
        predictions_weighted_hybrid = \
            get_weighted_hybrid_recommendations(profile['relevant_centroid'], profile['irrelevant_centroid'],
                                                predictions_content_based_user_centroid, predictions_item_collaborative)
        predictions_mixing_hybrid = \
            get_mixing_hybrid_recommendations(predictions_item_collaborative, predictions_content_based_user_centroid)

        print predictions_content_based
        print predictions_weighted_hybrid
        print predictions_mixing_hybrid

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
    _all_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
                               'join movielens_movie m on m.movielensid = r.movielensid '
                               'join trailers t on t.imdbid = m.imdbidtt '
                               'order by userid', conn)
    conn.close()
    print "all ratings read in", time.time() - start, "seconds"
    return _all_ratings


# Predict ratings
def build_user_profile(_user_profiles, _convnet_similarity_matrix):

    _all_ratings = read_all_ratings()

    manager = Manager()
    new_user_profiles = manager.dict()
    # jobs = []
    _max = 5
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
