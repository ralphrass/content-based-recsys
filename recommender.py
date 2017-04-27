# Simplicity is the final achievement. After one has played a vast quantity of notes and more notes, it is
# simplicity that emerges as the crowning reward of art. Frederic Chopin.

import sqlite3
import time
from utils.utils import read_user_general_baseline, read_movie_general_baseline
import pandas as pd
from multiprocessing import Process, Manager
from recommender_content_based import get_content_based_predictions, get_content_based_user_bof_predictions
from recommender_collaborative import get_user_collaborative_predictions_precomputed_similarities, \
    get_item_collaborative_predictions_precomputed_similarities
from recommender_hybrid import get_hybrid_recommendations
from recommender_svd import load_svd, get_predictions_svd, map_movie_to_index
from recommender_linear_regression import get_predictions_linear_regression
from utils.opening_feat import load_features


def calculate_user_rating_predictions(_start, _end, _user_profiles, new_user_profiles, _convnet_similarity_matrix,
                                      _low_level_similarity_matrix, _all_ratings, svd_matrix, movies_to_index,
                                      _general_baseline, _global_average, _ratings_by_movie,
                                      _user_user_collaborative_matrix, _item_item_collaborative_matrix):

    user_index = 0
    svd_u, svd_s, svd_v = svd_matrix

    for userid, profile in _user_profiles.iloc[_start:_end].iterrows():

        print userid, "userid"
        print "relevant set", profile['relevant_set']

        movies_set = profile['relevant_set'] + profile['irrelevant_set'] + profile['random_set']

        if type(movies_set) == float:
            continue
        # exit()

        predictions_svd = get_predictions_svd(movies_set, svd_u, svd_s, svd_v, movies_to_index, user_index, profile['avg'])
        print predictions_svd
        # exit()

        # predictions_linear_regression = get_predictions_linear_regression(movies_set, _deep_features, _user_theta_vectors, userid)
        # print predictions_linear_regression
        # exit()

        # start = time.time()
        # predictions_item_collaborative = get_item_collaborative_predictions(movies_set, _all_ratings, userid)
        # print predictions_item_collaborative
        # print "item-item CF tok", time.time() - start, "seconds"
        predictions_item_collaborative = get_item_collaborative_predictions_precomputed_similarities(
            movies_set, _all_ratings, userid, _item_item_collaborative_matrix, profile['user_baseline'],
            _ratings_by_movie, _global_average)
        print predictions_item_collaborative
        # exit()

        # start = time.time()
        # print "starting user-collaborative recommendations..."
        # predictions_user_collaborative = get_user_collaborative_predictions(movies_set, _user_profiles, _all_ratings,
        #                                                                     userid, _user_profiles.loc[userid]['avg'])
        # print predictions_user_collaborative
        # print "user-user CF tok", time.time() - start, "seconds"
        # PS: it uses pearson r similarity. it filters out when the intersection has less than 5 items.
        predictions_user_collaborative = get_user_collaborative_predictions_precomputed_similarities(
            movies_set, _user_profiles, _all_ratings, userid, profile['avg'], _user_user_collaborative_matrix)
        print predictions_user_collaborative

        predictions_content_based = get_content_based_predictions(profile['user_baseline'], movies_set,
                                                                  profile['all_movies'], _convnet_similarity_matrix,
                                                                  _ratings_by_movie, _global_average)
        print predictions_content_based

        # predictions_user_bof_based = get_content_based_user_bof_predictions(movies_set, profile['avg'], _all_ratings,
        #                                                                     _user_bof_sim_matrix, _user_profiles,
        #                                                                     userid)
        # print "Predictions", predictions_user_bof_based

        predictions_low_level = get_content_based_predictions(profile['user_baseline'], movies_set,
                                                              profile['all_movies'], _low_level_similarity_matrix,
                                                              _ratings_by_movie, _global_average)
        print predictions_low_level
        # predictions_hybrid = get_hybrid_recommendations(predictions_content_based + predictions_user_collaborative, movies_set)

        # print predictions_content_based
        # print predictions_low_level
        # print predictions_hybrid

        new_user_profiles[userid] = {'datasets': {'relevant_movies': profile['relevant_set'],
                                                  'irrelevant_movies': profile['irrelevant_set']},
                                     'predictions': {
                                                    'deep': predictions_content_based,
                                                    'low-level': predictions_low_level,
                                                    'svd': predictions_svd,
                                                    # 'user-bof': predictions_user_bof_based
                                                    # 'linear-regression': predictions_linear_regression,
                                                    'item-collaborative': predictions_item_collaborative,
                                                    'user-collaborative': predictions_user_collaborative,
                                                    # 'hybrid': predictions_hybrid
                                                     }
                                     }
        user_index += 1


def read_all_ratings():

    start = time.time()
    conn = sqlite3.connect('content/database.db')
    _all_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
                               'join movielens_movie m on m.movielensid = r.movielensid '
                               'join trailers t on t.imdbid = m.imdbidtt '
                               'where userid < 5000 '
                               'order by userid', conn)
    conn.close()
    print "all ratings read in", time.time() - start, "seconds"
    return _all_ratings


# Predict ratings
def build_user_profile(_user_profiles, _convnet_similarity_matrix, _low_level_similarity_matrix,
                       _user_user_collaborative_matrix, _item_item_collaborative_matrix):

    _all_ratings = read_all_ratings()
    # _all_ratings = None

    manager = Manager()
    new_user_profiles = manager.dict()
    # jobs = []
    # _max = 3112
    _max = 1
    _step = 1

    _general_baseline, _global_average = read_user_general_baseline()
    # this is for content-based recommendations
    _ratings_by_movie = read_movie_general_baseline()

    # this is for SVD
    svd_matrix = load_svd()
    movies_to_index = map_movie_to_index()
    # svd_matrix = None
    # movies_to_index = None

    # this is for linear regression
    # _deep_features = load_features('content/bof_128.bin')
    # _user_theta_vectors = load_features('content/users_theta_vectors.pkl')

    for idx in range(0, _max, _step):
        calculate_user_rating_predictions(idx, idx + _step, _user_profiles, new_user_profiles,
                                          _convnet_similarity_matrix, _low_level_similarity_matrix, _all_ratings,
                                          svd_matrix, movies_to_index, _general_baseline, _global_average,
                                          _ratings_by_movie, _user_user_collaborative_matrix,
                                          _item_item_collaborative_matrix)
    #     p = Process(target=calculate_user_rating_predictions, args=(idx, idx + _step, _user_profiles, new_user_profiles,
    #                                                                 _convnet_similarity_matrix, _all_ratings))
    #     jobs.append(p)
    #     p.start()
    #
    # for p in jobs:
    #     p.join()

    return dict(new_user_profiles)
