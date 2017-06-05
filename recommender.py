# Simplicity is the final achievement. After one has played a vast quantity of notes and more notes, it is
# simplicity that emerges as the crowning reward of art. Frederic Chopin.

import sqlite3
import time
from utils.utils import read_user_general_baseline, read_movie_general_baseline
import pandas as pd
from multiprocessing import Process, Manager
from recommender_content_based import get_content_based_predictions, get_tag_based_predictions
from recommender_collaborative import get_user_collaborative_predictions_precomputed_similarities, \
    get_item_collaborative_predictions, get_item_collaborative_predictions_precomputed_similarities
from recommender_hybrid import get_weighted_hybrid_recommendations, get_switching_hybrid_recommendations
from recommender_svd import load_svd, map_movie_to_index, get_predictions_svd


def calculate_user_rating_predictions(_start, _end, _user_profiles, new_user_profiles, _convnet_similarity_matrix,
                                      _low_level_similarity_matrix, _all_ratings, _global_average, _ratings_by_movie,
                                      _user_user_collaborative_matrix, _trailers_tfidf_sims_matrix,
                                      _trailers_tfidf_synopsis_sims_matrix,
                                      svd_matrix, movies_to_index):
                                      # , _item_item_collaborative_matrix):

    user_index = 0
    compute_collaborative = 1
    # count = 1
    # print_at = 10
    # svd_u, svd_s, svd_v = svd_matrix
    avg_time_ll, avg_time_deep, avg_time_tag, avg_time_synopsys, avg_time_item_collaborative, \
    avg_time_user_collaborative, avg_time_weighted_hybrid, avg_time_switching_hybrid = 0, 0, 0, 0, 0, 0, 0, 0

    for userid, profile in _user_profiles.iloc[_start:_end].iterrows():

        # count += 1
        # if count == print_at:
        print userid, "userid"
        #    count = 0
        # print "relevant set", profile['relevant_set']

        movies_set = profile['relevant_set'] + profile['irrelevant_set'] + profile['random_set']

        if type(movies_set) == float:
            continue

        start = time.time()
        print "Computing LL predictions..."
        predictions_low_level = get_content_based_predictions(profile['user_baseline'], movies_set,
                                                              profile['all_movies'], _low_level_similarity_matrix,
                                                              _ratings_by_movie, _global_average)
        # print "LL tok", time.time() - start, "seconds."
        avg_time_ll += time.time() - start

        start = time.time()
        print "Computing Deep predictions..."
        predictions_content_based = get_content_based_predictions(profile['user_baseline'], movies_set,
                                                                  profile['all_movies'], _convnet_similarity_matrix,
                                                                  _ratings_by_movie, _global_average)
        # print "Deep tok", time.time() - start, "seconds."
        avg_time_deep += time.time() - start

        start = time.time()
        print "Computing Synopsis based predictions..."
        predictions_synopsis_based = get_tag_based_predictions(profile['user_baseline'], movies_set,
                                                                   profile['all_movies'],
                                                                   _trailers_tfidf_synopsis_sims_matrix,
                                                                   _ratings_by_movie, _global_average)
        # print "Synopsis tok", time.time() - start, "seconds."
        avg_time_synopsys += time.time() - start

        start = time.time()
        print "Computing Tag based predictions..."
        predictions_tfidf_based = get_tag_based_predictions(profile['user_baseline'], movies_set,
                                                                profile['all_movies'],
                                                                _trailers_tfidf_sims_matrix, _ratings_by_movie,
                                                                _global_average)
        # print "Tag tok", time.time() - start, "seconds."
        avg_time_tag += time.time() - start

        if compute_collaborative == 1:

            start = time.time()
            print "Computing SVD predictions..."
            predictions_svd = get_predictions_svd(movies_set, svd_matrix, movies_to_index, user_index, profile['avg'])
            # print predictions_svd

            start = time.time()
            print "Computing Item-Collaborative predictions..."
            predictions_item_collaborative = get_item_collaborative_predictions(movies_set, _all_ratings, userid)
            # print "Computing Item-Collaborative precomputed sims predictions..."
            # predictions_item_collaborative_precomputed_sims = \
            #     get_item_collaborative_predictions_precomputed_similarities(movies_set, _all_ratings, userid,
            #                                                                 _item_item_collaborative_matrix)
            # print "Item_colaborative tok", time.time() - start, "seconds."
            avg_time_item_collaborative += time.time() - start
            # print predictions_item_collaborative
            # print predictions_item_collaborative_precomputed_sims


            # start = time.time()
            # print "Computing Switching Hybrid predictions..."
            # predictions_switching_hybrid = get_switching_hybrid_recommendations(movies_set, _all_ratings, userid,
            #                                                                   _convnet_similarity_matrix)
            # print "Switching Hybrid tok", time.time() - start, "seconds."
            # avg_time_switching_hybrid += time.time() - start

            start = time.time()
            print "Computing User-Collaborative predictions..."
            # PS: it uses pearson r similarity. it filters out when the intersection has less than 5 items.
            predictions_user_collaborative = get_user_collaborative_predictions_precomputed_similarities(
                movies_set, _user_profiles, _all_ratings, userid, profile['avg'], _user_user_collaborative_matrix)
            # print "User-Collaborative tok", time.time() - start, "seconds."
            avg_time_user_collaborative += time.time() - start

            start = time.time()
            print "Computing Weighted Hybrid predictions..."

            print "Relevant Set", profile['relevant_set']
            print "Low Level:", predictions_low_level
            print "RNC:", predictions_content_based
            print "FC-I", predictions_item_collaborative
            print "FC-U", predictions_user_collaborative
            print "Sinopse", predictions_synopsis_based
            print "Tags", predictions_tfidf_based

            # print "Weighted Hybrid tok", time.time() - start, "seconds."
            avg_time_weighted_hybrid += time.time() - start

            # Weighted Hybrid series
            predictions_wh_synopsis_tags = get_weighted_hybrid_recommendations(
                predictions_synopsis_based + predictions_tfidf_based, movies_set)
            predictions_wh_synopsis_low_level = get_weighted_hybrid_recommendations(
                predictions_synopsis_based + predictions_low_level, movies_set)
            predictions_wh_synopsis_item_collaborative = get_weighted_hybrid_recommendations(
                predictions_synopsis_based + predictions_item_collaborative, movies_set)
            predictions_wh_synopsis_deep = get_weighted_hybrid_recommendations(
                predictions_synopsis_based + predictions_content_based, movies_set)
            predictions_wh_synopsis_user_collaborative = get_weighted_hybrid_recommendations(
                predictions_synopsis_based + predictions_user_collaborative, movies_set)
            predictions_wh_tags_low_level = get_weighted_hybrid_recommendations(
                predictions_tfidf_based + predictions_low_level, movies_set)
            predictions_wh_tags_item_collaborative = get_weighted_hybrid_recommendations(
                predictions_tfidf_based + predictions_item_collaborative, movies_set)
            predictions_wh_tags_deep = get_weighted_hybrid_recommendations(
                predictions_tfidf_based + predictions_content_based, movies_set)
            predictions_wh_tags_user_collaborative = get_weighted_hybrid_recommendations(
                predictions_tfidf_based + predictions_user_collaborative, movies_set)
            predictions_wh_low_level_item_collaborative = get_weighted_hybrid_recommendations(
                predictions_low_level + predictions_item_collaborative, movies_set)
            predictions_wh_low_level_deep = get_weighted_hybrid_recommendations(
                predictions_low_level + predictions_content_based, movies_set)
            predictions_wh_low_level_user_collaborative = get_weighted_hybrid_recommendations(
                predictions_low_level + predictions_user_collaborative, movies_set)
            predictions_wh_item_collaborative_deep = get_weighted_hybrid_recommendations(
                predictions_item_collaborative + predictions_content_based, movies_set)
            predictions_wh_item_collaborative_user_collaborative = get_weighted_hybrid_recommendations(
                predictions_item_collaborative + predictions_user_collaborative, movies_set)
            predictions_wh_deep_user_collaborative = get_weighted_hybrid_recommendations(
                predictions_content_based + predictions_user_collaborative, movies_set)

            # predictions_hybrid_content_and_item_collaborative = get_weighted_hybrid_recommendations(
            #     predictions_content_based + predictions_item_collaborative, movies_set)
            # predictions_hybrid_user_and_item_collaborative = get_weighted_hybrid_recommendations(
            #     predictions_user_collaborative + predictions_item_collaborative, movies_set)

            new_user_profiles[userid] = {
                'datasets': {'relevant_movies': profile['relevant_set'],
                             'irrelevant_movies': profile['irrelevant_set']},
                'predictions': {
                    'deep': predictions_content_based,
                    'low-level': predictions_low_level,
                    'tfidf': predictions_tfidf_based,
                    'synopsis': predictions_synopsis_based,
                    'svd': predictions_svd,
                    'item-collaborative': predictions_item_collaborative,
                    'user-collaborative': predictions_user_collaborative,
                    # 'switching-hybrid': predictions_switching_hybrid,
                    'weighted-hybrid-synopsis-tags': predictions_wh_synopsis_tags,
                    'weighted-hybrid-synopsis-low-level': predictions_wh_synopsis_low_level,
                    'weighted-hybrid-synopsis-item-collaborative': predictions_wh_synopsis_item_collaborative,
                    'weighted-hybrid-synopsis-deep': predictions_wh_synopsis_deep,
                    'weighted-hybrid-synopsis-user-collaborative': predictions_wh_synopsis_user_collaborative,
                    'weighted-hybrid-tags-low-level': predictions_wh_tags_low_level,
                    'weighted-hybrid-tags-item-collaborative': predictions_wh_tags_item_collaborative,
                    'weighted-hybrid-tags-deep': predictions_wh_tags_deep,
                    'weighted-hybrid-tags-user-collaborative': predictions_wh_tags_user_collaborative,
                    'weighted-hybrid-low-level-item-collaborative': predictions_wh_low_level_item_collaborative,
                    'weighted-hybrid-low-level-deep': predictions_wh_low_level_deep,
                    'weighted-hybrid-low-level-user-collaborative': predictions_wh_low_level_user_collaborative,
                    'weighted-hybrid-item-collaborative-deep': predictions_wh_item_collaborative_deep,
                    'weighted-hybrid-item-collaborative-user-collaborative': predictions_wh_item_collaborative_user_collaborative,
                    'weighted-hybrid-deep-user-collaborative': predictions_wh_deep_user_collaborative
                }
            }
        else:
            new_user_profiles[userid] = {
                'datasets': {'relevant_movies': profile['relevant_set'],
                             'irrelevant_movies': profile['irrelevant_set']},
                'predictions': {
                    'deep': predictions_content_based,
                    'low-level': predictions_low_level,
                    'tfidf': predictions_tfidf_based,
                    'synopsis': predictions_synopsis_based
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
                       _user_user_collaborative_matrix, _trailers_tfidf_sims_matrix,
                       _trailers_tfidf_synopsis_sims_matrix):
                       # , _item_item_collaborative_matrix):

    _all_ratings = read_all_ratings()
    # _all_ratings = None

    manager = Manager()
    new_user_profiles = manager.dict()
    # jobs = []
    # _max = 3112
    _max = 3
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

    print "will calculate predictions for", _max, "users."

    for idx in range(0, _max, _step):
        calculate_user_rating_predictions(idx, idx + _step, _user_profiles, new_user_profiles,
                                          _convnet_similarity_matrix, _low_level_similarity_matrix, _all_ratings,
                                          _global_average, _ratings_by_movie, _user_user_collaborative_matrix,
                                          _trailers_tfidf_sims_matrix, _trailers_tfidf_synopsis_sims_matrix,
                                          svd_matrix, movies_to_index)
                                          # _item_item_collaborative_matrix)
    #     p = Process(target=calculate_user_rating_predictions, args=(idx, idx + _step, _user_profiles, new_user_profiles,
    #                                                                 _convnet_similarity_matrix, _all_ratings))
    #     jobs.append(p)
    #     p.start()
    #
    # for p in jobs:
    #     p.join()

    return dict(new_user_profiles)

