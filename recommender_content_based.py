import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import get_item_baseline, sort_desc
import operator

_avg_ratings = 3.51611876907599


def cosine(movieI, movieJ, feature_vector):

    traileri = movieI[0]
    trailerj = movieJ[0]

    try:
        featuresI = feature_vector[traileri]
        featuresJ = feature_vector[trailerj]
    except KeyError:
        return 0

    return cosine_similarity([featuresI], [featuresJ])


def predict_user_rating(user_baseline, trailer_id, all_similarities, _ratings_by_movie, _global_average):

    global _avg_ratings

    item_baseline = get_item_baseline(user_baseline, trailer_id, _ratings_by_movie, _global_average)
    user_item_baseline = (_avg_ratings + user_baseline + item_baseline)

    numerator = sum((rating - user_item_baseline) * sim if sim > 0 else 0 for rating, sim in all_similarities)
    denominator = reduce(operator.add, [abs(x[1]) for x in all_similarities])
    try:
        prediction = (numerator / denominator) + user_item_baseline
    except ZeroDivisionError:
        prediction = 0

    return prediction


def get_content_based_predictions(user_baseline, movies, all_movies, sim_matrix, _ratings_by_movie, _global_average):

    predictions = [(movie[0], predict_user_rating(user_baseline, movie[0],
                                                  [(movieJ[1], sim_matrix[movieJ[0]][movie[0]])
                                                   for movieJ in all_movies], _ratings_by_movie, _global_average))
                   for movie in movies]

    return sort_desc(predictions)


# def get_content_based_user_centroid_predictions(movies, _user_profiles, _user_relevant_centroid,
#                                                 _user_irrelevant_centroid, _user_average, _all_ratings):
#     predictions = []
#
#     for trailer_id, user_rating in movies:
#
#         numerator, denominator = (0, 0)
#         # Find neighbours that evaluated this movie
#         _all_users = _all_ratings[_all_ratings['id'] == trailer_id][:1000]  # todo limited to
#         for key, value in _all_users.iterrows():
#
#             user_id = int(value['userID'])
#
#             try:
#                 if len(_user_relevant_centroid) > 1 & len(_user_irrelevant_centroid) > 1 & \
#                         len(_user_profiles.loc[user_id]['relevant_centroid']) > 1 & \
#                         len(_user_profiles.loc[user_id]['irrelevant_centroid']) > 1:
#                     sim = cosine_similarity(np.concatenate((_user_profiles.loc[user_id]['relevant_centroid'],
#                                                            _user_profiles.loc[user_id]['irrelevant_centroid'])),
#                                             np.concatenate((_user_relevant_centroid, _user_irrelevant_centroid)))[0][0]
#                 else:
#                     sim = cosine_similarity(_user_profiles.loc[user_id]['relevant_centroid'], _user_relevant_centroid)[0][0]
#
#                 if sim != 0:
#                     numerator += sim * (value['rating'] - _user_profiles.loc[user_id]['avg'])
#                     denominator += abs(sim)
#
#             except ValueError:
#                 continue
#         try:
#             p_ui = _user_average + numerator / denominator
#         except ZeroDivisionError:
#             continue
#
#         predictions.append((trailer_id, p_ui))
#
#     return sort_desc(predictions)


# def get_user_item_interaction(_user_centroid, movies, movies_bof):
#
#     similarities = []
#
#     for movie_id, r in movies:
#         sim = cosine_similarity(movies_bof[movie_id].reshape(1, -1), _user_centroid.reshape(1, -1))[0][0]
#         similarities.append((movie_id, sim))
#
#     return sort_desc(similarities)


def get_content_based_user_bof_predictions(_movies_set, _user_avg, _all_ratings, _user_user_sim_matrix, _user_profiles,
                                           _target_user_id):

    predictions = []

    for trailer_id, rating in _movies_set:

        rating_neighbors = _all_ratings[_all_ratings['id'] == trailer_id]
        rating_neighbors_users = list(rating_neighbors['userID'])

        selected_neighbors = [(user, sim, rating_neighbors[rating_neighbors['userID'] == user]['rating'].iloc[0])
                              for user, sim in _user_user_sim_matrix[_target_user_id] if user in rating_neighbors_users]
        # print "Selected Neighbors"
        # print selected_neighbors
        # break

        try:
            # print sum([abs(sim) for u, sim, r in selected_neighbors])
            # break
            p_ui = _user_avg + sum([sim * (_user_profiles.loc[user]['avg'] - user_rating)
                                    for user, sim, user_rating in selected_neighbors]) / \
                               sum([abs(sim) for u, sim, r in selected_neighbors])
        except ZeroDivisionError:
            p_ui = 0
        predictions.append((trailer_id, p_ui))

    return sort_desc(predictions)