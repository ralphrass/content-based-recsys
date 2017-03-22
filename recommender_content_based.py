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


def get_content_based_user_centroid_predictions(movies, _user_profiles, _user_relevant_centroid,
                                                _user_irrelevant_centroid, _user_average, _all_ratings):
    predictions = []

    for trailer_id, user_rating, movielens_id in movies:

        numerator, denominator = (0, 0)
        # Find neighbours that evaluated this movie
        _all_users = _all_ratings[_all_ratings['movielensID'] == movielens_id]
        for key, value in _all_users.iterrows():

            user_id = int(value['userID'])

            try:
                sim_positive = cosine_similarity(_user_profiles.loc[user_id]['relevant_centroid'],
                                                 _user_relevant_centroid)[0][0]
                sim_negative = cosine_similarity(_user_profiles.loc[user_id]['irrelevant_centroid'],
                                                 _user_irrelevant_centroid)[0][0]
                avg_sim = (sim_positive + sim_negative) / 2

                neighbour_user_rating = value['rating']
                if avg_sim > 0:
                    numerator += avg_sim * (neighbour_user_rating - _user_profiles.loc[user_id]['avg'])
                    # denominator += abs(sim)
                    denominator += avg_sim

                p_ui = _user_average + numerator / denominator

            except (ValueError, ZeroDivisionError):
                p_ui = 0

            predictions.append((movielens_id, p_ui, trailer_id))

    return sort_desc(predictions)

