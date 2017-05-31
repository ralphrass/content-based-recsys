import pandas as pd
from utils.utils import sort_desc
from sklearn.metrics.pairwise import cosine_similarity


def get_weighted_hybrid_recommendations(relevant_centroid, irrelevant_centroid, predictions_content_based_user_centroid, predictions_item_collaborative):

    _base_weight = 0.25
    _max_weight = 1
    _threshold = 0

    weight_content_based = _base_weight * (len(relevant_centroid) > _threshold) + \
                           _base_weight * (len(irrelevant_centroid) > _threshold)

    weight_collaborative_based = _max_weight - weight_content_based

    # set weight to content-based filtering: [0, 0.25, 0.5]. collaborative filtering is the complement
    predictions_weighted_hybrid = [(item[0][0], (item[0][1] * weight_content_based +
                                                item[1][1] * weight_collaborative_based))
                                   for item in zip(predictions_content_based_user_centroid,
                                                  predictions_item_collaborative)]

    return predictions_weighted_hybrid


def get_mixing_hybrid_recommendations(predictions_item_collaborative, predictions_content_based_user_centroid):

    predictions_mixing_hybrid = predictions_item_collaborative[:5]
    cp_mixing_hybrid = predictions_mixing_hybrid[:]

    for item in predictions_content_based_user_centroid:

        if len(predictions_mixing_hybrid) == 10:
            break

        if item[0] not in [x[0] for x in cp_mixing_hybrid]:
            predictions_mixing_hybrid.append(item)

    return predictions_mixing_hybrid


def get_weighted_hybrid_recommendations(predictions, movie_set):

    hybrid_predictions = []
    _num_vectors = 2

    for trailer_id, ratings in movie_set:

        sum_ratings = sum([p_ui for tid, p_ui in predictions if tid == trailer_id])
        hybrid_predictions.append((trailer_id, sum_ratings / _num_vectors))

    return sort_desc(hybrid_predictions)


def get_switching_hybrid_recommendations(movies_to_predict, _all_ratings, _target_user_id, sim_matrix):

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
                top_neighbours.append((rated_movie, sim[0][0]))
            except ValueError:
                try:
                    sim = sim_matrix[rated_movie][trailer_id]
                    top_neighbours.append((rated_movie, sim))
                except KeyError:
                    continue

        top_n = sort_desc(top_neighbours)[:_limit_top_neighbours_to]

        numerator, denominator = (0, 0)
        for neighbour, sim in top_n:
            user_rating = _all_ratings[(_all_ratings['id'] == neighbour) & (_all_ratings['userID'] == _target_user_id)][
                'rating'].iloc[0]
            numerator += sim * user_rating
            denominator += abs(sim)

        try:
            p_ui = numerator / denominator
        except ZeroDivisionError:
            p_ui = 0

        predictions.append((trailer_id, p_ui))

    return sort_desc(predictions)