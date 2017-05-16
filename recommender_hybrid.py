from utils.utils import sort_desc


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
