import numpy as np
from utils.utils import sort_desc
# from numbapro import vectorize, cuda
#
# @vectorize(['float32(float32, float32, float32)',
#             'float64(float64, float64, float64)'],
#            target='gpu')
def get_predictions_linear_regression(movies_set, _deep_features, _user_theta_vectors, userid):

    # predictions = []
    #
    # for trailer_id, rating in movies_set:
    #
    #     p_ui = _user_theta_vectors[userid].dot(np.insert(_deep_features[trailer_id], 0, 1))
    #     predictions.append((trailer_id, p_ui))

    predictions = [(trailer_id, _user_theta_vectors[userid].dot(np.insert(_deep_features[trailer_id], 0, 1)))
                   for trailer_id, r in movies_set]

    return sort_desc(predictions)
