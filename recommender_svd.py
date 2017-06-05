import numpy as np
import sqlite3
import pandas as pd
from utils.utils import sort_desc
from utils.opening_feat import load_features

# matrix = np.loadtxt('content/full_matrix_for_svd')
#
# u, s, v = np.linalg.svd(matrix, full_matrices=0)
#
# print matrix.shape
# print u.shape, s.shape, v.shape
# print s
#
# _k = 20
# # all lines, k columns (shape: 3112 x 3473)
# reduced_u = u[:, :_k]
# reduced_s = s[:_k]
# reduced_v = v[:_k, :]
# print reduced_u[100][10]
#
# print reduced_u.shape, reduced_s.shape, reduced_v.shape


def map_movie_to_index():
    movies_to_index = dict()
    conn = sqlite3.connect('content/database.db')

    _movies_sql = 'select DISTINCT t.id from trailers t ' \
                  'join movielens_movie m on t.imdbid = m.imdbidtt ' \
                  'join movielens_rating r on m.movielensid = r.movielensid ' \
                  'where userid < 5000 ' \
                  'order by t.id'
    # that is the same query conditions and ordering that is used to precompute SVD and to build user profiles
    # _all_ratings = pd.read_sql('SELECT r.userid, t.id, r.rating, mt.avgrating ' \
    #                            'FROM trailers t ' \
    #                            'JOIN movielens_movie m ON m.imdbidtt = t.imdbid ' \
    #                            'JOIN movielens_rating r ON r.movielensid = m.movielensid ' \
    #                            'JOIN movielens_user_trailer mt ON mt.userid = r.userid ' \
    #                            'WHERE t.best_file = 1 ' \
    #                            'AND r.userid < 5000 ' \
    #                            'ORDER BY r.userid, t.id', conn)

    _movies = conn.cursor().execute(_movies_sql)
    # movies = _all_ratings['id'].unique()
    # movies.sort()

    idx = 0
    for movie in _movies.fetchall():
        movies_to_index[movie[0]] = idx
        idx += 1

    conn.close()

    return movies_to_index


def load_svd():
    _k = 100
    # matrix = np.loadtxt('content/full_matrix_for_svd.pkl')
    matrix = load_features('content/full_matrix_for_svd.pkl')
    np_matrix = matrix.as_matrix()

    u, s, v = np.linalg.svd(np_matrix)

    reduced_u = u[:, :_k]  # 3112 x _k
    reduced_s = s[:_k]  # _k x 1
    reduced_v = v[:_k, :]  # _k x 3473

    return reduced_u, reduced_s, reduced_v
    # return u, s, v


# movies_set: (trailer_id, rating) tuple
# u: m by k matrix; s: k by 1 matrix; v: k by n matrix
def get_predictions_svd(movies_set, svd_matrix, movies_to_index, user_index, user_average):

    u, s, v = svd_matrix

    predictions = []
    for trailer_id, rating in movies_set:

        try:
            movie_index = movies_to_index[trailer_id]

            p_ui = 0
            for singular_value in range(0, len(s)):
                p_ui += u[user_index][singular_value] * s[singular_value] * v[singular_value][movie_index]
            # p_ui += user_average

            # print p_ui
            # print sum(u[user_index] * s * v[:, movie_index])
            # break
            # p_ui = user_average + np.sum(u[user_index].dot(s.dot(v[:, movie_index])))
        except KeyError:
            p_ui = user_average

        predictions.append((trailer_id, p_ui))

    return sort_desc(predictions)
