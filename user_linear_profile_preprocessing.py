import sqlite3
import time
import pandas as pd
import numpy as np
from utils.utils import extract_features
from utils.opening_feat import save_obj

_vector_size = 128
_lambda = 0.5
_alpha = 1 * 10**-3
_stop_condition = 0.1
_safe_exit = 100
count = 0

_deep_features_bof = extract_features('content/bof_128.bin')

# print _deep_features_bof
# exit()

start = time.time()
conn = sqlite3.connect('content/database.db')
_all_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
                           'join movielens_movie m on m.movielensid = r.movielensid '
                           'join trailers t on t.imdbid = m.imdbidtt '
                           'where userid < 5000 '
                           'order by userid, t.id', conn)
_full_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
                           'join movielens_movie m on m.movielensid = r.movielensid '
                           'join trailers t on t.imdbid = m.imdbidtt '
                           'order by userid, t.id', conn)
conn.close()
print time.time() - start, "seconds"

users = _all_ratings['userID'].unique()
# movies = _all_ratings['id'].unique()
theta_vectors = {}

for user in users:

    cost = 0
    old_cost = 10000
    # theta_vectors[user] = np.random.randn(_vector_size+1)
    theta_vectors[user] = np.random.normal(0, 0.1, _vector_size+1)
    # print "original", theta_vectors[user]
    user_movies = _all_ratings[_all_ratings['userID'] == user]['id']

    while abs(cost - old_cost) > _stop_condition:

        count += 1
        if count == _safe_exit:
            "safe exit breaking"
            break

        old_cost = cost
        cost = 0
        # print old_cost, "old cost"

        for movie in user_movies:

            try:
                new_movie_vector = np.insert(_deep_features_bof[movie], 0, 1)
            except KeyError:
                continue

            rating = _all_ratings[(_all_ratings['userID'] == user) & (_all_ratings['id'] == movie)]['rating'].iloc[0]
            part1 = (theta_vectors[user].reshape(-1, 129).dot(new_movie_vector.reshape(129, -1))[0][0] - rating)**2

            cost += part1 + (_lambda/2 * sum([t**2 for t in theta_vectors[user]]))

        # print cost, "cost"

        for movie in user_movies:

            try:
                new_movie_vector = np.insert(_deep_features_bof[movie], 0, 1)
            except KeyError:
                continue

            rating = _all_ratings[(_all_ratings['userID'] == user) & (_all_ratings['id'] == movie)]['rating'].iloc[0]

            theta_vectors[user][0] -= _alpha * (theta_vectors[user].reshape(-1, 129).dot(
                new_movie_vector.reshape(129, -1))[0][0] - rating) * new_movie_vector[0]

            # for every theta (weight) value
            for index in range(1, len(theta_vectors[user])):

                part1 = (theta_vectors[user].reshape(-1, 129).dot(new_movie_vector.reshape(129, -1))[0][0] - rating)
                theta_vectors[user][index] -= _alpha * (part1 * new_movie_vector[index] +
                                                        _lambda * theta_vectors[user][index])

    print "user", user
    # break

# print "modified", theta_vectors[user]

save_obj(theta_vectors, 'users_theta_vectors')
