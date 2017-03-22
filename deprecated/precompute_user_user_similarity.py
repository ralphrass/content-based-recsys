import pandas as pd
import numpy as np
import sqlite3
from utils.opening_feat import load_features, save_obj
from sklearn.metrics.pairwise import cosine_similarity
import time
start = time.time()

user_profiles = load_features('content/user_profiles_dataframe.pkl')

# safe_exit = 1000
count = 1

conn = sqlite3.connect('content/database.db')

sql_all_movies = "SELECT t.id " \
                  "FROM movies m " \
                  "JOIN movielens_movie mm ON mm.imdbidtt = m.imdbid " \
                  "JOIN trailers t ON t.imdbID = m.imdbID AND t.best_file = 1 " \
                  "WHERE EXISTS (SELECT movielensid FROM movielens_rating r WHERE r.movielensid = mm.movielensid) " \
                 "ORDER BY t.id"

c = conn.cursor()
c.execute(sql_all_movies)
_all_movies = c.fetchall()

# print _all_movies
# exit()

# print user_user_matrix.columns
# print user_user_matrix.head()
# exit()
user_dict = {}
users_ids = []

for userid, profile in user_profiles.iterrows():

    movies_current_user = profile['all_movies']
    # print movies_current_user
    # exit()
    # start = time.time()
    current_user_sparse_vector = [rating if tid == trailer_id[0] else 0 for tid, rating, mid in profile['all_movies'] for trailer_id in _all_movies]
    # current_user_sparse_vector = [1 if tid in current_user_trailer_ids else 0 for tid in _all_movies]

    user_dict[userid] = current_user_sparse_vector
    users_ids.append(userid)

    # print current_user_sparse_vector
    # print "tok", time.time() - start, "seconds"
    # exit()

    other_profiles = user_profiles.loc[userid + 1:]

    # print current_user_trailer_ids

    # sims = []

    # for neighbourid, neighbour_profile in other_profiles.iterrows():
    #
    #     movies_neighbour_user = neighbour_profile['all_movies']
    #
    #     ratings_current_user = [rating for m, rating, trailer_id in movies_current_user if trailer_id in
    #                             current_user_trailer_ids]
    #     ratings_neighbour_user = [rating for m, rating, trailer_id in neighbour_profile['all_movies'] if trailer_id in
    #                               current_user_trailer_ids]
    #
    #     try:
    #         sim = cosine_similarity(np.array(ratings_current_user).reshape(1, -1), np.array(ratings_neighbour_user).
    #                                 reshape(1, -1))[0][0]
    #     except ValueError:
    #         sim = 0
    #
    #     sims.append((neighbourid, sim, neighbour_profile['avg']))

        # if sim[0][0] > 0.99:
        #     print ratings_current_user
        #     print ratings_neighbour_user
        #     print userid, neighbourid

    #user_user_matrix.loc[userid]['neighbours'] = sims

    if count % 100 == 0:
        print count, "users read"

    count += 1

    # if count == safe_exit:
    #     break

# print user_user_matrix.head()
# print user_user_matrix.columns

print time.time() - start, "seconds"

# user_ratings_sparse_matrix = pd.DataFrame.from_dict(data=user_dict, orient='index')
# print user_ratings_sparse_matrix.head()
# print user_ratings_sparse_matrix.columns
save_obj(user_dict, 'user_ratings_sparse_matrix')

