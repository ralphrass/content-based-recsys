import sqlite3
import time
import pandas as pd
import numpy as np
from sklearn import preprocessing

start = time.time()
conn = sqlite3.connect('content/database.db')
_3112_user_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
                           'join movielens_movie m on m.movielensid = r.movielensid '
                           'join trailers t on t.imdbid = m.imdbidtt '
                           'where userid < 5000 '
                           'order by userid, t.id', conn)

# _3112_user_ratings = pd.read_sql('select userID, t.id, COALESCE(rating, 0) '
#                                  'from movielens_movie m '
#                                  'join trailers t on t.imdbid = m.imdbidtt '
#                                  'left outer join movielens_rating r on m.movielensid = r.movielensid where userid < 5000 '
#                                  'order by userid, t.id', conn)
#
# print _3112_user_ratings.T
# exit()

# _full_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
#                            'join movielens_movie m on m.movielensid = r.movielensid '
#                            'join trailers t on t.imdbid = m.imdbidtt '
#                            'order by userid, t.id', conn)
conn.close()
print time.time() - start, "seconds"
# print _all_ratings.head()

users = _3112_user_ratings['userID'].unique()
users.sort()
movies = _3112_user_ratings['id'].unique()
movies.sort()

full_ratings = []

for user in users:

    print user
    user_ratings = []

    for movie in movies:
        rating = _3112_user_ratings[(_3112_user_ratings['userID'] == user) &
                                    (_3112_user_ratings['id'] == movie)]['rating']

        if rating.empty:
            # rating = _3112_user_ratings[_3112_user_ratings['id'] == movie]['rating'].mean()
            rating = 0
            user_ratings.append(rating)
        else:
            user_ratings.append(rating.iloc[0])

    full_ratings.append(user_ratings)

# print full_ratings
# exit()
scaled = preprocessing.scale(np.matrix(full_ratings))

# print np.matrix(full_ratings)
# np.savetxt('full_matrix_for_svd_item_mean_imputation', np.matrix(full_ratings))
np.savetxt('full_matrix_for_svd_zero_mean_imputation', scaled)
