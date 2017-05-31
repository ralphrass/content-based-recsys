import sqlite3
import time
import pandas as pd
import numpy as np
from utils.opening_feat import load_features, save_obj

matrix = load_features('/home/ralph/Dev/content-based-recsys/content/full_matrix_for_svd.pkl')
print type(matrix.as_matrix())
exit()

_movies_sql = 'select DISTINCT t.id from trailers t ' \
              'join movielens_movie m on t.imdbid = m.imdbidtt ' \
              'join movielens_rating r on m.movielensid = r.movielensid ' \
              'where userid < 5000 ' \
              'order by t.id'

start = time.time()
conn = sqlite3.connect('/home/ralph/Dev/content-based-recsys/content/database.db')
_3112_user_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
                           'join movielens_movie m on m.movielensid = r.movielensid '
                           'join trailers t on t.imdbid = m.imdbidtt '
                           'where userid < 5000 '
                           'order by userid, t.id', conn)

c = conn.cursor()
_movies = c.execute(_movies_sql)


def get_user_rating(df, userid, movieid):
    return df[(df['userID'] == userid) & (df['id'] == movieid)]

_nr_unique_users = len(_3112_user_ratings['userID'].unique())
_ratings_matrix = np.full((_nr_unique_users, len(_movies.fetchall())), np.NaN)

c = conn.cursor()
row = 0

for user in _3112_user_ratings['userID'].unique():
    _movies = c.execute(_movies_sql)
    column = 0
    for movie in _movies.fetchall():
        _user_rating = get_user_rating(_3112_user_ratings, user, movie[0])
        if not _user_rating.empty:
            _ratings_matrix[row][column] = _user_rating['rating'].iloc[0]
        column += 1
    row += 1

# print _ratings_matrix[0]
df = pd.DataFrame(_ratings_matrix)
# print df
df.fillna(df.mean(), inplace=True)
# print df
# print _ratings_matrix
# exit()

# Salva a matriz completa considerando a media de cada item nas celulas vazias
save_obj(df, 'full_matrix_for_svd')


# print full_ratings
# exit()
# scaled = preprocessing.scale(np.matrix(full_ratings))
# full_matrix = np.nan_to_num(np.array(full_ratings))
# normalized = preprocessing.normalize(full_matrix, norm='l2')

# print np.matrix(full_ratings)
# np.savetxt('full_matrix_for_svd_item_mean_imputation', np.matrix(full_ratings))
# np.savetxt('full_matrix_for_svd_normalized', normalized)
