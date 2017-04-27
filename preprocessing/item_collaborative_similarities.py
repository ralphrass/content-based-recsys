import sqlite3
import pandas as pd
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import sort_desc
from utils.opening_feat import save_obj

start = time.time()

conn = sqlite3.connect('content/database.db')
_all_ratings = pd.read_sql('select userID, movielensID, rating from movielens_rating order by userid', conn)
_all_ratings['movielensID'] = _all_ratings['movielensID'].astype('uint32')
_all_ratings['userID'] = _all_ratings['userID'].astype('uint32')
_all_ratings['rating'] = _all_ratings['rating'].astype('float16')
# print _all_ratings[_all_ratings['movielensID'] == 16].head()
_all_movies = pd.read_sql('SELECT m.movielensid FROM trailers t '
                          'JOIN movielens_movie m ON m.imdbidtt = t.imdbid WHERE t.best_file = 1 ORDER BY t.id', conn)
_all_movies['movielensID'] = _all_movies['movielensID'].astype('uint16')
conn.close()

# print _all_movies[_all_movies['movielensID'] == 108085].index[0]
# exit()
_safe_exit = 2
count = 0

print "loaded in", time.time() - start, "seconds"
start = time.time()
item_similarities = dict()

for key, target_movie_movielens_id in _all_movies.iterrows():

    other_items = _all_movies.iloc[key + 1:]
    similarities = []

    for sub_key, neighbour_movie_id in other_items.iterrows():

        join_ratings = pd.merge(_all_ratings[_all_ratings['movielensID'] == neighbour_movie_id.iloc[0]],
                                _all_ratings[_all_ratings['movielensID'] == target_movie_movielens_id.iloc[0]], on='userID')
        sim = 0

        if len(join_ratings) > 0:
            ratings_x = np.array(join_ratings['rating_x'])
            ratings_y = np.array(join_ratings['rating_y'])
            sim = cosine_similarity(ratings_x.reshape(1, -1), ratings_y.reshape(1, -1))[0][0]
        similarities.append((neighbour_movie_id, sim))

    ordered = sort_desc(similarities)[:30]
    item_similarities[target_movie_movielens_id.iloc[0]] = ordered

    count += 1
    if count % 100 == 0:
        print count, "movies read"
    break

# print item_similarities
print "finished in", time.time() - start, "seconds"
save_obj(item_similarities, 'item_collaborative_similarity')
