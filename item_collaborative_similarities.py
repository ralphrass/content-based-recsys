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
# print _all_ratings.head()
_all_movies = pd.read_sql('SELECT t.id, m.movielensid FROM trailers t '
                          'JOIN movielens_movie m ON m.imdbidtt = t.imdbid WHERE t.best_file = 1 ORDER BY t.id', conn)
conn.close()

# print _all_movies[_all_movies['movielensID'] == 108085].index[0]
# exit()
_safe_exit = 2
count = 0

print "loaded in", time.time() - start, "seconds"
start = time.time()
item_similarities = dict()

for key, value in _all_movies.iterrows():

    target_movie_trailer_id, target_movie_movielens_id = value
    # print target_movie_movielens_id, "is the target movie"
    other_items = _all_movies.iloc[key + 1:]
    similarities = []

    for sub_key, sub_value in other_items.iterrows():

        neighbour_trailer_id, neighbour_movie_id = sub_value
        join_ratings = pd.merge(_all_ratings[_all_ratings['movielensID'] == neighbour_movie_id],
                                _all_ratings[_all_ratings['movielensID'] == target_movie_movielens_id], on='userID')
        sim = 0

        if len(join_ratings) > 0:
            ratings_x = np.array(join_ratings['rating_x'])
            ratings_y = np.array(join_ratings['rating_y'])
            sim = cosine_similarity(ratings_x.reshape(1, -1), ratings_y.reshape(1, -1))[0][0]
        similarities.append((neighbour_movie_id, sim))

    ordered = sort_desc(similarities)[:30]
    item_similarities[target_movie_movielens_id] = ordered

    count += 1
    if count % 100 == 0:
        print count, "movies read"


# print item_similarities
print "finished in", time.time() - start, "seconds"
save_obj(item_similarities, 'item_collaborative_similarity')
