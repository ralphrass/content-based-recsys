import sqlite3
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import sort_desc
from utils.opening_feat import save_obj, load_features

# df = load_features('/home/ralph/Dev/content-based-recsys/content/item_item_collaborative_similarities.pkl')
# print len(df)
# exit()


conn = sqlite3.connect('/home/ralph/Dev/content-based-recsys/content/database.db')
_all_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
                           'join movielens_movie m on m.movielensid = r.movielensid '
                           'join trailers t on t.imdbid = m.imdbidtt '
                           'where t.best_file = 1 '
                           'order by t.id', conn)

movies = _all_ratings['id'].unique()

movie_similarity = {}

for movie in movies:

    print "current movie", movie
    movie_similarity[movie] = []

    for neighbor in movies:

        if movie == neighbor:
            continue

        intersect = pd.merge(_all_ratings[_all_ratings['id'] == movie],
                             _all_ratings[_all_ratings['id'] == neighbor], on='userID')

        if len(intersect) > 4:
            try:
                sim = cosine_similarity(intersect['rating_x'].reshape(1, -1), intersect['rating_y'].reshape(1, -1))
                movie_similarity[movie].append((neighbor, sim[0][0]))
            except ValueError:
                continue

        movie_similarity[movie] = sort_desc(movie_similarity[movie])
    # print movie_similarity[movie]
    # break

save_obj(movie_similarity, 'item_item_collaborative_similarities')
