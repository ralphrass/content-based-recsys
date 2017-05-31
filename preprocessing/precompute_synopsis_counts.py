import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.opening_feat import save_obj


v = TfidfVectorizer()

conn = sqlite3.connect('/home/ralph/Dev/content-based-recsys/content/database.db')

_all_movies = pd.read_sql('select distinct t.id, ms.Plot '
                          'from movielens_rating r '
                          'join movielens_movie m on m.movielensid = r.movielensid '
                          'join trailers t on t.imdbid = m.imdbidtt '
                          'join movies ms on ms.imdbID = t.imdbid '
                          'where t.best_file = 1 '
                          # 'and userid < 5000 '
                          'order by t.id ', conn)

plots = []

for key, movie in _all_movies.iterrows():
    # print movie['Plot']
    print key
    plots.append(movie['Plot'])

x = v.fit_transform(plots)

save_obj(x.toarray(), 'movies_tfidf_synopsis_array')
