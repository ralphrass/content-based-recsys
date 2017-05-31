import pandas as pd
import sqlite3
from sklearn.feature_extraction.text import TfidfTransformer
from utils.opening_feat import save_obj

transformer = TfidfTransformer(smooth_idf=False)

_safe_exit = 10
count = 0

conn = sqlite3.connect('/home/ralph/Dev/content-based-recsys/content/database.db')

_all_ratings = pd.read_sql('select distinct t.id '
                           'from movielens_rating r '
                           'join movielens_movie m on m.movielensid = r.movielensid '
                           'join trailers t on t.imdbid = m.imdbidtt '
                           'where t.best_file = 1 '
                           # 'and userid < 5000 '
                           'order by t.id', conn)

_all_tags = pd.read_sql('SELECT distinct tag '
                        'FROM movielens_tag', conn)

sql_count_tags = 'SELECT COUNT(*) ' \
                 'FROM movielens_tag t ' \
                 'JOIN movielens_movie m ON m.movielensid = t.movielensid ' \
                 'JOIN trailers tr ON tr.imdbid = m.imdbidtt ' \
                 'WHERE tr.id = ? AND t.tag = ? '

movies_tag_vectors = []

for key, movie in _all_ratings.iterrows():

    movie_tag_vector = []
    print movie[0]

    for subkey, tag in _all_tags.iterrows():

        c = conn.cursor()
        count_tag = c.execute(sql_count_tags, (movie[0], tag[0],))
        movie_count_tags = count_tag.fetchall()
        movie_tag_vector.append(movie_count_tags[0][0])

    movies_tag_vectors.append(movie_tag_vector)

    # print movie_tags

    # count += 1
    # if count == _safe_exit:
    #     break

# for movie_counts in movies_tag_vectors:
#     print sum(movie_counts)
tfidf = transformer.fit_transform(movies_tag_vectors)
# print tfidf.toarray()

save_obj(tfidf.toarray(), 'movies_tfidf_array')