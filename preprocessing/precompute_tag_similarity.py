import pandas as pd
import sqlite3
# import numpy as np
from utils.opening_feat import load_features, save_obj
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import sort_desc

# tfidf_sims = load_features('../content/trailer_tfidf_similarities.pkl')
# print tfidf_sims[5382]
# exit()

conn = sqlite3.connect('/home/ralph/Dev/content-based-recsys/content/database.db')
_all_ratings = pd.read_sql('select distinct t.id '
                           'from movielens_rating r '
                           'join movielens_movie m on m.movielensid = r.movielensid '
                           'join trailers t on t.imdbid = m.imdbidtt '
                           'where t.best_file = 1 '
                           # 'and userid < 5000 '
                           'order by t.id', conn)
# index_to_trailer_id = {}

tfidf_array = load_features('movies_tfidf_array.pkl')

# print _all_ratings.iloc[1]
# exit()

count = 0
_safe_exit = 10

trailer_tfidf_similarities = dict()

for i in range(0, len(tfidf_array)):
    # print sum(tfidf_array[i])
    trailer_id = _all_ratings.iloc[i]
    print trailer_id
    trailer_tfidf_similarities[trailer_id[0]] = {}

    for j in range(0, len(tfidf_array)):

        # if i == j:  # avoid self-comparison
        #    continue

        sim = cosine_similarity([tfidf_array[i]], [tfidf_array[j]])
        # trailer_tfidf_similarities[trailer_id[0]].append((_all_ratings.iloc[j][0], sim[0][0]))
        trailer_tfidf_similarities[trailer_id[0]][_all_ratings.iloc[j][0]] = sim[0][0]

    # trailer_tfidf_similarities[trailer_id[0]] = sort_desc(trailer_tfidf_similarities[trailer_id[0]])

    # count += 1
    # if count == _safe_exit:
    #     break

# print trailer_tfidf_similarities
save_obj(trailer_tfidf_similarities, 'trailer_tfidf_similarities')
