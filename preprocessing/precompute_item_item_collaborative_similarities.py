import sqlite3
import pandas as pd
import math
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import sort_desc
from utils.opening_feat import save_obj, load_features

# df = load_features('/home/ralph/Dev/content-based-recsys/item_item_collaborative_similarities.pkl')
# print df
# exit()

user_profiles = load_features('../content/user_profiles_dataframe_all_users.pkl')
# print user_profiles.index.values
# print user_profiles.loc[3858]['avg']
# exit()

conn = sqlite3.connect('/home/ralph/Dev/content-based-recsys/content/database.db')
_all_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
                           'join movielens_movie m on m.movielensid = r.movielensid '
                           'join trailers t on t.imdbid = m.imdbidtt '
                           'where t.best_file = 1 '
                           'and userid < 5000 '
                           'order by t.id', conn)

movies = _all_ratings['id'].unique()

movie_similarity = {}


def adjusted_cosine(intersection, user_profiles):

    numerator, denominator_1, denominator_2 = 0, 0, 0

    for idx, row in intersection.iterrows():

        user_id = row['userID']
        user_avg = user_profiles.loc[user_id]['avg']

        numerator += (row['rating_x'] - user_avg) * (row['rating_y'] - user_avg)
        denominator_1 += (row['rating_x'] - user_avg)**2
        denominator_2 += (row['rating_y'] - user_avg)**2

    sim = numerator / math.sqrt(denominator_1) * math.sqrt(denominator_2)

    return sim


for movie in movies:

    print "current movie", movie
    movie_similarity[movie] = []

    for neighbor in movies:

        if movie == neighbor:
            continue

        intersect = pd.merge(_all_ratings[_all_ratings['id'] == movie],
                             _all_ratings[_all_ratings['id'] == neighbor], on='userID')
        # print intersect
        # exit()

        # if len(intersect) > 4:
        if not intersect.empty:

            try:
                # sim = cosine_similarity(intersect['rating_x'].reshape(1, -1), intersect['rating_y'].reshape(1, -1))
                # sim = cosine_similarity([intersect['rating_x']], [intersect['rating_y']])

                sim = adjusted_cosine(intersect, user_profiles)
                movie_similarity[movie].append((neighbor, sim))

                # sim = cosine_similarity(intersect['rating_x'].reshape(1, -1), intersect['rating_y'].reshape(1, -1))
                # movie_similarity[movie].append((neighbor, sim[0][0]))
            except ValueError:
                continue
        else:
            movie_similarity[movie].append((neighbor, 0))

        movie_similarity[movie] = sort_desc(movie_similarity[movie])
    # print movie_similarity[movie]
    # break

save_obj(movie_similarity, 'item_item_collaborative_similarities')
