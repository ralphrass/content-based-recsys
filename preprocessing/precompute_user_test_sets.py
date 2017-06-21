import sqlite3
import pandas as pd
import time
import random
import numpy as np
from recommender import read_user_general_baseline
from utils.utils import extract_features

_avg_ratings = 3.51611876907599

start = time.time()

conn = sqlite3.connect('/home/ralph/Dev/content-based-recsys/content/database.db')

# _deep_features_bof = extract_features('content/bof_128.bin')

# sql_users = "SELECT r.userid, t.id, r.rating, mt.avgrating " \
#             "FROM trailers t " \
#             "JOIN movielens_movie m ON m.imdbidtt = t.imdbid " \
#             "JOIN movielens_rating r ON r.movielensid = m.movielensid " \
#             "JOIN movielens_user_trailer mt ON mt.userid = r.userid "\
#             "WHERE t.best_file = 1 " \
#             "AND r.userid < 5000 " \
#             "ORDER BY r.userid, t.id"

sql_users = "SELECT r.userid, t.id, r.rating " \
            "FROM trailers t " \
            "JOIN movielens_movie m ON m.imdbidtt = t.imdbid " \
            "JOIN movielens_rating r ON r.movielensid = m.movielensid " \
            "WHERE t.best_file = 1 " \
            " And userid < 5000 "\
            "ORDER BY r.userid, t.id"

# sql_users = "SELECT r.userid, t.id, r.rating " \
#             "FROM trailers t " \
#             "JOIN movielens_movie m ON m.imdbidtt = t.imdbid " \
#             "JOIN movielens_rating r ON r.movielensid = m.movielensid " \
#             "WHERE t.best_file = 1 " \
#             "ORDER BY r.userid, t.id"

# sql_all_movies = "SELECT t.id, 1 " \
#                  "FROM movies m " \
#                  "JOIN movielens_movie mm ON mm.imdbidtt = m.imdbid " \
#                  "JOIN trailers t ON t.imdbID = m.imdbID AND t.best_file = 1 " \
#                  "WHERE EXISTS (SELECT movielensid FROM movielens_rating r WHERE r.movielensid = mm.movielensid) "
# sql_all_movies = "SELECT DISTINCT t.id, 1 " \
#                  "FROM movies m " \
#                  "JOIN movielens_movie mm ON mm.imdbidtt = m.imdbid " \
#                  "JOIN trailers t ON t.imdbID = m.imdbID AND t.best_file = 1 " \
#                  "JOIN movielens_rating r ON r.movielensid = mm.movielensid " \
#                  "WHERE userid < 5000 " \
#                  "AND t.best_file = 1"

sql_all_movies = "SELECT DISTINCT t.id, 1 " \
                 "FROM movies m " \
                 "JOIN movielens_movie mm ON mm.imdbidtt = m.imdbid " \
                 "JOIN trailers t ON t.imdbID = m.imdbID AND t.best_file = 1 " \
                 "JOIN movielens_rating r ON r.movielensid = mm.movielensid "

# sql_all_movies = "SELECT DISTINCT t.id, 1 " \
#                  "FROM movies m " \
#                  "JOIN movielens_movie mm ON mm.imdbidtt = m.imdbid " \
#                  "JOIN trailers t ON t.imdbID = m.imdbID AND t.best_file = 1 " \
#                  "JOIN movielens_tag mt ON mt.movielensid = mm.movielensid " \
#                  "JOIN movielens_rating r ON r.movielensid = mm.movielensid AND r.userid = mt.userid "

# c = conn.cursor()
# c.execute(sql_users)
# _all_users_with_movies = c.fetchall()
_ratings = pd.read_sql(sql_users, conn)
_users = _ratings['userID'].unique()

# print _ratings
# exit()

c = conn.cursor()
c.execute(sql_all_movies)
_all_movies = c.fetchall()


_general_baseline, _global_average = read_user_general_baseline()
# print _general_baseline, _global_average
# exit()

users_dict = dict()
# users_bof_dict = dict()

count = 1
# current_user = 0
# exit()
# for userid, trailerid, rating, avgrating in _all_users_with_movies:
for user in _users:

    if count % 5000 == 0:
        print count, "user-movies read"
        # break

    user_movies = _ratings[_ratings['userID'] == user]
    user_ratings = _ratings[_ratings['userID'] == user]['rating']
    relevant_set = _ratings[(_ratings['userID'] == user) & (_ratings['rating'] > 4)]
    irrelevant_set = _ratings[(_ratings['userID'] == user) & (_ratings['rating'] < 3)]

    # print irrelevant_set
    # break
    all_movies_ids = [n['id'] for k, n in user_movies.iterrows()]
    unrated_movies = [movie for movie in _all_movies if movie[0] not in all_movies_ids]
    # users_dict[userid]['random_set'] = random.sample(unrated_movies, 100)

    # user_average = user_ratings.sum() / len(user_ratings)
    user_average = _general_baseline[_general_baseline['userID'] == user]['average'].iloc[0]
    b_u = sum([rating - _avg_ratings for rating in user_ratings]) / len(user_ratings)
    users_dict[user] = {'avg': user_average, 'user_baseline': b_u,
                        'relevant_set': [(item['id'], item['rating']) for k, item in relevant_set.iterrows()],
                        'irrelevant_set': [(item['id'], item['rating']) for k, item in irrelevant_set.iterrows()],
                        'all_movies': [(item['id'], item['rating']) for k, item in user_movies.iterrows()],
                        'random_set': random.sample(unrated_movies, 100)}
    count += 1

print "starting transformation..."

# user_movies_df = pd.DataFrame(data=users_dict, columns=['avg', 'user_baseline', 'relevant_set', 'irrelevant_set',
                                                        # 'all_movies', 'random_set'])
user_movies_df = pd.DataFrame.from_dict(data=users_dict, orient='index')
print user_movies_df.columns
print user_movies_df.head()

user_movies_df.to_pickle('user_profiles_dataframe_3112_users.pkl')
# user_movies_df.to_pickle('user_profiles_dataframe_all_users.pkl')

print "it tok", time.time() - start, "seconds"