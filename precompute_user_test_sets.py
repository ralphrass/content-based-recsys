import sqlite3
import pandas as pd
import time
import random
import numpy as np
from recommender import read_user_general_baseline
from utils.utils import extract_features

start = time.time()

conn = sqlite3.connect('content/database.db')

_deep_features_bof = extract_features('content/bof_128.bin')

sql_users = "SELECT r.userid, t.id, r.rating, m.movielensid, mt.avgrating " \
            "FROM trailers t " \
            "JOIN movielens_movie m ON m.imdbidtt = t.imdbid " \
            "JOIN movielens_rating r ON r.movielensid = m.movielensid " \
            "JOIN movielens_user_trailer mt ON mt.userid = r.userid "\
            "WHERE t.best_file = 1 " \
            "ORDER BY r.userid, t.id"

sql_all_movies = "SELECT t.id, 1, mm.movielensId " \
                  "FROM movies m " \
                  "JOIN movielens_movie mm ON mm.imdbidtt = m.imdbid " \
                  "JOIN trailers t ON t.imdbID = m.imdbID AND t.best_file = 1 " \
                  "WHERE EXISTS (SELECT movielensid FROM movielens_rating r WHERE r.movielensid = mm.movielensid) "

c = conn.cursor()
c.execute(sql_users)
_all_users_with_movies = c.fetchall()

c = conn.cursor()
c.execute(sql_all_movies)
_all_movies = c.fetchall()


_general_baseline, _global_average = read_user_general_baseline()

users_dict = dict()

count = 1
# current_user = 0
# exit()
for userid, trailerid, rating, movielensid, avgrating in _all_users_with_movies:

    if count % 50000 == 0:
        print count, "user-movies read"
        # break

    if userid not in users_dict:
        try:
            user_baseline = _general_baseline[_general_baseline['userID'] == userid]['average'] - _global_average
            users_dict[userid] = {'avg': avgrating, 'user_baseline': user_baseline, 'relevant_set': [],
                                  'irrelevant_set': [], 'all_movies': [], 'random_set': [],
                                  'relevant_bof': [], 'irrelevant_bof': []}
        except:
            print userid, "failed"

    if rating >= 4:
        users_dict[userid]['relevant_set'].append((trailerid, rating, movielensid))
        users_dict[userid]['relevant_bof'].append(_deep_features_bof[trailerid])
    elif rating < 3:
        users_dict[userid]['irrelevant_set'].append((trailerid, rating, movielensid))
        users_dict[userid]['irrelevant_bof'].append(_deep_features_bof[trailerid])

    users_dict[userid]['all_movies'].append((trailerid, rating, movielensid))

    count += 1

count = 0

for userid, profile in users_dict.iteritems():

    count += 1
    if count % 10000 == 0:
        print count, "random samples generated"

    all_movies_ids = [n[0] for n in users_dict[userid]['all_movies']]
    unrated_movies = [movie for movie in _all_movies if movie[0] not in all_movies_ids]
    users_dict[userid]['random_set'] = random.sample(unrated_movies, 100)

    users_dict[userid]['relevant_centroid'] = np.mean(np.array(users_dict[userid]['relevant_bof']), axis=0).reshape(1, -1)
    users_dict[userid]['irrelevant_centroid'] = np.mean(np.array(users_dict[userid]['relevant_bof']), axis=0).reshape(1, -1)

print "starting transformation..."

# user_movies_df = pd.DataFrame(data=users_dict, columns=['avg', 'user_baseline', 'relevant_set', 'irrelevant_set',
                                                        # 'all_movies', 'random_set'])
user_movies_df = pd.DataFrame.from_dict(data=users_dict, orient='index')
print user_movies_df.columns
print user_movies_df.head()

# user_movies_df.to_pickle('user_profiles_dataframe_1k.pkl')
user_movies_df.to_pickle('user_profiles_dataframe_with_user_centroid.pkl')

print "it tok", time.time() - start, "seconds"