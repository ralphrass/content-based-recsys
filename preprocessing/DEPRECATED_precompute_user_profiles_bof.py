import pandas as pd
import sqlite3
from utils.opening_feat import load_features, save_obj
from utils.utils import extract_features

# _movies_bof = load_features('content/bof_128.bin')
# print _movies_bof[9089]
_movies_bof_normalized = extract_features('content/bof_128.bin')
# print _movies_bof_normalized[9089]

conn = sqlite3.connect('content/database.db')

_user_ratings = pd.read_sql("SELECT r.userid, t.id "
                            "FROM movielens_rating r "
                            "JOIN movielens_movie m ON m.movielensid = r.movielensid "
                            "JOIN trailers t ON t.imdbid = m.imdbidtt "
                            "AND t.best_file = 1 "
                            "WHERE r.rating > 4 "
                            "AND r.userid < 5000 "
                            "ORDER BY r.userid", conn)
users_bof = {}
_users = _user_ratings['userID'].unique()

for user in _users:
    users_bof[user] = []

    _current_user_ratings = _user_ratings[_user_ratings['userID'] == user]

    for key, item in _current_user_ratings.iterrows():
        item_bof = _movies_bof_normalized[item['id']]
        users_bof[user].append(item_bof)

save_obj(users_bof, '3112_users_bof')
