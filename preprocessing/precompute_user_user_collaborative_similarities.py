import sqlite3
import pandas as pd
import numpy as np
import sys
import math
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import sort_desc
from scipy.stats import pearsonr
from utils.opening_feat import load_features, save_obj

user_profiles = load_features('/home/ralph/Dev/content-based-recsys/content/user_profiles_dataframe_all_users.pkl')
# print user_profiles.loc[3113]
# exit()
# print user_profiles.columns
# exit()

conn = sqlite3.connect('content/database.db')
_all_ratings = pd.read_sql('select userID, t.id, rating from movielens_rating r '
                           'join movielens_movie m on m.movielensid = r.movielensid '
                           'join trailers t on t.imdbid = m.imdbidtt '
                           # 'where userid < 5000 '
                           'order by userid', conn)
conn.close()

users = _all_ratings['userID'].unique()
movies = _all_ratings['id'].unique()

user_user_similarities = {}

for user in users:

    print user, "current user"

    user_user_similarities[user] = []

    target_user_ratings = _all_ratings[_all_ratings['userID'] == user]
    target_user_average = user_profiles.loc[user]['avg']

    for neighbor in users:

        if user == neighbor:
            continue

        try:
            neighbor_average = user_profiles.loc[neighbor]['avg']
        except IndexError as e:
            print e, "neighbor", neighbor, "failed"

        try:
            intersect = pd.merge(_all_ratings[_all_ratings['userID'] == neighbor], target_user_ratings, on='id')

            if len(intersect) < 5:
                sim = 0
            else:
                sim = pearsonr(intersect['rating_x'], intersect['rating_y'])[0]
                # ssim = sum([(item['rating_x'] - neighbor_average) * (item['rating_y'] - target_user_average)
                #             for k, item in intersect.iterrows()]) / (
                #     math.sqrt(sum([(item['rating_x'] - neighbor_average) ** 2 for k, item in intersect.iterrows()])) *
                #     math.sqrt(sum([(item['rating_y'] - target_user_average) ** 2 for k, item in intersect.iterrows()])))
        except ValueError:
            sim = 0

        if not (sim > 0 or sim < 0):
            sim = 0

        user_user_similarities[user].append((neighbor, sim))

    user_user_similarities[user] = sort_desc(user_user_similarities[user])
    # break

save_obj(user_user_similarities, 'user_user_collaborative_similarities')
