import pandas as pd
import numpy as np
from utils.opening_feat import load_features
from sklearn.metrics.pairwise import cosine_similarity

user_profiles = load_features('content/user_profiles_dataframe.pkl')
# user_profiles = user_profiles.iloc[0:5000]

safe_exit = 10
count = 0

user_user_matrix = pd.DataFrame(index=user_profiles['userid'], columns=['neighbours'])

for userid, profile in user_profiles.iterrows():

    other_profiles = user_profiles[user_profiles['userid'] > userid]
    movies_current_user = user_profiles.iloc[userid]['all_movies']
    # print movies_current_user
    current_user_trailer_ids = [trailer_id for movieid, rating, trailer_id in movies_current_user]

    sims = []

    for neighbourid, neighbour_profile in other_profiles.iterrows():

        try:
            movies_neighbour_user = user_profiles.iloc[neighbourid]['all_movies']
        except:
            continue
        # print movies_neighbour_user

        try:
            movies_intersection = [trailer_id for m, rating, trailer_id in movies_neighbour_user if trailer_id in current_user_trailer_ids]
        except:
            continue

        # print movies_intersection

        ratings_current_user = [rating for m, rating, trailer_id in movies_current_user if trailer_id in movies_intersection]
        ratings_neighbour_user = [rating for m, rating, trailer_id in movies_neighbour_user if trailer_id in movies_intersection]

        # print ratings_current_user
        # print ratings_neighbour_user

        try:
            sim = cosine_similarity(np.array(ratings_current_user).reshape(1, -1), np.array(ratings_neighbour_user).reshape(1, -1))[0][0]
        except ValueError:
            sim = 0
        sims.append((neighbourid, sim))

        # print sim

        # if sim[0][0] > 0.99:
        #     print ratings_current_user
        #     print ratings_neighbour_user
        #     print userid, neighbourid

    user_user_matrix.loc[userid]['neighbours'] = sims

    count += 1

    if count == safe_exit:
        break

user_user_matrix.to_pickle('user_user_similarity_matrix.pkl')

