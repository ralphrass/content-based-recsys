import pandas as pd
import numpy as np
from utils.opening_feat import load_features
from sklearn.metrics.pairwise import cosine_similarity

user_profiles = load_features('content/user_profiles_dataframe.pkl')

safe_exit = 4
count = 0

user_user_matrix = pd.DataFrame(index=[userid for userid, profile in user_profiles.iterrows()], columns=['neighbours'])
print user_user_matrix.columns
print user_user_matrix.head()
# exit()
for userid, profile in user_profiles.iterrows():

    movies_current_user = profile['all_movies']
    current_user_trailer_ids = [trailer_id for movieid, rating, trailer_id in movies_current_user]
    other_profiles = user_profiles.iloc[userid+1:]

    # print current_user_trailer_ids

    sims = []

    for neighbourid, neighbour_profile in other_profiles.iterrows():

        movies_neighbour_user = neighbour_profile['all_movies']

        _intersection = [trailer_id for m, rating, trailer_id in movies_neighbour_user if trailer_id in current_user_trailer_ids]

        # print _intersection
        # exit()

        if not bool(_intersection):
            continue

        # print movies_intersection

        ratings_current_user = [rating for m, rating, trailer_id in movies_current_user if trailer_id in _intersection]
        ratings_neighbour_user = [rating for m, rating, trailer_id in movies_neighbour_user if trailer_id in _intersection]

        # print ratings_current_user
        # print ratings_neighbour_user
        # exit()

        try:
            sim = cosine_similarity(np.array(ratings_current_user).reshape(1, -1), np.array(ratings_neighbour_user).
                                    reshape(1, -1))[0][0]
        except ValueError:
            sim = 0
        sims.append((neighbourid, sim, neighbour_profile['avg']))

        # print sim
        # exit()

        # if sim[0][0] > 0.99:
        #     print ratings_current_user
        #     print ratings_neighbour_user
        #     print userid, neighbourid

    user_user_matrix.iloc[userid]['neighbours'] = sims

    if count % 2 == 0:
        print count, "users read"

    count += 1

    if count == safe_exit:
        break

print user_user_matrix.head()
print user_user_matrix.columns

user_user_matrix.to_pickle('user_user_similarity_matrix.pkl')

