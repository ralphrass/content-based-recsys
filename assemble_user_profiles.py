import numpy as np
import time
from utils.opening_feat import load_features, save_obj
from hausdorff import hausdorff
from utils.utils import sort_desc
from sklearn.metrics.pairwise import cosine_similarity

start = time.time()

_user_profiles = load_features('content/user_profiles_dataframe_with_bof.pkl')

print time.time() - start, "seconds"
start = time.time()
# _deep_features_bof = extract_features('content/bof_128.bin')

# print _user_profiles.iloc[0]['relevant_bof']
# print hausdorff(np.array(_user_profiles.iloc[0]['relevant_bof']), np.array(_user_profiles.iloc[1]['relevant_bof']))

user_distances = {}

for user, profile in _user_profiles.iterrows():

    distances = []
    user_bof = np.mean(np.array(profile['relevant_bof']), axis=0).reshape(1, -1)

    for neighbour_user, neighbour_profile in _user_profiles.iterrows():
        if user == neighbour_user:
            continue

        neighbour_user_bof = np.mean(np.array(neighbour_profile['relevant_bof']), axis=0).reshape(1, -1)

        try:
            sim = cosine_similarity(user_bof, neighbour_user_bof)
        except ValueError:
            continue

        distances.append((neighbour_user, sim))

    user_distances[user] = sort_desc(distances, True)
    break

print user_distances
# for user_id, profile in _user_profiles.iterrows():
#     print d
#     break

# user profile is a collection of favourite movies

save_obj(user_distances, 'user_bof_euclidean_hausdorff_similarities')

print time.time() - start, "seconds"