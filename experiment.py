import sqlite3, time
import recommender, evaluation
# import pandas as pd
from multiprocessing import Process, Manager
from utils.utils import extract_features
from utils.opening_feat import load_features, save_obj

start = time.time()

# load random users and feature vectors
conn = sqlite3.connect('content/database.db')

batch = 0
print "batch", batch+1

# 85040 is the full set size (4252 is 20 iterations)
# users = select_random_users(conn, 100 * batch, 100)

user_profiles = load_features('content/user_profiles_dataframe.pkl')
# user_profiles = user_profiles[:20]
# print "AVG", user_profiles.iloc[7]['avg'], "."

DEEP_FEATURES_BOF = extract_features('content/bof_128.bin')

# Map every similarity between each movie
convnet_similarity_matrix = load_features('content/movie_cosine_similarities_deep.bin')

start = time.time()

new_user_profiles = recommender.build_user_profile(user_profiles, convnet_similarity_matrix)

print "Profiles built"
print (time.time() - start), "seconds"


def experiment(N, user_profiles, convnet_similarity_matrix):

    # DEEP FEATURES - BOF
    dp, dr, dd, dm = evaluation.evaluate(user_profiles, N, 'deep', convnet_similarity_matrix)

    # User-User Collaborative Filtering
    cp, cr, cd, cm = evaluation.evaluate(user_profiles, N, 'collaborative', convnet_similarity_matrix)

    return {'deep': {'precision': dp, 'recall': dr, 'diversity': dd, 'mae': dm},
            'collaborative': {'precision': cp, 'recall': cr, 'diversity': cd, 'mae': cm}}

# manager = Manager()
# results = manager.dict()
results = {}

for index in range(1, 6):
    results[index] = experiment(index, new_user_profiles, convnet_similarity_matrix)

# jobs = []
# for index in iterations:
#     results[index] = {}
#     p = Process(target=experiment, args=(results, index, new_user_profiles, convnet_similarity_matrix))
#     jobs.append(p)
#     p.start()
#
# for proc in jobs:
#     proc.join()

print results

save_obj(results, 'results_2k')

end = time.time()
print "Execution time", (end - start)
