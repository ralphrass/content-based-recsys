import time
import recommender, evaluation
from utils.opening_feat import load_features, save_obj

start = time.time()

# 85040 is the full set size (4252 is 20 iterations)
# users = select_random_users(conn, 100 * batch, 100)

# user_profiles = load_features('content/user_profiles_dataframe.pkl')
user_profiles = load_features('content/user_profiles_dataframe_with_user_centroid.pkl')
# user_profiles = user_profiles[:20]
# print "AVG", user_profiles.iloc[7]['avg'], "."
# DEEP_FEATURES_BOF = extract_features('content/bof_128.bin')

# Map every similarity between each movie
convnet_similarity_matrix = load_features('content/movie_cosine_similarities_deep.bin')

start = time.time()

new_user_profiles = recommender.build_user_profile(user_profiles, convnet_similarity_matrix)

print "Profiles built"
print (time.time() - start), "seconds"


def experiment(N, user_profiles, convnet_similarity_matrix):

    # DEEP FEATURES - BOF
    dp, dr, dd, dm = evaluation.evaluate(user_profiles, N, 'deep', convnet_similarity_matrix)
    # Content Based - User Centroid
    ucp, ucr, ucd, ucm = evaluation.evaluate(user_profiles, N, 'user-centroid', convnet_similarity_matrix)
    # Collaborative Filtering
    cp, cr, cd, cm = evaluation.evaluate(user_profiles, N, 'collaborative', convnet_similarity_matrix)
    # Weighted Hybrid
    whp, whr, whd, whm = evaluation.evaluate(user_profiles, N, 'weighted-hybrid', convnet_similarity_matrix)
    # Mixing Hybrid
    mhp, mhr, mhd, mhm = evaluation.evaluate(user_profiles, N, 'mixing-hybrid', convnet_similarity_matrix)

    return {'deep': {'precision': dp, 'recall': dr, 'diversity': dd, 'mae': dm},
            'user-centroid': {'precision': ucp, 'recall': ucr, 'diversity': ucd, 'mae': ucm},
            'collaborative': {'precision': cp, 'recall': cr, 'diversity': cd, 'mae': cm},
            'weighted-hybrid': {'precision': whp, 'recall': whr, 'diversity': whd, 'mae': whm},
            'mixing-hybrid': {'precision': mhp, 'recall': mhr, 'diversity': mhd, 'mae': mhm},
            }

results = {}

for index in range(2, 16):
    results[index] = experiment(index, new_user_profiles, convnet_similarity_matrix)

print results

save_obj(results, 'results_1_users')
end = time.time()
print "Execution time", (end - start)
