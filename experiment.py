import time
import recommender
import evaluation
from utils.opening_feat import load_features, save_obj

start = time.time()

# 85040 is the full set size (4252 is 20 iterations)
# users = select_random_users(conn, 100 * batch, 100)

print "loading user profiles"
user_profiles = load_features('content/user_profiles_dataframe_3112_users.pkl')
print "user profiles loaded in", time.time() - start, "seconds"
# user_profiles = load_features('content/user_profiles_dataframe_with_user_centroid.pkl')
# user_profiles = user_profiles[:20]
# print "AVG", user_profiles.iloc[7]['avg'], "."
# DEEP_FEATURES_BOF = extract_features('content/bof_128.bin')

# Map every similarity between each movie
convnet_sim_matrix = load_features('content/movie_cosine_similarities_deep.bin')
low_level_sim_matrix = load_features('content/movie_cosine_similarities_low_level.bin')

start = time.time()

new_user_profiles = recommender.build_user_profile(user_profiles, convnet_sim_matrix, low_level_sim_matrix)

print "Profiles built"
print (time.time() - start), "seconds"


def experiment(N, user_profiles, convnet_similarity_matrix, low_level_similarity_matrix):

    # DEEP FEATURES - BOF
    dp, dr, dd, dm = evaluation.evaluate(user_profiles, N, 'deep', convnet_similarity_matrix)
    # Low Level
    llp, llr, lld, llm = evaluation.evaluate(user_profiles, N, 'low-level', low_level_similarity_matrix)

    # Collaborative Filtering
    # cp, cr, cd, cm = evaluation.evaluate(user_profiles, N, 'user-collaborative', convnet_similarity_matrix)
    # Collaborative Filtering - Item
    # cip, cir, cid, cim = evaluation.evaluate(user_profiles, N, 'item-collaborative', convnet_similarity_matrix)
    # Hybrid
    # hp, hr, hd, hm = evaluation.evaluate(user_profiles, N, 'hybrid', convnet_similarity_matrix)

    # SVD
    svdp, svdr, svdd, svdm = evaluation.evaluate(user_profiles, N, 'svd', convnet_similarity_matrix)

    # lrp, lrr, lrd, lrm = evaluation.evaluate(user_profiles, N, 'linear-regression', convnet_similarity_matrix)

    return {
            'deep': {'precision': dp, 'recall': dr, 'diversity': dd, 'mae': dm},
            'low-level': {'precision': llp, 'recall': llr, 'diversity': lld, 'mae': llm},
            # 'user-collaborative': {'precision': cp, 'recall': cr, 'diversity': cd, 'mae': cm},
            # 'item-collaborative': {'precision': cip, 'recall': cir, 'diversity': cid, 'mae': cim},
            # 'hybrid': {'precision': hp, 'recall': hr, 'diversity': hd, 'mae': hm},
            'svd': {'precision': svdp, 'recall': svdr, 'diversity': svdd, 'mae': svdm},
            # 'linear-regression': {'precision': lrp, 'recall': lrr, 'diversity': lrd, 'mae': lrm}
            }

results = {}

for index in range(2, 16):
    results[index] = experiment(index, new_user_profiles, convnet_sim_matrix, low_level_sim_matrix)

print results

save_obj(new_user_profiles, 'profiles_with_predictions')
save_obj(results, 'results_3112_users')
end = time.time()
print "Execution time", (end - start)
