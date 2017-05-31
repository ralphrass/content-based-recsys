import time
import recommender
import evaluation
from utils.opening_feat import load_features, save_obj

start = time.time()

# 85040 is the full set size (4252 is 20 iterations)
# users = select_random_users(conn, 100 * batch, 100)

print "loading user profiles..."
user_profiles = load_features('content/user_profiles_dataframe_3112_users.pkl')
# user_profiles = load_features('content/user_profiles_dataframe_all_users.pkl')
print "user profiles loaded in", time.time() - start, "seconds."

# user_profiles = load_features('content/user_profiles_dataframe_with_user_centroid.pkl')
# user_profiles = user_profiles[:20]
# print "AVG", user_profiles.iloc[7]['avg'], "."
# DEEP_FEATURES_BOF = extract_features('content/bof_128.bin')

# Map every similarity between each movie
convnet_sim_matrix = load_features('content/movie_cosine_similarities_deep.bin')
# print convnet_sim_matrix[4484]
# exit()
low_level_sim_matrix = load_features('content/movie_cosine_similarities_low_level.bin')
# _user_bof_sim_matrix = load_features('content/3112_user_user_bof_similarities.pkl')
_user_user_collaborative_matrix = load_features('content/user_user_collaborative_similarities.pkl')
# _item_item_collaborative_matrix = load_features('content/item_item_collaborative_similarities.pkl')
# _item_item_collaborative_matrix = None
_trailers_tfidf_sims_matrix = load_features('content/trailer_tfidf_similarities.pkl')
_trailers_tfidf_synopsis_sims_matrix = load_features('content/trailer_tfidf_synopsis_similarities.pkl')
# print _trailers_tfidf_synopsis_sims_matrix[5382]
# exit()
start = time.time()

new_user_profiles = recommender.build_user_profile(user_profiles, convnet_sim_matrix, low_level_sim_matrix,
                                                   _user_user_collaborative_matrix, _trailers_tfidf_sims_matrix,
                                                   _trailers_tfidf_synopsis_sims_matrix)
                                                   # _item_item_collaborative_matrix)

print "Profiles built"
print (time.time() - start), "seconds"


def experiment(N, user_profiles, convnet_similarity_matrix, low_level_similarity_matrix,
               tfidf_sim_matrix, synopsis_sim_matrix):

    # DEEP FEATURES - BOF
    dp, dr, dd, dm, drs, df1 = evaluation.evaluate(user_profiles, N, 'deep', convnet_similarity_matrix)
    # Low Level
    # llp, llr, lld, llm = evaluation.evaluate(user_profiles, N, 'low-level', low_level_similarity_matrix)
    llp, llr, lld, llm, llrs, llf1 = evaluation.evaluate(user_profiles, N, 'low-level', low_level_similarity_matrix)
    # TF-IDF
    tfidfp, tfidfr, tfidfd, tfidfm, tfidfrs, tfidff1 = evaluation.evaluate(user_profiles, N, 'tfidf', tfidf_sim_matrix)
    #Synopsis
    syp, syr, syd, sym, syrs, syf1 = evaluation.evaluate(user_profiles, N, 'synopsis', synopsis_sim_matrix)

    # Collaborative Filtering
    cp, cr, cd, cm, crs, cf1 = evaluation.evaluate(user_profiles, N, 'user-collaborative', convnet_similarity_matrix)
    # Collaborative Filtering - Item
    cip, cir, cid, cim, cirs, cif1 = evaluation.evaluate(user_profiles, N, 'item-collaborative', convnet_similarity_matrix)
    # Hybrid
    hp, hr, hd, hm, hrs, hf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid', convnet_similarity_matrix)
    swp, swr, swd, swm, swrs, swf1 = evaluation.evaluate(user_profiles, N, 'switching-hybrid', convnet_similarity_matrix)

    # h2p, h2r, h2d, h2m, h2rs, h2f1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-content-item', convnet_similarity_matrix)
    # h3p, h3r, h3d, h3m, h3rs, h3f1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-collaborative', convnet_similarity_matrix)

    # SVD
    svdp, svdr, svdd, svdm , svdrs, svdf1 = evaluation.evaluate(user_profiles, N, 'svd', convnet_similarity_matrix)

    # lrp, lrr, lrd, lrm = evaluation.evaluate(user_profiles, N, 'linear-regression', convnet_similarity_matrix)

    return {
            'deep': {'precision': dp, 'recall': dr, 'diversity': dd, 'mae': dm, 'rankscore': drs, 'f1': df1},
            'low-level': {'precision': llp, 'recall': llr, 'diversity': lld, 'mae': llm, 'rankscore': llrs, 'f1': llf1},
            'tfidf': {'precision': tfidfp, 'recall': tfidfr, 'diversity': tfidfd, 'mae': tfidfm, 'rankscore': tfidfrs, 'f1': tfidff1},
            'synopsis': {'precision': syp, 'recall': syr, 'diversity': syd, 'mae': sym, 'rankscore': syrs, 'f1': syf1},
            'user-collaborative': {'precision': cp, 'recall': cr, 'diversity': cd, 'mae': cm, 'rankscore': crs, 'f1': cf1},
            'item-collaborative': {'precision': cip, 'recall': cir, 'diversity': cid, 'mae': cim, 'rankscore': cirs, 'f1': cif1},
            'weighted-hybrid': {'precision': hp, 'recall': hr, 'diversity': hd, 'mae': hm, 'rankscore': hrs, 'f1': hf1},
            'switching-hybrid': {'precision': swp, 'recall': swr, 'diversity': swd, 'mae': swm, 'rankscore': swrs, 'f1': swf1},
            'svd': {'precision': svdp, 'recall': svdr, 'diversity': svdd, 'mae': svdm, 'rankscore': svdrs, 'f1': svdf1},

            # 'linear-regression': {'precision': lrp, 'recall': lrr, 'diversity': lrd, 'mae': lrm}
            # 'weighted-hybrid-content-item': {'precision': h2p, 'recall': h2r, 'diversity': h2d, 'mae': h2m, 'rankscore': h2rs, 'f1': h2f1},
            # 'weighted-hybrid-collaborative': {'precision': h3p, 'recall': h3r, 'diversity': h3d, 'mae': h3m, 'rankscore': h3rs, 'f1': h3f1},
            }

results = {}

for index in range(2, 16):
    results[index] = experiment(index, new_user_profiles, convnet_sim_matrix, low_level_sim_matrix,
                                _trailers_tfidf_sims_matrix, _trailers_tfidf_synopsis_sims_matrix)

# print results
print results[15]

save_obj(new_user_profiles, 'profiles_with_predictions')
save_obj(results, 'results_1_users')
end = time.time()
print "Execution time", (end - start), "seconds."
