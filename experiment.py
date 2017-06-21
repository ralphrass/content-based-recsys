import time
import recommender
import evaluation
from utils.opening_feat import load_features, save_obj

start = time.time()

# 85040 is the full set size (4252 is 20 iterations)
# users = select_random_users(conn, 100 * batch, 100)

_item_item_collaborative_matrix = load_features('content/item_item_collaborative_similarities.pkl')

# print _item_item_collaborative_matrix[4484]
# x = [k for k, v in _item_item_collaborative_matrix[4484] if v == (4485, 23.988368963108908)]
# print x
# print _item_item_collaborative_matrix[4484].index((4485, 23.988368963108908))
# print _item_item_collaborative_matrix[4484].index((4486, -40.004855289600997))
# exit()

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


# _item_item_collaborative_matrix = None
_trailers_tfidf_sims_matrix = load_features('content/trailer_tfidf_similarities.pkl')
_trailers_tfidf_synopsis_sims_matrix = load_features('content/trailer_tfidf_synopsis_similarities.pkl')
# print _trailers_tfidf_synopsis_sims_matrix[5382]
# exit()
start = time.time()

new_user_profiles = recommender.build_user_profile(user_profiles, convnet_sim_matrix, low_level_sim_matrix,
                                                   _user_user_collaborative_matrix, _trailers_tfidf_sims_matrix,
                                                   _trailers_tfidf_synopsis_sims_matrix # )
                                                   , _item_item_collaborative_matrix)

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
    hstp, hstr, hstd, hstm, hstrs, hstf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-synopsis-tags',
                                                   convnet_similarity_matrix)
    hsllp, hsllr, hslld, hsllm, hsllrs, hsllf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-synopsis-low-level',
                                                   convnet_similarity_matrix)
    hsicp, hsicr, hsicd, hsicm, hsicrs, hsicf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-synopsis-item-collaborative',
                                                   convnet_similarity_matrix)
    hsdp, hsdr, hsdd, hsdm, hsdrs, hsdf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-synopsis-deep',
                                                   convnet_similarity_matrix)
    hsucp, hsucr, hsucd, hsucm, hsucrs, hsucf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-synopsis-user-collaborative',
                                                   convnet_similarity_matrix)
    htllp, htllr, htlld, htllm, htllrs, htllf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-tags-low-level',
                                                   convnet_similarity_matrix)
    hticp, hticr, hticd, hticm, hticrs, hticf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-tags-item-collaborative',
                                                   convnet_similarity_matrix)
    htdp, htdr, htdd, htdm, htdrs, htdf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-tags-deep',
                                                   convnet_similarity_matrix)
    htucp, htucr, htucd, htucm, htucrs, htucf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-tags-user-collaborative',
                                                   convnet_similarity_matrix)
    hllicp, hllicr, hllicd, hllicm, hllicrs, hllicf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-low-level-item-collaborative',
                                                   convnet_similarity_matrix)
    hlldp, hlldr, hlldd, hlldm, hlldrs, hlldf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-low-level-deep',
                                                   convnet_similarity_matrix)
    hllucp, hllucr, hllucd, hllucm, hllucrs, hllucf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-low-level-user-collaborative',
                                                   convnet_similarity_matrix)
    hicdp, hicdr, hicdd, hicdm, hicdrs, hicdf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-item-collaborative-deep',
                                                   convnet_similarity_matrix)
    hicucp, hicucr, hicucd, hicucm, hicucrs, hicucf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-item-collaborative-user-collaborative',
                                                   convnet_similarity_matrix)
    hducp, hducr, hducd, hducm, hducrs, hducf1 = evaluation.evaluate(user_profiles, N, 'weighted-hybrid-deep-user-collaborative',
                                                   convnet_similarity_matrix)

    # swp, swr, swd, swm, swrs, swf1 = evaluation.evaluate(user_profiles, N, 'switching-hybrid', convnet_similarity_matrix)

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
            'svd': {'precision': svdp, 'recall': svdr, 'diversity': svdd, 'mae': svdm, 'rankscore': svdrs, 'f1': svdf1},
            # Weighted Hybrid: all possible combinations
            'weighted-hybrid-synopsis-tags': {'precision': hstp, 'recall': hstr, 'diversity': hstd, 'mae': hstm, 'rankscore': hstrs, 'f1': hstf1},
            'weighted-hybrid-synopsis-low-level': {'precision': hsllp, 'recall': hsllr, 'diversity': hslld, 'mae': hsllm,
                                                   'rankscore': hsllrs, 'f1': hsllf1},
            'weighted-hybrid-synopsis-deep': {'precision': hsdp, 'recall': hsdr, 'diversity': hsdd, 'mae': hsdm,
                                              'rankscore': hsdrs, 'f1': hsdf1},
            'weighted-hybrid-synopsis-item-collaborative': {'precision': hsicp, 'recall': hsicr, 'diversity': hsicd, 'mae': hsicm,
                                                            'rankscore': hsicrs, 'f1': hsicf1},
            'weighted-hybrid-synopsis-user-collaborative': {'precision': hsucp, 'recall': hsucr, 'diversity': hsucd,
                                                            'mae': hsucm, 'rankscore': hsucrs, 'f1': hsucf1},
            'weighted-hybrid-tags-low-level': {'precision': htllp, 'recall': htllr, 'diversity': htlld,
                                               'mae': htllm, 'rankscore': htllrs, 'f1': htllf1},
            'weighted-hybrid-tags-item-collaborative': {'precision': hticp, 'recall': hticr, 'diversity': hticd,
                                                        'mae': hticm, 'rankscore': hticrs, 'f1': hticf1},
            'weighted-hybrid-tags-deep': {'precision': htdp, 'recall': htdr, 'diversity': htdd,
                                          'mae': htdm, 'rankscore': htdrs, 'f1': htdf1},
            'weighted-hybrid-tags-user-collaborative': {'precision': htucp, 'recall': htucr, 'diversity': htucd,
                                                        'mae': htucm, 'rankscore': htucrs, 'f1': htucf1},
            'weighted-hybrid-low-level-item-collaborative': {'precision': hllicp, 'recall': hllicr, 'diversity': hllicd,
                                                             'mae': hllicm, 'rankscore': hllicrs, 'f1': hllicf1},
            'weighted-hybrid-low-level-deep': {'precision': hlldp, 'recall': hlldr, 'diversity': hlldd,
                                               'mae': hlldm, 'rankscore': hlldrs, 'f1': hlldf1},
            'weighted-hybrid-low-level-user-collaborative': {'precision': hllucp, 'recall': hllucr, 'diversity': hllucd,
                                                             'mae': hllucm, 'rankscore': hllucrs, 'f1': hllucf1},
            'weighted-hybrid-item-collaborative-deep': {'precision': hicdp, 'recall': hicdr, 'diversity': hicdd,
                                                        'mae': hicdm, 'rankscore': hicdrs, 'f1': hicdf1},
            'weighted-hybrid-item-collaborative-user-collaborative': {'precision': hicucp, 'recall': hicucr, 'diversity': hicucd,
                                                                      'mae': hicucm, 'rankscore': hicucrs, 'f1': hicucf1},
            'weighted-hybrid-deep-user-collaborative': {'precision': hducp, 'recall': hducr, 'diversity': hducd,
                                                        'mae': hducm, 'rankscore': hducrs, 'f1': hducf1},
            # 'switching-hybrid': {'precision': swp, 'recall': swr, 'diversity': swd, 'mae': swm, 'rankscore': swrs, 'f1': swf1},

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
save_obj(results, 'results_50_users')
end = time.time()
print "Execution time", (end - start), "seconds."
