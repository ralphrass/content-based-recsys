import sqlite3, time
import recommender, evaluation
from multiprocessing import Process
from utils.utils import extract_features, select_random_users
from utils.opening_feat import load_features

iterations = range(1, 11)

start = time.time()

# load random users and feature vectors
conn = sqlite3.connect('content/database.db')

batch = 0
print "batch", batch+1

# 85040 is the full set size (4252 is 20 iterations)
users = select_random_users(conn, 42 * batch, 42)

DEEP_FEATURES_BOF = extract_features('content/bof_128.bin')

print len(users)

# Map every similarity between each movie
convnet_similarity_matrix = load_features('content/movie_cosine_similarities_deep.bin')

start = time.time()

user_profiles = recommender.build_user_profile(users, convnet_similarity_matrix)

print time.time()
print "Profiles buit"
print (time.time() - start), "seconds"


def experiment(N, user_profiles, convnet_similarity_matrix):
    # global LOW_LEVEL_FEATURES, DEEP_FEATURES_BOF, HYBRID_FEATURES_BOF, _ratings, _global_average, _ratings_by_movie

    # Tag-based
    # p_t, r_t = run(user_profiles, N, USER_TFIDF_FEATURES, MOVIE_TFIDF_FEATURES)
    # print "Tag-based Recall", r_t, "Tag-based Precision", p_t, "For iteration with", N

    # DEEP FEATURES - BOF
    # p_d, r_d, m_d, s_d = recommender.run(user_profiles, N, DEEP_FEATURES_BOF, 'deep')
    p, r, d, m = evaluation.evaluate(user_profiles, N, 'deep', convnet_similarity_matrix)
    print "Deep BOF Recall", r, "Deep BOF Precision", p, "Deep Diversity", d, "Deep MAE", m, "For iteration with", N

    # p, r, m, s = recommender.run(user_profiles, N, None, 'random')
    p, r, d, m = evaluation.evaluate(user_profiles, N, 'random', None)
    print "Random Recall", r, "Random Precision", p, "Random Diversity", d, "Random MAE", m, "For iteration with", N

    # if N == 1:
    #     print "LL MAE", m_l
    #     print "Deep MAE", m_d
    #     print "Hybrid MAE", m_h
    #     print "Random MAE", m


jobs = []
for index in iterations:
    p = Process(target=experiment, args=(index, user_profiles, convnet_similarity_matrix))
    jobs.append(p)
    p.start()

for proc in jobs:
    proc.join()

end = time.time()
print "Execution time", (end - start)
