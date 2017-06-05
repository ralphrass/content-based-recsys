from utils.opening_feat import load_features
import operator

_trailers_tfidf_sims_matrix = load_features('/home/ralph/Dev/content-based-recsys/content/trailer_tfidf_similarities.pkl')
print _trailers_tfidf_sims_matrix[4484]
print type(_trailers_tfidf_sims_matrix[4484])

# print _trailers_tfidf_sims_matrix[4484]
sorted_x = sorted(_trailers_tfidf_sims_matrix[4484].items(), key=operator.itemgetter(1), reverse=True)
print sorted_x
