from utils.utils import evaluate_average
from sklearn.metrics import mean_absolute_error


def evaluate(user_profiles, _n, feature_vector_name, sim_matrix):

    sum_recall, sum_precision, sum_false_positive_rate, sum_diversity, sum_mae = 0, 0, 0, 0, 0

    # print feature_vector_name

    for user, profile in user_profiles.iteritems():

        relevant_set = profile['datasets']['relevant_movies']
        # irrelevant_set = profile['datasets']['irrelevant_movies']
        full_prediction_set = profile['predictions'][feature_vector_name]
        #
        # print "Relevant Set", relevant_set
        # print "Predicted Set", full_prediction_set
        # print "N is", N
        #
        # exit()

        topN = [x[0] for x in full_prediction_set[:_n]]  # topN list composed by movies IDs
        # rec_set = [x[0] for x in full_prediction_set[:_n]]  # topN list composed by trailers IDs

        # how many items of the relevant set are retrieved (top-N)?
        true_positives = float(sum([1 if movie[0] in topN else 0 for movie in relevant_set]))
        # true_negatives = float(sum([1 if movie[2] not in topN else 0 for movie in irrelevant_set]))

        false_negatives = float(len(relevant_set) - true_positives)
        # false_positives = float(len(irrelevant_set) - true_negatives)

        if _n > 1:  # and sim_matrix is not None:  # content-based filtering
            try:
                diversity = sum([sim_matrix[i][j] for i in topN for j in topN if i != j]) / 2
            except KeyError:
                diversity = 0
            sum_diversity += diversity
        # else:  # collaborative filtering

        try:
            precision = true_positives / float(_n)
        except ZeroDivisionError:
            precision = 0

        try:
            recall = true_positives / (true_positives + false_negatives)
        except ZeroDivisionError:
            # print "failed to evaluate", true_positives, "TP,", false_negatives, "FN, for user", user
            recall = 0

        sum_precision += precision
        sum_recall += recall

        try:
            real_ratings = [movie[1] for movie in relevant_set]
            predicted_ratings = [movie[1] for movie in full_prediction_set if movie[0] in [real_movie[0] for real_movie
                                                                                           in relevant_set]]
            mae = mean_absolute_error(real_ratings, predicted_ratings)
            sum_mae += mae
        except ValueError:
            pass

    size = len(user_profiles)
    return evaluate_average(sum_precision, size), evaluate_average(sum_recall, size), evaluate_average(
        sum_diversity, size), evaluate_average(sum_mae, size)


def evaluate_precision_recall(relevant_set, full_prediction_set, _n):
    topN = [x[0] for x in full_prediction_set[:_n]]  # topN list composed by movies IDs
    true_positives = float(sum([1 if movie[0] in topN else 0 for movie in relevant_set]))
    false_negatives = float(len(relevant_set) - true_positives)
    precision = true_positives / float(_n)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall


def evaluate_diversity(sim_matrix, _n, topN):
    if _n > 1:  # and sim_matrix is not None:  # content-based filtering
        diversity = sum([sim_matrix[i][j] for i in topN for j in topN if i != j]) / 2
    return diversity


def evaluate_mae(relevant_set, full_prediction_set):
    real_ratings = [movie[1] for movie in relevant_set]
    predicted_ratings = [movie[1] for movie in full_prediction_set if movie[0] in [real_movie[0] for real_movie
                                                                                   in relevant_set]]
    mae = mean_absolute_error(real_ratings, predicted_ratings)
    return mae