# Simplicity is the final achievement. After one has played a vast quantity of notes and more notes, it is
# simplicity that emerges as the crowning reward of art. Frederic Chopin.

import random, operator, sqlite3
from utils.utils import get_user_baseline, get_item_baseline, evaluate_average, get_user_training_test_movies, \
    get_random_movie_set
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Manager

_avg_ratings = 3.51611876907599
_std_deviation = 1.098732183


def cosine(movieI, movieJ, feature_vector):

    traileri = movieI[0]
    trailerj = movieJ[0]

    try:
        featuresI = feature_vector[traileri]
        featuresJ = feature_vector[trailerj]
    except KeyError:
        return 0

    return cosine_similarity([featuresI], [featuresJ])


def predict_user_rating(user_baseline, movieid, all_similarities, _ratings_by_movie, _global_average):

    global _avg_ratings

    item_baseline = get_item_baseline(user_baseline, movieid, _ratings_by_movie, _global_average)
    user_item_baseline = (_avg_ratings + user_baseline + item_baseline)

    numerator = sum((rating - user_item_baseline) * sim if sim > 0 else 0 for rating, sim in all_similarities)
    denominator = reduce(operator.add, [abs(x[1]) for x in all_similarities])
    try:
        prediction = (numerator / denominator) + user_item_baseline
    except ZeroDivisionError:
        prediction = 0

    return prediction


def get_predictions(user_baseline, movies, all_movies, sim_matrix, _ratings_by_movie, _global_average):

    predictions = [(movie[2], predict_user_rating(user_baseline, movie[2],
                                                  [(movieJ[1], sim_matrix[movieJ[0]][movie[0]])
                                                   for movieJ in all_movies], _ratings_by_movie, _global_average),
                    movie[0])
                   for movie in movies]

    return sort_desc(predictions)


def get_random_predictions(movies):

    global _avg_ratings, _std_deviation

    # predictions = (np.random.uniform(low=0.5, high=5.0, size=len(movies))).tolist()
    random_movies = [(movie[2], random.uniform(0.5, 5), movie[0]) for movie in movies]
    random_movies = sort_desc(random_movies)

    return random_movies


def sort_desc(list_to_sort):

    list_to_sort.sort(key=operator.itemgetter(1), reverse=True)
    return list_to_sort


def evaluate(user_profiles, N, feature_vector_name):

    sum_recall, sum_precision, sum_false_positive_rate = 0, 0, 0
    sum_diversity = 0

    for user, profile in user_profiles.iteritems():

        relevant_set = profile['datasets']['relevant_movies']
        irrelevant_set = profile['datasets']['irrelevant_movies']

        full_prediction_set = profile['predictions'][feature_vector_name]
        # full_set.sort(reverse=True)
        topN = [x[0] for x in full_prediction_set[:N]]  # topN list composed by movies IDs

        # how many items of the relevant set are retrieved (top-N)?
        true_positives = float(sum([1 if movie[2] in topN else 0 for movie in relevant_set]))
        true_negatives = float(sum([1 if movie[2] not in topN else 0 for movie in irrelevant_set]))

        false_negatives = float(len(relevant_set) - true_positives)
        false_positives = float(len(irrelevant_set) - true_negatives)

        # print "Relevant Set", relevant_set
        # print "Size", len(relevant_set)
        # print "Irrelevant Set", irrelevant_set
        # print "Size", len(irrelevant_set)
        # print "Full Set", full_prediction_set
        # print len(full_prediction_set)
        # print "Feature Vector", feature_vector_name
        # print "True Positives", true_positives
        # print "True Negatives", true_negatives
        # print "False Negatives", false_negatives
        # print "False Positives", false_positives
        # print "Top-N", topN
        # exit()

        try:
            precision = true_positives / (true_positives + false_positives)
        except ZeroDivisionError:
            precision = 0

        recall = true_positives / (true_positives + false_negatives)

        sum_precision += precision
        sum_recall += recall

    size = len(user_profiles)
    return evaluate_average(sum_precision, size), evaluate_average(sum_recall, size)


def read_user_general_baseline():

    conn = sqlite3.connect('content/database.db')

    _ratings = pd.read_sql("SELECT userid, SUM(rating)/COUNT(rating) AS average "
                           "FROM movielens_rating GROUP BY userid ORDER BY userid", conn, columns=['average'],
                           index_col='userID')

    conn.close()

    return _ratings, _ratings['average'].mean()


def read_movie_general_baseline():

    conn = sqlite3.connect('content/database.db')

    _ratings_by_movie = pd.read_sql("SELECT movielensid, SUM(rating)/COUNT(rating) AS average "
                                    "FROM movielens_rating GROUP BY movielensid ORDER BY movielensid", conn,
                                    columns=['average'], index_col='movielensID')

    conn.close()

    return _ratings_by_movie


# Predict ratings
def build_user_profile(users, convnet_similarity_matrix):

    _general_baseline, _global_average = read_user_general_baseline()
    _ratings_by_movie = read_movie_general_baseline()

    user_profiles = {}

    for user in users:

        user_baseline = get_user_baseline(user[0], _general_baseline, _global_average)
        user_movies_test, all_movies, garbage_test_set = get_user_training_test_movies(user[0])

        if len(user_movies_test) == 0:
            continue

        random_movies = get_random_movie_set(user[0])
        movies_set = user_movies_test + garbage_test_set + random_movies

        predictions = get_predictions(user_baseline, movies_set, all_movies, convnet_similarity_matrix,
                                      _ratings_by_movie, _global_average)

        predictions_random = get_random_predictions(movies_set)

        user_profiles[user[0]] = {'datasets': {'relevant_movies': user_movies_test,
                                               'irrelevant_movies': garbage_test_set},
                                  'userid': user[0],
                                  'predictions': {'deep': predictions,
                                                  'random': predictions_random}}
    return user_profiles
