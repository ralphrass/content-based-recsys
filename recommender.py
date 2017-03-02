# Simplicity is the final achievement. After one has played a vast quantity of notes and more notes, it is
# simplicity that emerges as the crowning reward of art. Frederic Chopin.

import random, operator, sqlite3, time
from utils.utils import get_user_baseline, get_item_baseline
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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


# Each movie to predict contains: trailer id, user rating and movielens id
def get_collaborative_predictions(userid, user_average, movies_to_predict, user_profiles, user_user_similarity, all_ratings):

    # conn = sqlite3.connect('content/database.db')

    predictions = []
    neighbours = user_user_similarity.iloc[userid]['neighbours']

    # find the set of neighbours that also rated movie 'a'
    for movie in movies_to_predict:

        denominator = sum([abs(sim) for user, sim in neighbours])
        numerator = 0

        for user, sim in neighbours:
            avg = user_profiles.iloc[user]['avg']
            numerator += sim * (all_ratings.iloc[user]['rating'] - avg)

        # numerator = sum([sim * (all_ratings.iloc[user]['rating'] - user_profiles.iloc[user]['avg'])
        #                  for user, sim in neighbours])

        prediction_user_item = user_average + numerator / denominator

        predictions.append((movie[2], prediction_user_item))

    # conn.close()

    return predictions


def sort_desc(list_to_sort):

    list_to_sort.sort(key=operator.itemgetter(1), reverse=True)
    return list_to_sort


def read_user_general_baseline():

    conn = sqlite3.connect('content/database.db')

    _ratings = pd.read_sql("SELECT userid, SUM(rating)/COUNT(rating) AS average "
                           "FROM movielens_rating GROUP BY userid ORDER BY userid", conn, columns=['userID', 'average'])
    # print _ratings.columns
    # print _ratings.head()
    # print _ratings.tail()
    # print _ratings[_ratings['userID'] == 85043]

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
def build_user_profile(user_profiles, convnet_similarity_matrix, user_user_matrix):

    _general_baseline, _global_average = read_user_general_baseline()
    _ratings_by_movie = read_movie_general_baseline()

    conn = sqlite3.connect('content/database.db')
    all_ratings = pd.read_sql('select userID, movielensID, rating from movielens_rating', conn, index_col='userID')

    conn.close()
    print "all ratings read"
    # start = time.time()

    new_user_profiles = {}

    for userid, profile in user_profiles.iterrows():

        movies_set = profile['relevant_set'] + profile['irrelevant_set'] + profile['random_set']

        if type(movies_set) == float:
            continue

        # predictions_content_based = get_predictions(profile['user_baseline'], movies_set, profile['all_movies'],
        #                                             convnet_similarity_matrix, _ratings_by_movie, _global_average)

        # print "part 1 tok", time.time() - start, "seconds"

        predictions_random = get_random_predictions(movies_set)

        # print "part 2 tok", time.time() - start, "seconds"

        # print movies_set
        # exit()

        predictions_collaborative = get_collaborative_predictions(userid, profile['avg'], movies_set,
                                                                  user_profiles, user_user_matrix, all_ratings)
        print predictions_collaborative
        exit()

        new_user_profiles[userid] = {'datasets': {'relevant_movies': profile['relevant_set'],
                                                  'irrelevant_movies': profile['irrelevant_set']},
                                     'userid': userid,
                                     'predictions': {'deep': predictions_content_based,
                                                     'random': predictions_random}}
    return new_user_profiles