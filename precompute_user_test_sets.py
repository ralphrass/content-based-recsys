import sqlite3
import pandas as pd
import time
from utils.utils import select_random_users, get_random_movie_set, get_user_training_test_movies
from recommender import get_user_baseline, read_user_general_baseline

start = time.time()

conn = sqlite3.connect('content/database.db')
users = select_random_users(conn, 0, 1000)

_general_baseline, _global_average = read_user_general_baseline()

user_movies_df = pd.DataFrame(columns=['userid', 'avg', 'user_baseline', 'relevant_set', 'irrelevant_set', 'all_movies',
                                       'random_set'], index=[u[0] for u in users])

count = 1

for user in users:

    if count % 5000 == 0:
        print count, "users read"
        # break

    user_baseline = get_user_baseline(user[0], _general_baseline, _global_average)
    relevant_set, all_movies, irrelevant_set = get_user_training_test_movies(user[0])
    random_set = get_random_movie_set(user[0])

    user_movies_df.loc[user[0]] = pd.Series({'userid': user[0], 'avg': user[1], 'user_baseline': user_baseline,
                                             'relevant_set': relevant_set, 'irrelevant_set': irrelevant_set,
                                             'all_movies': all_movies, 'random_set': random_set})

    count += 1

print user_movies_df.columns
print user_movies_df.head()

# user_movies_df.to_pickle('user_profiles_dataframe_5k.pkl')
user_movies_df.to_pickle('user_profiles_dataframe.pkl')

print "it tok", time.time() - start, "seconds"