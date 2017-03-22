import pandas as pd
import sqlite3
import random
import time
from sklearn.metrics.pairwise import cosine_similarity

conn = sqlite3.connect('../content/database.db')
all_ratings = pd.read_sql('select userID, movielensID, rating from movielens_rating', conn, index_col='userID')
conn.close()

start = time.time()

# user_ids_size = len(user_ids)
max_limit = 100
# limit = user_ids_size if user_ids_size < max_limit else max_limit
# neighbours = random.sample(all_ratings[all_ratings['movielensID'] == 1079].index, max_limit)
neighbours = all_ratings[all_ratings['movielensID'] == 1079].sample(n=10000).index

current_user_ratings = all_ratings.loc[1]

numerator, denominator = 0, 0

print time.time() - start, "seconds"

part11 = 0
part12 = 0
part2 = 0
part3 = 0

# bottleneck from here
start = time.time()

for neighbour_id in neighbours:

    if neighbour_id == 1:
        continue

    start = time.time()
    neighbour_movies = all_ratings.loc[neighbour_id]
    end = time.time()
    part11 += end - start

    start = time.time()
    # intersection = pd.merge(neighbour_movies, current_user_ratings, on='movielensID')

    ratings_user_x = current_user_ratings[current_user_ratings['movielensID'].isin(neighbour_movies['movielensID'].tolist())]['rating']
    ratings_user_y = neighbour_movies[neighbour_movies['movielensID'].isin(current_user_ratings['movielensID'].tolist())]['rating']

    # print len(ratings_user_x), ratings_user_x.reshape(1, -1)
    # print len(intersection['rating_x']), intersection['rating_x'].reshape(1, -1)
    # print len(ratings_user_y), ratings_user_y.reshape(1, -1)
    # print len(intersection['rating_y']), intersection['rating_y'].reshape(1, -1)
    # exit()

    end = time.time()
    part12 += end - start

    start = time.time()
    try:
        # similarity = cosine_similarity(intersection['rating_x'].reshape(1, -1),
        #                                intersection['rating_y'].reshape(1, -1))[0][0]
        similarity = cosine_similarity(ratings_user_x.reshape(1, -1), ratings_user_y.reshape(1, -1))
    except ValueError:
        similarity = 0
    end = time.time()
    part2 += end-start

    start = time.time()
    denominator += abs(similarity)
    neighbour_movie_rating = neighbour_movies[neighbour_movies['movielensID'] == 1079]['rating'].iloc[0]
    numerator += similarity * (neighbour_movie_rating - neighbour_movies['rating'].mean())
    end = time.time()
    part3 += end - start

print time.time() - start, "seconds"

print "part 1.1", part11 / neighbours.size, "seconds on average", part11 * neighbours.size, "on total"
print "part 1.2", part12 / neighbours.size, "seconds on average", part12 * neighbours.size, "on total"
print "part 2", part2 / neighbours.size, "seconds on average", part2 * neighbours.size, "on total"
print "part 3", part3 / neighbours.size, "seconds on average", part3 * neighbours.size, "on total"