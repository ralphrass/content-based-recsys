# import pandas as pd
from utils.opening_feat import load_features

user_profiles_with_predictions = load_features('content/profiles_with_predictions.pkl')
# df = pd.DataFrame.from_dict(user_profiles_with_predictions)

for index, profile in user_profiles_with_predictions.iteritems():
    print "index", index
    print profile

