from utils.opening_feat import load_features, save_obj
from hausdorff import hausdorff
from utils.utils import sort_desc
import numpy as np

_users_bof = load_features('content/3112_users_bof.pkl')

# test_user_1 = np.array(_users_bof[1])
# test_user_3 = np.array(_users_bof[7])
# print test_user_1
# print hausdorff(test_user_1, test_user_3)

users_bof_similarities = {}

for key, user_bof in _users_bof.iteritems():
    users_bof_similarities[key] = []
    print "current user", key

    for neighbor, neighbor_bof in _users_bof.iteritems():
        if neighbor == key:
            continue

        sim = hausdorff(np.array(user_bof), np.array(neighbor_bof))
        users_bof_similarities[key].append((neighbor, sim))

    users_bof_similarities[key] = sort_desc(users_bof_similarities[key], desc=False)
    # print users_bof_similarities[key]
    # break

save_obj(users_bof_similarities, '3112_user_user_bof_similarities')
