from utils.opening_feat import load_features, save_obj
import numpy as np
from matplotlib import pyplot as plt

# results = load_features('results_3112_users.pkl')
# results_low_level = load_features('results_3112_users_low_level_features.pkl')

# for i in range(2, 16):
#     results[i]['low-level'] = results_low_level[i]['low-level']

# print results
# save_obj(results, 'full_results_3112_users')
# exit()

results = load_features('full_results_3112_users.pkl')

# collaborative, DeepRecVis (deep), user-centroid, user-centroid-relevant-movies, mixing-hybrid, weighted-hybrid
listing = []
user_collaborative, item_collaborative, deep, hybrid, low_level = \
    {'precision': [], 'recall': [], 'diversity': []}, {'precision': [], 'recall': [], 'diversity': []}, \
    {'precision': [], 'recall': [], 'diversity': []}, {'precision': [], 'recall': [], 'diversity': []},\
    {'precision': [], 'recall': [], 'diversity': []}

uc_mae, ic_mae, d_mae, h_mae, ll_mae = (0, 0, 0, 0, 0)

# print results
# exit()

for result in results.iteritems():

    uc_mae = result[1]['user-collaborative']['mae']
    ic_mae = result[1]['item-collaborative']['mae']
    d_mae = result[1]['deep']['mae']
    h_mae = result[1]['hybrid']['mae']
    ll_mae = result[1]['low-level']['mae']

    listing.append(result[0])

    user_collaborative['precision'].append(result[1]['user-collaborative']['precision'])
    user_collaborative['recall'].append(result[1]['user-collaborative']['recall'])
    user_collaborative['diversity'].append(result[1]['user-collaborative']['diversity'])

    deep['precision'].append(result[1]['deep']['precision'])
    deep['recall'].append(result[1]['deep']['recall'])
    deep['diversity'].append(result[1]['deep']['diversity'])

    item_collaborative['precision'].append(result[1]['item-collaborative']['precision'])
    item_collaborative['recall'].append(result[1]['item-collaborative']['recall'])
    item_collaborative['diversity'].append(result[1]['item-collaborative']['diversity'])

    hybrid['precision'].append(result[1]['hybrid']['precision'])
    hybrid['recall'].append(result[1]['hybrid']['recall'])
    hybrid['diversity'].append(result[1]['hybrid']['diversity'])

    low_level['precision'].append(result[1]['low-level']['precision'])
    low_level['recall'].append(result[1]['low-level']['recall'])
    low_level['diversity'].append(result[1]['low-level']['diversity'])

linewidth = 1.1
plt.plot(listing, user_collaborative['precision'], '--', color='blue', linewidth=linewidth, label='User-Collaborative')
plt.plot(listing, deep['precision'], '>--', color='k', markersize=7, linewidth=linewidth, label='DeepRecVis')
plt.plot(listing, item_collaborative['precision'], '-<', color='g', linewidth=linewidth, label='Item-Collaborative')
plt.plot(listing, hybrid['precision'],  '+-', color='darkorange', linewidth=linewidth, markersize=10, label='Hybrid')
plt.plot(listing, low_level['precision'],  '-+', color='red', linewidth=linewidth, markersize=10, label='Low-Level')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Precision', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('precision.pdf', bbox_inches='tight')
plt.close()

plt.plot(listing, user_collaborative['recall'], '--', color='blue', linewidth=linewidth, label='User-Collaborative')
plt.plot(listing, deep['recall'], '>--', color='k', markersize=7, linewidth=linewidth, label='DeepRecVis')
plt.plot(listing, item_collaborative['recall'], '-<', color='g', linewidth=linewidth, label='Item-Collaborative')
plt.plot(listing, hybrid['recall'], '+-', color='darkorange', linewidth=linewidth, label='Hybrid')
plt.plot(listing, low_level['recall'],  '-+', color='red', linewidth=linewidth, markersize=10, label='Low-Level')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Recall', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('recall.pdf', bbox_inches='tight')
plt.close()

plt.plot(listing, user_collaborative['diversity'], '--', color='blue', linewidth=linewidth, label='User-Collaborative')
plt.plot(listing, deep['diversity'], '>--', color='k', markersize=7, linewidth=linewidth, label='DeepRecVis')
plt.plot(listing, item_collaborative['diversity'], '-<', color='g', linewidth=linewidth, label='Item-Collaborative')
plt.plot(listing, hybrid['diversity'], '+-', color='darkorange', linewidth=linewidth, label='Hybrid')
plt.plot(listing, low_level['diversity'],  '-+', color='red', linewidth=linewidth, markersize=10, label='Low-Level')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Diversity', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('diversity.pdf', bbox_inches='tight')
plt.close()

plt.rcdefaults()
# fig, ax = plt.subplots()

# Example data
metrics = ('User-Collaborative', 'Low-Level', 'DeepRecVis', 'Hybrid', 'Item-Collaborative')
y_pos = np.arange(len(metrics))
mae = np.array([uc_mae, ll_mae, d_mae, h_mae, ic_mae])
index = np.arange(5)
bar_width = 0.35

plt.barh(y_pos, mae, align='center', color='darkblue', ecolor='black', height=0.4)
# plt.figure(figsize=(10, 5))
# plt.xlabel('Strategy')
# plt.ylabel('Mean Absolute Error')
plt.title('MAE')
plt.yticks(index, ('User-Collaborative', 'Low-Level', 'DeepRecVis', 'Hybrid', 'Item-Collaborative'))

plt.savefig('mae.pdf', bbox_inches='tight')
plt.show()
