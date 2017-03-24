from utils.opening_feat import load_features
import numpy as np
from matplotlib import pyplot as plt

results = load_features('results_5_users.pkl')

# collaborative, DeepRecVis (deep), user-centroid, user-centroid-relevant-movies, mixing-hybrid, weighted-hybrid
listing = []
collaborative, deep, user_centroid, weighted_hybrid, mixing_hybrid = \
    {'precision': [], 'recall': [], 'diversity': []}, {'precision': [], 'recall': [], 'diversity': []}, \
    {'precision': [], 'recall': [], 'diversity': []}, {'precision': [], 'recall': [], 'diversity': []},\
    {'precision': [], 'recall': [], 'diversity': []}

for result in results.iteritems():
    listing.append(result[0])
    collaborative['precision'].append(result[1]['collaborative']['precision'])
    collaborative['recall'].append(result[1]['collaborative']['recall'])
    collaborative['diversity'].append(result[1]['collaborative']['diversity'])

    deep['precision'].append(result[1]['deep']['precision'])
    deep['recall'].append(result[1]['deep']['recall'])
    deep['diversity'].append(result[1]['deep']['diversity'])

    # print result[1]['user-centroid']

    user_centroid['precision'].append(result[1]['user-centroid']['precision'])
    user_centroid['recall'].append(result[1]['user-centroid']['recall'])
    user_centroid['diversity'].append(result[1]['user-centroid']['diversity'])

    weighted_hybrid['precision'].append(result[1]['weighted-hybrid']['precision'])
    weighted_hybrid['recall'].append(result[1]['weighted-hybrid']['recall'])
    weighted_hybrid['diversity'].append(result[1]['weighted-hybrid']['diversity'])

    mixing_hybrid['precision'].append(result[1]['mixing-hybrid']['precision'])
    mixing_hybrid['recall'].append(result[1]['mixing-hybrid']['recall'])
    mixing_hybrid['diversity'].append(result[1]['mixing-hybrid']['diversity'])

# linewidth = 1.1
# plt.plot(listing, collaborative['precision'], '--', color='blue', linewidth=linewidth, label='Collaborative')
# plt.plot(listing, deep['precision'], '>--', color='k', markersize=7, linewidth=linewidth, label='DeepRecVis')
# plt.plot(listing, user_centroid['precision'], '-<', color='g', linewidth=linewidth, label='User-Centroid')
# plt.plot(listing, weighted_hybrid['precision'],  '+-', color='darkorange', linewidth=linewidth, markersize=10, label='Weighted-Hybrid')
# plt.grid(True)
# plt.xlim((2, 15))
# plt.ylabel('Precision', fontsize=13)
# plt.xlabel('Iteration', fontsize=13)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
#           fancybox=True, shadow=True, ncol=2, fontsize=13)
# plt.savefig('precision.pdf', bbox_inches='tight')

linewidth = 1.1
plt.plot(listing, collaborative['recall'], '--', color='blue', linewidth=linewidth, label='Collaborative')
plt.plot(listing, deep['recall'], '>--', color='k', markersize=7, linewidth=linewidth, label='DeepRecVis')
plt.plot(listing, user_centroid['recall'], '-<', color='g', linewidth=linewidth, label='User-Centroid')
plt.plot(listing, weighted_hybrid['recall'], '+-', color='darkorange', linewidth=linewidth, label='Weighted-Hybrid')
plt.plot(listing, weighted_hybrid['recall'], '>-', color='r', linewidth=linewidth, label='Mixing-Hybrid')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Precision', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('recall.pdf', bbox_inches='tight')