# coding=utf-8
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

results = load_features('results_3112_users.pkl')

# collaborative, DeepRecVis (deep), user-centroid, user-centroid-relevant-movies, mixing-weighted-hybrid, weighted-weighted-hybrid
listing = []
# user_collaborative, item_collaborative, deep, weighted-hybrid, low_level = \
#     {'precision': [], 'recall': [], 'diversity': []}, {'precision': [], 'recall': [], 'diversity': []}, \
#     {'precision': [], 'recall': [], 'diversity': []}, {'precision': [], 'recall': [], 'diversity': []},\
#     {'precision': [], 'recall': [], 'diversity': []}
# user_collaborative, item_collaborative, deep, weighted_hybrid, low_level, weighted_hybrid_collaborative, \
# weighted_hybrid_item_content, switching_hybrid, tfidf, synopsis = \
deep, low_level, tfidf, synopsis = \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}

d_mae, ll_mae, tfidf_mae, synopsis_mae = (0, 0, 0, 0)

# print results
# exit()

for result in results.iteritems():

    d_mae = result[1]['deep']['mae']
    ll_mae = result[1]['low-level']['mae']
    tfidf_mae = result[1]['tfidf']['mae']
    synopsis_mae = result[1]['synopsis']['mae']

    listing.append(result[0])

    deep['precision'].append(result[1]['deep']['precision'])
    deep['recall'].append(result[1]['deep']['recall'])
    deep['diversity'].append(result[1]['deep']['diversity'])
    deep['rankscore'].append(result[1]['deep']['rankscore'])
    deep['f1'].append(result[1]['deep']['f1'])

    low_level['precision'].append(result[1]['low-level']['precision'])
    low_level['recall'].append(result[1]['low-level']['recall'])
    low_level['diversity'].append(result[1]['low-level']['diversity'])
    low_level['rankscore'].append(result[1]['low-level']['rankscore'])
    low_level['f1'].append(result[1]['low-level']['f1'])

    tfidf['precision'].append(result[1]['tfidf']['precision'])
    tfidf['recall'].append(result[1]['tfidf']['recall'])
    tfidf['diversity'].append(result[1]['tfidf']['diversity'])
    tfidf['rankscore'].append(result[1]['tfidf']['rankscore'])
    tfidf['f1'].append(result[1]['tfidf']['f1'])

    synopsis['precision'].append(result[1]['synopsis']['precision'])
    synopsis['recall'].append(result[1]['synopsis']['recall'])
    synopsis['diversity'].append(result[1]['synopsis']['diversity'])
    synopsis['rankscore'].append(result[1]['synopsis']['rankscore'])
    synopsis['f1'].append(result[1]['synopsis']['f1'])

linewidth = 1.1
list_styles = ['solid', 'dashed', 'dotted', 'dashdot']

# Precision
plt.plot(listing, deep['precision'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, low_level['precision'],  ls='solid', color='red', linewidth=linewidth, markersize=10, label='Baixo Nivel')
plt.plot(listing, tfidf['precision'], '->', color='maroon', linewidth=linewidth, label='Tags')
plt.plot(listing, synopsis['precision'], '-+', color='teal', linewidth=linewidth, label='Sinopse')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Precision', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('precision_content_based.pdf', bbox_inches='tight')
plt.close()

# Recall
plt.plot(listing, deep['recall'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, low_level['recall'], ls='solid', color='black', linewidth=linewidth, markersize=10, label='Baixo Nivel')
plt.plot(listing, tfidf['recall'], '->', color='maroon', linewidth=linewidth, label='Tags')
plt.plot(listing, synopsis['recall'], '<-', color='teal', linewidth=linewidth, label='Sinopse')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Recall', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('recall_content_based.pdf', bbox_inches='tight')
plt.close()

# Diversity
plt.plot(listing, deep['diversity'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, low_level['diversity'], ls='solid', color='red', linewidth=linewidth, markersize=10, label='Baixo Nivel')
plt.plot(listing, tfidf['diversity'], '->', color='maroon', linewidth=linewidth, label='Tags')
plt.plot(listing, synopsis['diversity'], '-+', color='teal', linewidth=linewidth, label='Sinopse')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Diversidade', fontsize=13)
plt.xlabel('N', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('diversity_content_based.pdf', bbox_inches='tight')
plt.close()

# Rankscore
plt.plot(listing, deep['rankscore'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, low_level['rankscore'], ls='solid', color='black', linewidth=linewidth, markersize=10, label='Baixo Nivel')
plt.plot(listing, tfidf['rankscore'], '->', color='maroon', linewidth=linewidth, label='Tags')
plt.plot(listing, synopsis['rankscore'], '<-', color='teal', linewidth=linewidth, label='Sinopse')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('rankscore', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('rankscore_content_based.pdf', bbox_inches='tight')
plt.close()

# F1
plt.plot(listing, deep['f1'], ls='solid', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, low_level['f1'],  ls='solid', color='red', linewidth=linewidth, markersize=10, label='Baixo Nivel')
plt.plot(listing, tfidf['f1'], '->', color='maroon', linewidth=linewidth, label='Tags')
plt.plot(listing, synopsis['f1'], '-+', color='teal', linewidth=linewidth, label='Sinopse')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('F1', fontsize=13)
plt.xlabel('N', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('f1_content_based.pdf', bbox_inches='tight')
plt.close()


plt.rcdefaults()
# fig, ax = plt.subplots()

# Example data
# metrics = ('Low-Level', 'DeepRecVis', 'weighted-hybrid-content-item', 'Item-Collaborative', 'weighted-hybrid',
#            'weighted-hybrid-collaborative', 'User-Collaborative')
metrics = ('Sinopse', 'Baixo Nivel', 'Tags', 'RNC')
y_pos = np.arange(len(metrics))
# mae = np.array([ll_mae, d_mae, hci_mae, ic_mae, h_mae, hc_mae, uc_mae])
mae = np.array([synopsis_mae, ll_mae, tfidf_mae, d_mae])
_qt_methods_to_compare = 4
index = np.arange(_qt_methods_to_compare)
bar_width = 0.35

plt.barh(y_pos, mae, align='center', color='darkblue', ecolor='black', height=0.4)
# plt.figure(figsize=(10, 5))
# plt.xlabel('Strategy')
# plt.ylabel('Mean Absolute Error')
plt.title('MAE')
plt.yticks(index, ('Sinopse', 'Baixo Nivel', 'Tags', 'RNC'))

plt.savefig('mae_content_based.pdf', bbox_inches='tight')
plt.show()
