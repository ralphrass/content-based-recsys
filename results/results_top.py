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

results = load_features('results_500_users.pkl')

# collaborative, DeepRecVis (deep), user-centroid, user-centroid-relevant-movies, mixing-weighted-hybrid, weighted-weighted-hybrid
listing = []
# user_collaborative, item_collaborative, deep, weighted-hybrid, low_level = \
#     {'precision': [], 'recall': [], 'diversity': []}, {'precision': [], 'recall': [], 'diversity': []}, \
#     {'precision': [], 'recall': [], 'diversity': []}, {'precision': [], 'recall': [], 'diversity': []},\
#     {'precision': [], 'recall': [], 'diversity': []}
# user_collaborative, item_collaborative, deep, weighted_hybrid, low_level, weighted_hybrid_collaborative, \
# weighted_hybrid_item_content, switching_hybrid, tfidf, synopsis = \
user_collaborative, item_collaborative, deep, weighted_hybrid, low_level, switching_hybrid, tfidf, synopsis = \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}
    # {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    # {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}

uc_mae, ic_mae, d_mae, h_mae, ll_mae, switching_mae, tfidf_mae, synopsis_mae = (0, 0, 0, 0, 0, 0, 0, 0)

# print results
# exit()

for result in results.iteritems():

    uc_mae = result[1]['user-collaborative']['mae']
    ic_mae = result[1]['item-collaborative']['mae']
    d_mae = result[1]['deep']['mae']
    h_mae = result[1]['weighted-hybrid']['mae']
    ll_mae = result[1]['low-level']['mae']
    # hc_mae = result[1]['weighted-hybrid-collaborative']['mae']
    # hci_mae = result[1]['weighted-hybrid-content-item']['mae']
    switching_mae = result[1]['switching-hybrid']['mae']
    tfidf_mae = result[1]['tfidf']['mae']
    synopsis_mae = result[1]['synopsis']['mae']

    listing.append(result[0])

    user_collaborative['precision'].append(result[1]['user-collaborative']['precision'])
    user_collaborative['recall'].append(result[1]['user-collaborative']['recall'])
    user_collaborative['diversity'].append(result[1]['user-collaborative']['diversity'])
    user_collaborative['rankscore'].append(result[1]['user-collaborative']['rankscore'])
    user_collaborative['f1'].append(result[1]['user-collaborative']['f1'])

    deep['precision'].append(result[1]['deep']['precision'])
    deep['recall'].append(result[1]['deep']['recall'])
    deep['diversity'].append(result[1]['deep']['diversity'])
    deep['rankscore'].append(result[1]['deep']['rankscore'])
    deep['f1'].append(result[1]['deep']['f1'])

    item_collaborative['precision'].append(result[1]['item-collaborative']['precision'])
    item_collaborative['recall'].append(result[1]['item-collaborative']['recall'])
    item_collaborative['diversity'].append(result[1]['item-collaborative']['diversity'])
    item_collaborative['rankscore'].append(result[1]['item-collaborative']['rankscore'])
    item_collaborative['f1'].append(result[1]['item-collaborative']['f1'])

    weighted_hybrid['precision'].append(result[1]['weighted-hybrid']['precision'])
    weighted_hybrid['recall'].append(result[1]['weighted-hybrid']['recall'])
    weighted_hybrid['diversity'].append(result[1]['weighted-hybrid']['diversity'])
    weighted_hybrid['rankscore'].append(result[1]['weighted-hybrid']['rankscore'])
    weighted_hybrid['f1'].append(result[1]['weighted-hybrid']['f1'])

    low_level['precision'].append(result[1]['low-level']['precision'])
    low_level['recall'].append(result[1]['low-level']['recall'])
    low_level['diversity'].append(result[1]['low-level']['diversity'])
    low_level['rankscore'].append(result[1]['low-level']['rankscore'])
    low_level['f1'].append(result[1]['low-level']['f1'])

    # weighted_hybrid_collaborative['precision'].append(result[1]['weighted-hybrid-collaborative']['precision'])
    # weighted_hybrid_collaborative['recall'].append(result[1]['weighted-hybrid-collaborative']['recall'])
    # weighted_hybrid_collaborative['diversity'].append(result[1]['weighted-hybrid-collaborative']['diversity'])
    # weighted_hybrid_collaborative['rankscore'].append(result[1]['weighted-hybrid-collaborative']['rankscore'])
    # weighted_hybrid_collaborative['f1'].append(result[1]['weighted-hybrid-collaborative']['f1'])
    #
    # weighted_hybrid_item_content['precision'].append(result[1]['weighted-hybrid-content-item']['precision'])
    # weighted_hybrid_item_content['recall'].append(result[1]['weighted-hybrid-content-item']['recall'])
    # weighted_hybrid_item_content['diversity'].append(result[1]['weighted-hybrid-content-item']['diversity'])
    # weighted_hybrid_item_content['rankscore'].append(result[1]['weighted-hybrid-content-item']['rankscore'])
    # weighted_hybrid_item_content['f1'].append(result[1]['weighted-hybrid-content-item']['f1'])

    switching_hybrid['precision'].append(result[1]['switching-hybrid']['precision'])
    switching_hybrid['recall'].append(result[1]['switching-hybrid']['recall'])
    switching_hybrid['diversity'].append(result[1]['switching-hybrid']['diversity'])
    switching_hybrid['rankscore'].append(result[1]['switching-hybrid']['rankscore'])
    switching_hybrid['f1'].append(result[1]['switching-hybrid']['f1'])

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
plt.plot(listing, user_collaborative['precision'], ls='solid', color='blue', linewidth=linewidth, label='FC-U')
plt.plot(listing, deep['precision'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, item_collaborative['precision'], ls='dashdot', color='green', linewidth=linewidth, label='FC-I')
plt.plot(listing, weighted_hybrid['precision'],  ls='dashdot', color='olive', linewidth=linewidth, markersize=10, label='Hibrido por Pesos')
plt.plot(listing, low_level['precision'],  ls='solid', color='red', linewidth=linewidth, markersize=10, label='Baixo Nivel')
plt.plot(listing, switching_hybrid['precision'], ls='dashed', color='purple', linewidth=linewidth, markersize=10, label='Hibrido Alternado')
plt.plot(listing, tfidf['precision'], '->', color='maroon', linewidth=linewidth, label='Tags')
plt.plot(listing, synopsis['precision'], '<-', color='teal', linewidth=linewidth, label='Sinopse')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Precision', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('precision.pdf', bbox_inches='tight')
plt.close()

# Recall
plt.plot(listing, user_collaborative['recall'], ls='solid', color='blue', linewidth=linewidth, label='User-Collaborative')
plt.plot(listing, deep['recall'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='DeepRecVis')
plt.plot(listing, item_collaborative['recall'], ls='dashdot', color='green', linewidth=linewidth, label='Item-Collaborative')
plt.plot(listing, weighted_hybrid['recall'], ls='dashdot', color='olive', linewidth=linewidth, label='weighted-hybrid')
plt.plot(listing, low_level['recall'], ls='solid', color='black', linewidth=linewidth, markersize=10, label='Low-Level')
# plt.plot(listing, weighted_hybrid_collaborative['recall'], color='fuchsia', linewidth=linewidth, label='weighted-hybrid-collaborative')
# plt.plot(listing, weighted_hybrid_item_content['recall'], color='aqua', linewidth=linewidth, label='weighted-hybrid-content-item')
plt.plot(listing, switching_hybrid['recall'], ls='dashed', color='purple', linewidth=linewidth, markersize=10, label='Switching Hybrid')
plt.plot(listing, tfidf['recall'], '->', color='maroon', linewidth=linewidth, label='TF-IDF')
plt.plot(listing, synopsis['recall'], '<-', color='teal', linewidth=linewidth, label='Synopsis')

# plt.plot(listing, user_collaborative['recall'], linewidth=linewidth, label='User-Collaborative')
# plt.plot(listing, deep['recall'], markersize=7, linewidth=linewidth, label='DeepRecVis')
# plt.plot(listing, item_collaborative['recall'], linewidth=linewidth, label='Item-Collaborative')
# plt.plot(listing, weighted_hybrid['recall'], linewidth=linewidth, label='weighted-hybrid')
# plt.plot(listing, low_level['recall'], linewidth=linewidth, markersize=10, label='Low-Level')
# plt.plot(listing, weighted_hybrid_collaborative['recall'], linewidth=linewidth, label='weighted-hybrid-collaborative')
# plt.plot(listing, weighted_hybrid_item_content['recall'], linewidth=linewidth, label='weighted-hybrid-content-item')

plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Recall', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('recall.pdf', bbox_inches='tight')
plt.close()

# Diversity
plt.plot(listing, user_collaborative['diversity'], ls='solid', color='blue', linewidth=linewidth, label='User-Collaborative')
plt.plot(listing, deep['diversity'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='DeepRecVis')
plt.plot(listing, item_collaborative['diversity'], ls='dashdot', color='green', linewidth=linewidth, label='Item-Collaborative')
plt.plot(listing, weighted_hybrid['diversity'], ls='dashdot', color='olive', linewidth=linewidth, label='weighted-hybrid')
plt.plot(listing, low_level['diversity'], ls='solid', color='black', linewidth=linewidth, markersize=10, label='Low-Level')
plt.plot(listing, switching_hybrid['diversity'], ls='dashed', color='purple', linewidth=linewidth, markersize=10, label='Switching Hybrid')
plt.plot(listing, tfidf['diversity'], '->', color='maroon', linewidth=linewidth, label='TF-IDF')
plt.plot(listing, synopsis['diversity'], '<-', color='teal', linewidth=linewidth, label='Synopsis')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Diversity', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('diversity.pdf', bbox_inches='tight')
plt.close()

# Rankscore
plt.plot(listing, user_collaborative['rankscore'], ls='solid', color='blue', linewidth=linewidth, label='User-Collaborative')
plt.plot(listing, deep['rankscore'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='DeepRecVis')
plt.plot(listing, item_collaborative['rankscore'], ls='dashdot', color='green', linewidth=linewidth, label='Item-Collaborative')
plt.plot(listing, weighted_hybrid['rankscore'], ls='dashdot', color='olive', linewidth=linewidth, label='weighted-hybrid')
plt.plot(listing, low_level['rankscore'], ls='solid', color='black', linewidth=linewidth, markersize=10, label='Low-Level')
plt.plot(listing, switching_hybrid['rankscore'], ls='dashed', color='purple', linewidth=linewidth, markersize=10, label='Switching Hybrid')
plt.plot(listing, tfidf['rankscore'], '->', color='maroon', linewidth=linewidth, label='TF-IDF')
plt.plot(listing, synopsis['rankscore'], '<-', color='teal', linewidth=linewidth, label='Synopsis')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('rankscore', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('rankscore.pdf', bbox_inches='tight')
plt.close()

# F1
plt.plot(listing, user_collaborative['f1'], ls='solid', color='blue', linewidth=linewidth, label='User-Collaborative')
plt.plot(listing, deep['f1'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='DeepRecVis')
plt.plot(listing, item_collaborative['f1'], ls='dashdot', color='green', linewidth=linewidth, label='Item-Collaborative')
plt.plot(listing, weighted_hybrid['f1'], ls='dashdot', color='olive', linewidth=linewidth, label='weighted-hybrid')
plt.plot(listing, low_level['f1'],  ls='solid', color='black', linewidth=linewidth, markersize=10, label='Low-Level')
plt.plot(listing, switching_hybrid['f1'], ls='dashed', color='purple', linewidth=linewidth, markersize=10, label='Switching Hybrid')
plt.plot(listing, tfidf['f1'], '->', color='maroon', linewidth=linewidth, label='TF-IDF')
plt.plot(listing, synopsis['f1'], '<-', color='teal', linewidth=linewidth, label='Synopsis')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('F1', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('f1.pdf', bbox_inches='tight')
plt.close()


plt.rcdefaults()
# fig, ax = plt.subplots()

# Example data
# metrics = ('Low-Level', 'DeepRecVis', 'weighted-hybrid-content-item', 'Item-Collaborative', 'weighted-hybrid',
#            'weighted-hybrid-collaborative', 'User-Collaborative')
metrics = ('FC-U', 'RNC', 'Hibrido por Pesos', 'FC-I', 'Hibrido Alternado', 'Tags', 'Baixo Nivel', 'Sinopse')
y_pos = np.arange(len(metrics))
# mae = np.array([ll_mae, d_mae, hci_mae, ic_mae, h_mae, hc_mae, uc_mae])
mae = np.array([uc_mae, h_mae, ic_mae, switching_mae, d_mae, tfidf_mae, ll_mae, synopsis_mae])
_qt_methods_to_compare = 8
index = np.arange(_qt_methods_to_compare)
bar_width = 0.35

plt.barh(y_pos, mae, align='center', color='darkblue', ecolor='black', height=0.4)
# plt.figure(figsize=(10, 5))
# plt.xlabel('Strategy')
# plt.ylabel('Mean Absolute Error')
plt.title('MAE')
plt.yticks(index, ('FC-U', 'Hibrido por Pesos', 'FC-I', 'Hibrido Alternado', 'RNC', 'Tags', 'Baixo Nivel', 'Sinopse'))

plt.savefig('mae.pdf', bbox_inches='tight')
plt.show()
