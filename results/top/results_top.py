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

results = load_features('../results_3112_users.pkl')

listing = []

user_collaborative, item_collaborative, deep = \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}


uc_mae, ic_mae, d_mae = (0, 0, 0)

# print results
# exit()

for result in results.iteritems():

    uc_mae = result[1]['user-collaborative']['mae']
    ic_mae = result[1]['item-collaborative']['mae']
    d_mae = result[1]['deep']['mae']

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

linewidth = 1.1
list_styles = ['solid', 'dashed', 'dotted', 'dashdot']

# Precision
plt.plot(listing, user_collaborative['precision'], ls='solid', color='blue', linewidth=linewidth, label='FC-U')
plt.plot(listing, deep['precision'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, item_collaborative['precision'], ls='dotted', color='green', linewidth=linewidth, label='FC-I')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Precision', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('precision_top.pdf', bbox_inches='tight')
plt.close()

# Recall
plt.plot(listing, user_collaborative['recall'], ls='solid', color='blue', linewidth=linewidth, label='FC-U')
plt.plot(listing, deep['recall'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, item_collaborative['recall'], ls='solid', color='green', linewidth=linewidth, label='FC-I')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Recall', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('recall_top.pdf', bbox_inches='tight')
plt.close()

# Diversity
plt.plot(listing, user_collaborative['diversity'], ls='solid', color='blue', linewidth=linewidth, label='FC-U')
plt.plot(listing, deep['diversity'], ls='solid', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, item_collaborative['diversity'], ls='solid', color='green', linewidth=linewidth, label='FC-I')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Diversidade', fontsize=13)
plt.xlabel('N', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('diversity_top.pdf', bbox_inches='tight')
plt.close()

# Rankscore
plt.plot(listing, user_collaborative['rankscore'], ls='solid', color='blue', linewidth=linewidth, label='FC-U')
plt.plot(listing, deep['rankscore'], ls='dashed', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, item_collaborative['rankscore'], ls='dotted', color='green', linewidth=linewidth, label='FC-I')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('rankscore', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('rankscore_top.pdf', bbox_inches='tight')
plt.close()

# F1
plt.plot(listing, user_collaborative['f1'], ls='solid', color='blue', linewidth=linewidth, label='FC-U')
plt.plot(listing, deep['f1'], ls='solid', color='navy', markersize=7, linewidth=linewidth, label='RNC')
plt.plot(listing, item_collaborative['f1'], ls='solid', color='green', linewidth=linewidth, label='FC-I')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('F1', fontsize=13)
plt.xlabel('N', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('f1_top.pdf', bbox_inches='tight')
plt.close()


plt.rcdefaults()
# fig, ax = plt.subplots()

# Example data
# metrics = ('Low-Level', 'DeepRecVis', 'weighted-hybrid-content-item', 'Item-Collaborative', 'weighted-hybrid',
#            'weighted-hybrid-collaborative', 'User-Collaborative')
metrics = ('RNC', 'FC-I', 'FC-U')
y_pos = np.arange(len(metrics))
# mae = np.array([ll_mae, d_mae, hci_mae, ic_mae, h_mae, hc_mae, uc_mae])
mae = np.array([d_mae, ic_mae, uc_mae])
_qt_methods_to_compare = 3
index = np.arange(_qt_methods_to_compare)
bar_width = 0.35

plt.barh(y_pos, mae, align='center', color='darkblue', ecolor='black', height=0.4)
# plt.figure(figsize=(10, 5))
# plt.xlabel('Strategy')
# plt.ylabel('Mean Absolute Error')
plt.title('MAE')
plt.yticks(index, ('RNC', 'FC-I', 'FC-U'))

plt.savefig('mae_top.pdf', bbox_inches='tight')
plt.show()
