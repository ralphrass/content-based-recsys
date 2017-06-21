from utils.opening_feat import load_features, save_obj
import numpy as np
from matplotlib import pyplot as plt

results = load_features('../results_3112_users.pkl')
listing = []

wh_synopsis_tags, wh_synopsis_low_level, wh_synopsis_item_collaborative, \
wh_synopsis_deep, wh_synopsis_user_collaborative, wh_tags_low_level, \
wh_tags_item_collaborative, wh_tags_deep, wh_tags_user_collaborative, \
wh_low_level_deep, wh_low_level_item_collaborative, \
wh_low_level_user_collaborative, wh_item_collaborative_deep, \
wh_item_collaborative_user_collaborative, wh_deep_user_collaborative = \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}, \
    {'precision': [], 'recall': [], 'diversity': [], 'rankscore': [], 'f1': []}

wh_synopsis_tags_mae, wh_synopsis_low_level_mae, \
wh_synopsis_item_collaborative_mae, wh_synopsis_deep_mae, \
wh_synopsis_user_collaborative_mae, wh_tags_low_level_mae, \
wh_tags_item_collaborative_mae, wh_tags_deep_mae, \
wh_tags_user_collaborative_mae, wh_low_level_deep_mae, \
wh_low_level_item_collaborative_mae, wh_low_level_user_collaborative_mae, \
wh_item_collaborative_deep_mae, wh_item_collaborative_user_collaborative_mae, \
wh_deep_user_collaborative_mae = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

# print results
# exit()

for result in results.iteritems():

    wh_synopsis_tags_mae = result[1]['weighted-hybrid-synopsis-tags']['mae']
    wh_synopsis_low_level_mae = result[1]['weighted-hybrid-synopsis-low-level']['mae']
    wh_synopsis_item_collaborative_mae = result[1]['weighted-hybrid-synopsis-item-collaborative']['mae']
    wh_synopsis_deep_mae = result[1]['weighted-hybrid-synopsis-deep']['mae']
    wh_synopsis_user_collaborative_mae = result[1]['weighted-hybrid-synopsis-user-collaborative']['mae']
    wh_tags_low_level_mae = result[1]['weighted-hybrid-tags-low-level']['mae']
    wh_tags_item_collaborative_mae = result[1]['weighted-hybrid-tags-item-collaborative']['mae']
    wh_tags_deep_mae = result[1]['weighted-hybrid-tags-deep']['mae']
    wh_tags_user_collaborative_mae = result[1]['weighted-hybrid-tags-user-collaborative']['mae']
    wh_low_level_deep_mae = result[1]['weighted-hybrid-low-level-deep']['mae']
    wh_low_level_item_collaborative_mae = result[1]['weighted-hybrid-low-level-item-collaborative']['mae']
    wh_low_level_user_collaborative_mae = result[1]['weighted-hybrid-low-level-user-collaborative']['mae']
    wh_item_collaborative_deep_mae = result[1]['weighted-hybrid-item-collaborative-deep']['mae']
    wh_item_collaborative_user_collaborative_mae = \
        result[1]['weighted-hybrid-item-collaborative-user-collaborative']['mae']
    wh_deep_user_collaborative_mae = result[1]['weighted-hybrid-deep-user-collaborative']['mae']

    listing.append(result[0])

    wh_synopsis_tags['precision'].append(result[1]['weighted-hybrid-synopsis-tags']['precision'])
    wh_synopsis_tags['recall'].append(result[1]['weighted-hybrid-synopsis-tags']['recall'])
    wh_synopsis_tags['diversity'].append(result[1]['weighted-hybrid-synopsis-tags']['diversity'])
    wh_synopsis_tags['rankscore'].append(result[1]['weighted-hybrid-synopsis-tags']['rankscore'])
    wh_synopsis_tags['f1'].append(result[1]['weighted-hybrid-synopsis-tags']['f1'])

    wh_synopsis_low_level['precision'].append(result[1]['weighted-hybrid-synopsis-low-level']['precision'])
    wh_synopsis_low_level['recall'].append(result[1]['weighted-hybrid-synopsis-low-level']['recall'])
    wh_synopsis_low_level['diversity'].append(result[1]['weighted-hybrid-synopsis-low-level']['diversity'])
    wh_synopsis_low_level['rankscore'].append(result[1]['weighted-hybrid-synopsis-low-level']['rankscore'])
    wh_synopsis_low_level['f1'].append(result[1]['weighted-hybrid-synopsis-low-level']['f1'])

    wh_synopsis_item_collaborative['precision'].append(result[1]['weighted-hybrid-synopsis-item-collaborative']['precision'])
    wh_synopsis_item_collaborative['recall'].append(result[1]['weighted-hybrid-synopsis-item-collaborative']['recall'])
    wh_synopsis_item_collaborative['diversity'].append(result[1]['weighted-hybrid-synopsis-item-collaborative']['diversity'])
    wh_synopsis_item_collaborative['rankscore'].append(result[1]['weighted-hybrid-synopsis-item-collaborative']['rankscore'])
    wh_synopsis_item_collaborative['f1'].append(result[1]['weighted-hybrid-synopsis-item-collaborative']['f1'])

    wh_synopsis_deep['precision'].append(result[1]['weighted-hybrid-synopsis-deep']['precision'])
    wh_synopsis_deep['recall'].append(result[1]['weighted-hybrid-synopsis-deep']['recall'])
    wh_synopsis_deep['diversity'].append(result[1]['weighted-hybrid-synopsis-deep']['diversity'])
    wh_synopsis_deep['rankscore'].append(result[1]['weighted-hybrid-synopsis-deep']['rankscore'])
    wh_synopsis_deep['f1'].append(result[1]['weighted-hybrid-synopsis-deep']['f1'])

    wh_synopsis_user_collaborative['precision'].append(result[1]['weighted-hybrid-synopsis-user-collaborative']['precision'])
    wh_synopsis_user_collaborative['recall'].append(result[1]['weighted-hybrid-synopsis-user-collaborative']['recall'])
    wh_synopsis_user_collaborative['diversity'].append(result[1]['weighted-hybrid-synopsis-user-collaborative']['diversity'])
    wh_synopsis_user_collaborative['rankscore'].append(result[1]['weighted-hybrid-synopsis-user-collaborative']['rankscore'])
    wh_synopsis_user_collaborative['f1'].append(result[1]['weighted-hybrid-synopsis-user-collaborative']['f1'])

    wh_tags_low_level['precision'].append(result[1]['weighted-hybrid-tags-low-level']['precision'])
    wh_tags_low_level['recall'].append(result[1]['weighted-hybrid-tags-low-level']['recall'])
    wh_tags_low_level['diversity'].append(result[1]['weighted-hybrid-tags-low-level']['diversity'])
    wh_tags_low_level['rankscore'].append(result[1]['weighted-hybrid-tags-low-level']['rankscore'])
    wh_tags_low_level['f1'].append(result[1]['weighted-hybrid-tags-low-level']['f1'])

    wh_tags_item_collaborative['precision'].append(result[1]['weighted-hybrid-tags-item-collaborative']['precision'])
    wh_tags_item_collaborative['recall'].append(result[1]['weighted-hybrid-tags-item-collaborative']['recall'])
    wh_tags_item_collaborative['diversity'].append(result[1]['weighted-hybrid-tags-item-collaborative']['diversity'])
    wh_tags_item_collaborative['rankscore'].append(result[1]['weighted-hybrid-tags-item-collaborative']['rankscore'])
    wh_tags_item_collaborative['f1'].append(result[1]['weighted-hybrid-tags-item-collaborative']['f1'])

    wh_tags_deep['precision'].append(result[1]['weighted-hybrid-tags-deep']['precision'])
    wh_tags_deep['recall'].append(result[1]['weighted-hybrid-tags-deep']['recall'])
    wh_tags_deep['diversity'].append(result[1]['weighted-hybrid-tags-deep']['diversity'])
    wh_tags_deep['rankscore'].append(result[1]['weighted-hybrid-tags-deep']['rankscore'])
    wh_tags_deep['f1'].append(result[1]['weighted-hybrid-tags-deep']['f1'])

    wh_tags_user_collaborative['precision'].append(result[1]['weighted-hybrid-tags-user-collaborative']['precision'])
    wh_tags_user_collaborative['recall'].append(result[1]['weighted-hybrid-tags-user-collaborative']['recall'])
    wh_tags_user_collaborative['diversity'].append(result[1]['weighted-hybrid-tags-user-collaborative']['diversity'])
    wh_tags_user_collaborative['rankscore'].append(result[1]['weighted-hybrid-tags-user-collaborative']['rankscore'])
    wh_tags_user_collaborative['f1'].append(result[1]['weighted-hybrid-tags-user-collaborative']['f1'])

    wh_low_level_deep['precision'].append(result[1]['weighted-hybrid-low-level-deep']['precision'])
    wh_low_level_deep['recall'].append(result[1]['weighted-hybrid-low-level-deep']['recall'])
    wh_low_level_deep['diversity'].append(result[1]['weighted-hybrid-low-level-deep']['diversity'])
    wh_low_level_deep['rankscore'].append(result[1]['weighted-hybrid-low-level-deep']['rankscore'])
    wh_low_level_deep['f1'].append(result[1]['weighted-hybrid-low-level-deep']['f1'])

    wh_low_level_item_collaborative['precision'].append(result[1]['weighted-hybrid-low-level-item-collaborative']['precision'])
    wh_low_level_item_collaborative['recall'].append(result[1]['weighted-hybrid-low-level-item-collaborative']['recall'])
    wh_low_level_item_collaborative['diversity'].append(result[1]['weighted-hybrid-low-level-item-collaborative']['diversity'])
    wh_low_level_item_collaborative['rankscore'].append(result[1]['weighted-hybrid-low-level-item-collaborative']['rankscore'])
    wh_low_level_item_collaborative['f1'].append(result[1]['weighted-hybrid-low-level-item-collaborative']['f1'])

    wh_low_level_user_collaborative['precision'].append(result[1]['weighted-hybrid-low-level-user-collaborative']['precision'])
    wh_low_level_user_collaborative['recall'].append(result[1]['weighted-hybrid-low-level-user-collaborative']['recall'])
    wh_low_level_user_collaborative['diversity'].append(result[1]['weighted-hybrid-low-level-user-collaborative']['diversity'])
    wh_low_level_user_collaborative['rankscore'].append(result[1]['weighted-hybrid-low-level-user-collaborative']['rankscore'])
    wh_low_level_user_collaborative['f1'].append(result[1]['weighted-hybrid-low-level-user-collaborative']['f1'])

    wh_item_collaborative_deep['precision'].append(result[1]['weighted-hybrid-item-collaborative-deep']['precision'])
    wh_item_collaborative_deep['recall'].append(result[1]['weighted-hybrid-item-collaborative-deep']['recall'])
    wh_item_collaborative_deep['diversity'].append(result[1]['weighted-hybrid-item-collaborative-deep']['diversity'])
    wh_item_collaborative_deep['rankscore'].append(result[1]['weighted-hybrid-item-collaborative-deep']['rankscore'])
    wh_item_collaborative_deep['f1'].append(result[1]['weighted-hybrid-item-collaborative-deep']['f1'])

    wh_item_collaborative_user_collaborative['precision'].append(result[1]['weighted-hybrid-item-collaborative-user-collaborative']['precision'])
    wh_item_collaborative_user_collaborative['recall'].append(result[1]['weighted-hybrid-item-collaborative-user-collaborative']['recall'])
    wh_item_collaborative_user_collaborative['diversity'].append(result[1]['weighted-hybrid-item-collaborative-user-collaborative']['diversity'])
    wh_item_collaborative_user_collaborative['rankscore'].append(result[1]['weighted-hybrid-item-collaborative-user-collaborative']['rankscore'])
    wh_item_collaborative_user_collaborative['f1'].append(result[1]['weighted-hybrid-item-collaborative-user-collaborative']['f1'])

    wh_deep_user_collaborative['precision'].append(result[1]['weighted-hybrid-deep-user-collaborative']['precision'])
    wh_deep_user_collaborative['recall'].append(result[1]['weighted-hybrid-deep-user-collaborative']['recall'])
    wh_deep_user_collaborative['diversity'].append(result[1]['weighted-hybrid-deep-user-collaborative']['diversity'])
    wh_deep_user_collaborative['rankscore'].append(result[1]['weighted-hybrid-deep-user-collaborative']['rankscore'])
    wh_deep_user_collaborative['f1'].append(result[1]['weighted-hybrid-deep-user-collaborative']['f1'])


print "Sinopse+UC", wh_synopsis_user_collaborative_mae
print "Tags+UC", wh_tags_user_collaborative_mae
print "LL+UC", wh_low_level_user_collaborative_mae
print "IC+UC", wh_item_collaborative_user_collaborative_mae
print "D+UC", wh_deep_user_collaborative_mae
exit()

linewidth = 1.1
list_styles = ['solid', 'dashed', 'dotted', 'dashdot']

# Precision
# plt.plot(listing, wh_synopsis_tags['precision'], linewidth=linewidth, label='Sinopse+Tags')
# plt.plot(listing, wh_synopsis_low_level['precision'], markersize=7, linewidth=linewidth, label='Sinopse+Baixo Nivel')
# plt.plot(listing, wh_synopsis_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Sinopse+FC-I')
# plt.plot(listing, wh_synopsis_deep['precision'], markersize=7, linewidth=linewidth, label='Sinopse+RNC')
plt.plot(listing, wh_synopsis_user_collaborative['precision'], markersize=7, linewidth=linewidth, label='Sinopse+FC-U')
# plt.plot(listing, wh_tags_low_level['precision'], markersize=7, linewidth=linewidth, label='Tags+Baixo Nivel')
# plt.plot(listing, wh_tags_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Tags+FC-I')
# plt.plot(listing, wh_tags_deep['precision'], markersize=7, linewidth=linewidth, label='Tags+RNC')
plt.plot(listing, wh_tags_user_collaborative['precision'], markersize=7, linewidth=linewidth, label='Tags+FC-U')
# plt.plot(listing, wh_low_level_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-I')
# plt.plot(listing, wh_low_level_deep['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+RNC')
plt.plot(listing, wh_low_level_user_collaborative['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-U')
# plt.plot(listing, wh_item_collaborative_deep['precision'], markersize=7, linewidth=linewidth, label='FC-I+RNC')
plt.plot(listing, wh_item_collaborative_user_collaborative['precision'], markersize=7, linewidth=linewidth, label='FC-I+FC-U')
plt.plot(listing, wh_deep_user_collaborative['precision'], markersize=7, linewidth=linewidth, label='RNC+FC-U')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Precision', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('precision_wh.pdf', bbox_inches='tight')
plt.close()

# Recall
# plt.plot(listing, wh_synopsis_tags['precision'], linewidth=linewidth, label='Sinopse+Tags')
# plt.plot(listing, wh_synopsis_low_level['precision'], markersize=7, linewidth=linewidth, label='Sinopse+Baixo Nivel')
# plt.plot(listing, wh_synopsis_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Sinopse+FC-I')
# plt.plot(listing, wh_synopsis_deep['precision'], markersize=7, linewidth=linewidth, label='Sinopse+RNC')
plt.plot(listing, wh_synopsis_user_collaborative['recall'], markersize=7, linewidth=linewidth, label='Sinopse+FC-U')
# plt.plot(listing, wh_tags_low_level['precision'], markersize=7, linewidth=linewidth, label='Tags+Baixo Nivel')
# plt.plot(listing, wh_tags_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Tags+FC-I')
# plt.plot(listing, wh_tags_deep['precision'], markersize=7, linewidth=linewidth, label='Tags+RNC')
plt.plot(listing, wh_tags_user_collaborative['recall'], markersize=7, linewidth=linewidth, label='Tags+FC-U')
# plt.plot(listing, wh_low_level_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-I')
# plt.plot(listing, wh_low_level_deep['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+RNC')
plt.plot(listing, wh_low_level_user_collaborative['recall'], markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-U')
# plt.plot(listing, wh_item_collaborative_deep['precision'], markersize=7, linewidth=linewidth, label='FC-I+RNC')
plt.plot(listing, wh_item_collaborative_user_collaborative['recall'], markersize=7, linewidth=linewidth, label='FC-I+FC-U')
plt.plot(listing, wh_deep_user_collaborative['recall'], markersize=7, linewidth=linewidth, label='RNC+FC-U')

# plt.plot(listing, user_collaborative['recall'], linewidth=linewidth, label='User-Collaborative')
# plt.plot(listing, deep['recall'], markersize=7, linewidth=linewidth, label='DeepRecVis')
# plt.plot(listing, item_collaborative['recall'], linewidth=linewidth, label='Item-Collaborative')
# plt.plot(listing, weighted_hybrid['recall'], linewidth=linewidth, label='weighted-hybrid')
# plt.plot(listing, low_level['recall'], linewidth=linewidth, markersize=10, label='Low-Level')
# plt.plot(listing, wh_collaborative['recall'], linewidth=linewidth, label='weighted-hybrid-collaborative')
# plt.plot(listing, wh_item_content['recall'], linewidth=linewidth, label='weighted-hybrid-content-item')

plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Recall', fontsize=13)
plt.xlabel('N', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('recall_wh.pdf', bbox_inches='tight')
plt.close()

# Diversity
# plt.plot(listing, wh_synopsis_tags['precision'], linewidth=linewidth, label='Sinopse+Tags')
# plt.plot(listing, wh_synopsis_low_level['precision'], markersize=7, linewidth=linewidth, label='Sinopse+Baixo Nivel')
# plt.plot(listing, wh_synopsis_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Sinopse+FC-I')
# plt.plot(listing, wh_synopsis_deep['precision'], markersize=7, linewidth=linewidth, label='Sinopse+RNC')
plt.plot(listing, wh_synopsis_user_collaborative['diversity'], '<-', markersize=7, linewidth=linewidth, label='Sinopse+FC-U')
# plt.plot(listing, wh_tags_low_level['precision'], markersize=7, linewidth=linewidth, label='Tags+Baixo Nivel')
# plt.plot(listing, wh_tags_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Tags+FC-I')
# plt.plot(listing, wh_tags_deep['precision'], markersize=7, linewidth=linewidth, label='Tags+RNC')
plt.plot(listing, wh_tags_user_collaborative['diversity'], '->', markersize=7, linewidth=linewidth, label='Tags+FC-U')
# plt.plot(listing, wh_low_level_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-I')
# plt.plot(listing, wh_low_level_deep['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+RNC')
plt.plot(listing, wh_low_level_user_collaborative['diversity'], '-+', markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-U')
# plt.plot(listing, wh_item_collaborative_deep['precision'], markersize=7, linewidth=linewidth, label='FC-I+RNC')
plt.plot(listing, wh_item_collaborative_user_collaborative['diversity'], '--', markersize=7, linewidth=linewidth, label='FC-I+FC-U')
plt.plot(listing, wh_deep_user_collaborative['diversity'], markersize=7, linewidth=linewidth, label='RNC+FC-U')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('Diversidade', fontsize=13)
plt.xlabel('N', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('diversity_wh.pdf', bbox_inches='tight')
plt.close()

# Rankscore
# plt.plot(listing, wh_synopsis_tags['precision'], linewidth=linewidth, label='Sinopse+Tags')
# plt.plot(listing, wh_synopsis_low_level['precision'], markersize=7, linewidth=linewidth, label='Sinopse+Baixo Nivel')
# plt.plot(listing, wh_synopsis_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Sinopse+FC-I')
# plt.plot(listing, wh_synopsis_deep['precision'], markersize=7, linewidth=linewidth, label='Sinopse+RNC')
plt.plot(listing, wh_synopsis_user_collaborative['rankscore'], markersize=7, linewidth=linewidth, label='Sinopse+FC-U')
# plt.plot(listing, wh_tags_low_level['precision'], markersize=7, linewidth=linewidth, label='Tags+Baixo Nivel')
# plt.plot(listing, wh_tags_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Tags+FC-I')
# plt.plot(listing, wh_tags_deep['precision'], markersize=7, linewidth=linewidth, label='Tags+RNC')
plt.plot(listing, wh_tags_user_collaborative['rankscore'], markersize=7, linewidth=linewidth, label='Tags+FC-U')
# plt.plot(listing, wh_low_level_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-I')
# plt.plot(listing, wh_low_level_deep['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+RNC')
plt.plot(listing, wh_low_level_user_collaborative['rankscore'], markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-U')
# plt.plot(listing, wh_item_collaborative_deep['precision'], markersize=7, linewidth=linewidth, label='FC-I+RNC')
plt.plot(listing, wh_item_collaborative_user_collaborative['rankscore'], markersize=7, linewidth=linewidth, label='FC-I+FC-U')
plt.plot(listing, wh_deep_user_collaborative['rankscore'], markersize=7, linewidth=linewidth, label='RNC+FC-U')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('rankscore', fontsize=13)
plt.xlabel('Iteration', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('rankscore.pdf', bbox_inches='tight')
plt.close()

# F1
# plt.plot(listing, wh_synopsis_tags['precision'], linewidth=linewidth, label='Sinopse+Tags')
# plt.plot(listing, wh_synopsis_low_level['precision'], markersize=7, linewidth=linewidth, label='Sinopse+Baixo Nivel')
# plt.plot(listing, wh_synopsis_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Sinopse+FC-I')
# plt.plot(listing, wh_synopsis_deep['precision'], markersize=7, linewidth=linewidth, label='Sinopse+RNC')
plt.plot(listing, wh_synopsis_user_collaborative['f1'], '<-', markersize=7, linewidth=linewidth, label='Sinopse+FC-U')
# plt.plot(listing, wh_tags_low_level['precision'], markersize=7, linewidth=linewidth, label='Tags+Baixo Nivel')
# plt.plot(listing, wh_tags_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Tags+FC-I')
# plt.plot(listing, wh_tags_deep['precision'], markersize=7, linewidth=linewidth, label='Tags+RNC')
plt.plot(listing, wh_tags_user_collaborative['f1'], '->', markersize=7, linewidth=linewidth, label='Tags+FC-U')
# plt.plot(listing, wh_low_level_item_collaborative['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-I')
# plt.plot(listing, wh_low_level_deep['precision'], markersize=7, linewidth=linewidth, label='Baixo Nivel+RNC')
plt.plot(listing, wh_low_level_user_collaborative['f1'], '-+', markersize=7, linewidth=linewidth, label='Baixo Nivel+FC-U')
# plt.plot(listing, wh_item_collaborative_deep['precision'], markersize=7, linewidth=linewidth, label='FC-I+RNC')
plt.plot(listing, wh_item_collaborative_user_collaborative['f1'], '--', markersize=7, linewidth=linewidth, label='FC-I+FC-U')
plt.plot(listing, wh_deep_user_collaborative['f1'], markersize=7, linewidth=linewidth, label='RNC+FC-U')
plt.grid(True)
plt.xlim((2, 15))
plt.ylabel('F1', fontsize=13)
plt.xlabel('N', fontsize=13)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
          fancybox=True, shadow=True, ncol=2, fontsize=13)
plt.savefig('f1_wh.pdf', bbox_inches='tight')
plt.close()


plt.rcdefaults()
# fig, ax = plt.subplots()

# Example data
# metrics = ('Low-Level', 'DeepRecVis', 'weighted-hybrid-content-item', 'Item-Collaborative', 'weighted-hybrid',
#            'weighted-hybrid-collaborative', 'User-Collaborative')
metrics = ('Sinopse+FC-U', 'Baixo Nivel+FC-U', 'Tags+FC-U', 'RNC+FC-U', 'FC-I+FC-U')
y_pos = np.arange(len(metrics))
# mae = np.array([ll_mae, d_mae, hci_mae, ic_mae, h_mae, hc_mae, uc_mae])
mae = np.array([wh_synopsis_user_collaborative_mae, wh_low_level_user_collaborative_mae, wh_tags_user_collaborative_mae,
                wh_deep_user_collaborative_mae, wh_item_collaborative_user_collaborative_mae])
_qt_methods_to_compare = 5
index = np.arange(_qt_methods_to_compare)
bar_width = 0.35

plt.barh(y_pos, mae, align='center', color='darkblue', ecolor='black', height=0.4)
# plt.figure(figsize=(10, 5))
# plt.xlabel('Strategy')
# plt.ylabel('Mean Absolute Error')
plt.title('MAE')
plt.yticks(index, ('Sinopse+FC-U', 'Baixo Nivel+FC-U', 'Tags+FC-U', 'RNC+FC-U', 'FC-I+FC-U'))

plt.savefig('mae_wh.pdf', bbox_inches='tight')
plt.show()
