# import sqlite3
import cPickle as pickle

def save_obj(obj, name ):

	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_features(path):

	with open(path, 'rb') as f:
		return pickle.load(f)

if __name__ == '__main__':
	# feat = load_features('features/feat_128_loss_1.bin')
	feat = load_features('res_neurons_128_feat_1024_scenes_350.bin') # LSTM 128 imagenet
	# feat = load_features('res_neurons_32_feat_1024_scenes_350.bin') # GRU 32 imagenet
	# print 'nb trailer features', len(feat)

	# conn = sqlite3.connect("database.db")
	# c = conn.cursor()

	# print feat[1]

	for key, value in feat.iteritems():
		# print key, value.shape, value.mean()
		exit()
		# for feature in value:
			# conn.execute("INSERT INTO trailer_feature(trailerid, feature) VALUES(?, ?)", (int(key),float(feature),))
		# exit()

	# conn.commit()
	# conn.close()
