# from pathlib import Path
# import numpy as np 
# import pandas as pd 
# import scipy.sparse as sp

# import torch.utils.data as data

# import config

# import random
# # random = np.random.default_rng()

# class MovieLen1MDataset(data.Dataset):
# 	file_names = {
# 		"train_ratings": "ml-1m.train.rating",
# 		"test_ratings": "ml-1m.test.rating",
# 		"test_negative": "ml-1m.test.negative",
# 	}
# 	meta = {
# 		'num_users': 6040,
# 		'num_items': 3706,
# 	}

# 	def __init__(self, root, train=False, num_negatives=4) -> None:
# 		super().__init__()
# 		self.root = Path(root)
# 		self._train = train
# 		self.num_negatives = num_negatives
# 		if train:
# 			self.rating_df = self._load_files()
# 			self.num_users = self.rating_df['user'].max() + 1
# 			self.num_items = self.rating_df['item'].max() + 1
# 			self.neg_item_dict, self.user_interaction_count = self._get_neg_items(self.rating_df)
# 			self.sample_negatives()
# 		else:
# 			self.data = self._load_files()

		
			

# 	def _get_neg_items(self, rating_df: pd.DataFrame):
# 		"""return all negative items & 100 sampled negative items"""
# 		item_pool = set(rating_df['item'].unique())
# 		interactions = rating_df.groupby('user')['item'].apply(set).reset_index().rename(
# 			columns={'item': 'pos_items'})
# 		user_interaction_count = interactions[['user',]]
# 		user_interaction_count['num_interaction'] = interactions['pos_items'].apply(len)
# 		user_interaction_count = dict(zip(user_interaction_count['user'].values, user_interaction_count['num_interaction'].values.tolist()))

# 		interactions['neg_items'] = interactions['pos_items'].apply(lambda x: (list(item_pool - x)))
# 		neg_item_dict = dict(zip(interactions['user'].values, interactions['neg_items'].values))
# 		return neg_item_dict, user_interaction_count
	
# 	def _load_files(self):
# 		if self._train: 
# 			train_df = pd.read_csv(
# 				self.root / self.file_names['train_ratings'], 
# 				sep='\t', header=None, names=['user', 'item', 'rating'], 
# 				usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32}
# 			)
# 			# print("Num 0 rating", len(train_df[train_df['rating'] == 0]))
# 			train_df = train_df[train_df['rating'] > 0]
# 			train_df['rating'] = 1.0
# 			return train_df 
# 		else:
# 			test_data = []
# 			with open(self.root / self.file_names['test_negative'], 'r') as fd:
# 				line = fd.readline()
# 				while line != None and line != '':
# 					arr = line.split('\t')
# 					u, pos_item = eval(arr[0])
# 					test_data.append([u, pos_item, 1.0])
# 					for i in arr[1:]:
# 						test_data.append([u, int(i), 0.0])
# 					line = fd.readline()
# 			return test_data
	
# 	def _sample_nagatives(self):
# 		assert self._train, 'no need to sampling when testing'

# 		num_negatives = self.num_negatives
		
# 		# Sampling negative examples 
# 		neg_samples = {}
# 		for u, neg_items in self.neg_item_dict.items():
# 			sample_size = num_negatives*self.user_interaction_count[u]
# 			neg_samples[u] = []
# 			while int(sample_size) >= len(neg_items):
# 				neg_samples[u] += neg_items
# 				sample_size -= len(neg_items)
# 			neg_samples[u] += random.sample(neg_items, sample_size)

# 		neg_rating_df = pd.DataFrame.from_records(list(neg_samples.items()), columns=['user', 'neg_sample'])

# 		# neg_rating_df = self.rarting_df[['user', 'rating']]
# 		# neg_rating_df['neg_sample'] = self.rarting_df['user'].map(lambda u: random.sample(self.neg_item_dict[u], num_negatives))

# 		neg_rating_df = neg_rating_df.explode('neg_sample').reset_index(drop=True)
# 		neg_rating_df['rating'] = 0.0
# 		# neg_rating_df = neg_rating_df[['user', 'neg_sample', 'rating']]
# 		neg_rating_df.columns = ['user', 'item', 'rating']
		
# 		train_rating_df = pd.concat([self.rating_df, neg_rating_df], ignore_index=True)
# 		return train_rating_df
	
# 	def sample_negatives(self):
# 		train_rating_df = self._sample_nagatives()
# 		# train_rarting_df = train_rarting_df.sample(frac=1).reset_index(drop=True)
# 		# self.data = self.rating_df.values.tolist() + neg_rating_df.values.tolist()
# 		self.data = train_rating_df.values.tolist() 
	
# 	def __len__(self):
# 		if not self._train:
# 			return len(self.data)
# 		else:
# 			return len(self.rating_df) * (self.num_negatives + 1)

# 	def __getitem__(self, idx):
# 		return self.data[idx] # user, item ,label


# def load_all(test_num=100):
# 	""" We load all the three file here to save time in each epoch. """
# 	train_data = pd.read_csv(
# 		config.train_rating, 
# 		sep='\t', header=None, names=['user', 'item'], 
# 		usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

# 	user_num = train_data['user'].max() + 1
# 	item_num = train_data['item'].max() + 1

# 	train_data = train_data.values.tolist()

# 	# load ratings as a dok matrix
# 	train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
# 	for x in train_data:
# 		train_mat[x[0], x[1]] = 1.0

# 	test_data = []
# 	with open(config.test_negative, 'r') as fd:
# 		line = fd.readline()
# 		while line != None and line != '':
# 			arr = line.split('\t')
# 			u = eval(arr[0])[0]
# 			test_data.append([u, eval(arr[0])[1]])
# 			for i in arr[1:]:
# 				test_data.append([u, int(i)])
# 			line = fd.readline()
# 	return train_data, test_data, user_num, item_num, train_mat


# class NCFData(data.Dataset):
# 	def __init__(self, features, 
# 				num_item, train_mat=None, num_ng=0, is_training=None):
# 		super(NCFData, self).__init__()
# 		""" Note that the labels are only useful when training, we thus 
# 			add them in the ng_sample() function.
# 		"""
# 		self.features_ps = features
# 		self.num_item = num_item
# 		self.train_mat = train_mat
# 		self.num_ng = num_ng
# 		self.is_training = is_training
# 		self.labels = [0 for _ in range(len(features))]

# 	def ng_sample(self):
# 		assert self.is_training, 'no need to sampling when testing'

# 		self.features_ng = []
# 		for x in self.features_ps:
# 			u = x[0]
# 			for t in range(self.num_ng):
# 				j = np.random.randint(self.num_item)
# 				while (u, j) in self.train_mat:
# 					j = np.random.randint(self.num_item)
# 				self.features_ng.append([u, j])

# 		labels_ps = [1 for _ in range(len(self.features_ps))]
# 		labels_ng = [0 for _ in range(len(self.features_ng))]

# 		self.features_fill = self.features_ps + self.features_ng
# 		self.labels_fill = labels_ps + labels_ng

# 	def __len__(self):
# 		return (self.num_ng + 1) * len(self.labels)

# 	def __getitem__(self, idx):
# 		features = self.features_fill if self.is_training \
# 					else self.features_ps
# 		labels = self.labels_fill if self.is_training \
# 					else self.labels

# 		user = features[idx][0]
# 		item = features[idx][1]
# 		label = labels[idx]
# 		return user, item ,label
		
# if __name__ == "__main__":
# 	from time import time
# 	ds = MovieLen1MDataset("./Data", train=True)
# 	start = time()
# 	ds.sample_negatives(2)
# 	print(time() - start)
# 	print(ds[0])
# 	print(ds.train_rarting_df.head(20))


from pathlib import Path
import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data
from recbole.data import dataset
from recbole.config import Config

import config

import random
# random = np.random.default_rng()

class MovieLen1MDataset(data.Dataset):
	file_names = {
		"train_ratings": "ml-1m.train.rating",
		"test_ratings": "ml-1m.test.rating",
		"test_negative": "ml-1m.test.negative",
	}
	meta = {
		'num_users': 6040,
		'num_items': 3706,
	}

	def __init__(self, root, train=False, num_negatives=4) -> None:
		super().__init__()
		self.root = Path(root)
		self._train = train
		self.num_negatives = num_negatives
		if train:
			self.rating_df = self._load_files()
			self.num_users = self.rating_df['user'].max() + 1
			self.num_items = self.rating_df['item'].max() + 1
			self.neg_item_dict, self.user_interaction_count = self._get_neg_items(self.rating_df)
			self.sample_negatives()
		else:
			self.data = self._load_files()
			self.num_negatives = 99

		
			

	def _get_neg_items(self, rating_df: pd.DataFrame):
		"""return all negative items & 100 sampled negative items"""
		item_pool = set(rating_df['item'].unique())
		interactions = rating_df.groupby('user')['item'].apply(set).reset_index().rename(
			columns={'item': 'pos_items'})
		user_interaction_count = interactions[['user',]]
		user_interaction_count['num_interaction'] = interactions['pos_items'].apply(len)
		user_interaction_count = dict(zip(user_interaction_count['user'].values, user_interaction_count['num_interaction'].values.tolist()))

		interactions['neg_items'] = interactions['pos_items'].apply(lambda x: (list(item_pool - x)))
		neg_item_dict = dict(zip(interactions['user'].values, interactions['neg_items'].values))
		return neg_item_dict, user_interaction_count
	
	def _load_files(self):
		if self._train: 
			train_df = pd.read_csv(
				self.root / self.file_names['train_ratings'], 
				sep='\t', header=None, names=['user', 'item', 'rating'], 
				usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32}
			)
			# print("Num 0 rating", len(train_df[train_df['rating'] == 0]))
			train_df = train_df[train_df['rating'] > 0]
			train_df['rating'] = 1.0
			return train_df 
		else:
			test_data = []
			with open(self.root / self.file_names['test_negative'], 'r') as fd:
				line = fd.readline()
				while line != None and line != '':
					arr = line.split('\t')
					u, pos_item = eval(arr[0])
					test_data.append([u, pos_item, 1.0])
					for i in arr[1:]:
						test_data.append([u, int(i), 0.0])
					line = fd.readline()
			return test_data
	
	def _sample_nagatives(self):
		assert self._train, 'no need to sampling when testing'

		num_negatives = self.num_negatives
		
		# Sampling negative examples 
		neg_samples = {}
		for u, neg_items in self.neg_item_dict.items():
			sample_size = num_negatives*self.user_interaction_count[u]
			neg_samples[u] = []
			while int(sample_size) >= len(neg_items):
				neg_samples[u] += neg_items
				sample_size -= len(neg_items)
			neg_samples[u] += random.sample(neg_items, sample_size)


		neg_rating_df = pd.DataFrame.from_records(list(neg_samples.items()), columns=['user', 'neg_sample'])
		# neg_rating_df = self.rarting_df[['user', 'rating']]
		# neg_rating_df['neg_sample'] = self.rarting_df['user'].map(lambda u: random.sample(self.neg_item_dict[u], num_negatives))

		neg_rating_df = neg_rating_df.explode('neg_sample').reset_index(drop=True)
		neg_rating_df['rating'] = 0.0
		# neg_rating_df = neg_rating_df[['user', 'neg_sample', 'rating']]
		neg_rating_df.columns = ['user', 'item', 'rating']
		# neg_rating_df = neg_rating_df.astype({'user': 'int64', 'item':'int64'})
		
		train_rating_df = pd.concat([self.rating_df, neg_rating_df], ignore_index=True)
		return train_rating_df
	
	def sample_negatives(self):
		train_rating_df = self._sample_nagatives()
		# train_rarting_df = train_rarting_df.sample(frac=1).reset_index(drop=True)
		# self.data = self.rating_df.values.tolist() + neg_rating_df.values.tolist()
		self.data = train_rating_df.values.tolist() 
	
	def __len__(self):
		return len(self.data)
		# if not self._train:
		# 	return len(self.data)
		# else:
		# 	return len(self.rating_df) * (self.num_negatives + 1)

	def __getitem__(self, idx):
		return self.data[idx] # user, item ,label


class FedMovieLen1MDataset(MovieLen1MDataset):
    def sample_negatives(self):
        self.train_rating_data = self._sample_nagatives()
    
    def set_client(self, cid):
        self.cid = cid
        self.data = self.train_rating_data[self.train_rating_data['user'] == self.cid].values.tolist()
    
    def __len__(self):
        return len(self.data)

class PinterestDataset(MovieLen1MDataset):
	file_names = {
		"train_ratings": "pinterest-20.train.rating",
		"test_ratings": "pinterest-20.test.rating",
		"test_negative": "pinterest-20.test.negative",
	}
	meta = {
		'num_users': 55187,
		'num_items': 9916,
	}

class FedPinterestDataset(PinterestDataset):
	def sample_negatives(self):
		self.train_rating_data = self._sample_nagatives()
	
	def set_client(self, cid):
		self.cid = cid
		self.data = self.train_rating_data[self.train_rating_data['user'] == self.cid].values.tolist()
	
	def __len__(self):
		return len(self.data)

class LastFM2kDataset(MovieLen1MDataset):
	def __init__(self, root, train=False, num_negatives=4) -> None:
		# super().__init__(root, train, num_negatives)
		self.num_negatives = num_negatives
		self._train = train
		ds, df = self._get_df()
		df['rating'] = 1.0
		grouped_inter_feat_index = ds._grouped_index(
            ds.inter_feat["user_id"].values
        )
		next_index = ds._split_index_by_leave_one_out(
			grouped_inter_feat_index, leave_one_num=1
		)
		df['user'] = df['user'].apply(int)
		df['item'] = df['item'].apply(int)
		
		user_reid_dict = {u: i for i,u in enumerate(df['user'].unique())}
		df['user'] = df['user'].apply(lambda u: user_reid_dict[u])

		item_reid_dict = {i: j for j, i in enumerate(df['item'].unique())}
		df['item'] = df['item'].apply(lambda i: item_reid_dict[i])

		self.neg_item_dict, self.user_interaction_count = self._get_neg_items(df)
		for k, v in self.user_interaction_count.items():
			self.user_interaction_count[k] = int(v) - 1

		self.num_users = df['user'].max() + 1
		self.num_items = df['item'].max() + 1
		if train:
			train_ds = df.iloc[next_index[0]]
			train_ds = train_ds[['user', 'item', 'rating']].reset_index(drop=True)
			self.rating_df = train_ds
			self.sample_negatives()
		else:
			test_ds = df.iloc[next_index[1]]
			test_ds = test_ds[['user', 'item', 'rating']].reset_index(drop=True)
			self.num_negatives = 99
			data = []
			random.seed(42)
			for index, row in test_ds.iterrows():
				u = int(row['user'])
				data.append([u, int(row['item']), 1.0])
				neg_sample = random.sample(self.neg_item_dict[u], self.num_negatives)
				assert len(neg_sample) == self.num_negatives
				for i in neg_sample:
					data.append([u, int(i), 0.0])
			# self.rating_df = test_ds
			self.data = tuple(data)
			# print(self.data[:10])
			# self.sample_negatives()
	
	def _get_df(self):
		dataset_name = 'lastfm'
		self.cfg = Config(model="BPR", dataset=dataset_name, config_dict={'ITEM_ID_FIELD': 'artist_id', 'LABEL_FIELD': 'label', 'NEG_PREFIX': 'neg_','USER_ID_FIELD': 'user_id','field_separator': '\t','load_col': {'inter': ['user_id', 'artist_id']},'seq_separator': ' '} )
		self.cfg["eval_args"]["split"] = {'LS': 'test_only'}
		self.cfg["user_inter_num_interval"] = '[5,inf]'
		self.cfg["item_inter_num_interval"] = '[1,inf]'
		ds = dataset.Dataset(self.cfg)
		df = ds.inter_feat.copy()
		df.rename(columns={'user_id': 'user', 'artist_id': 'item'}, inplace=True)
		return ds, df

class FedLastfmDataset(LastFM2kDataset):
	def sample_negatives(self):
		self.train_rating_data = self._sample_nagatives()
	
	def set_client(self, cid):
		self.cid = cid
		self.data = self.train_rating_data[self.train_rating_data['user'] == self.cid].values.tolist()
	
	def __len__(self):
		return len(self.data)

class AmazonVideoDataset(LastFM2kDataset):
	def __init__(self, root, train=False, num_negatives=4) -> None:
		super().__init__(root, train, num_negatives)
	
	def _get_df(self):
		self.cfg = Config(model="BPR", dataset='Amazon_Instant_Video', 
		    config_dict={'ITEM_ID_FIELD': 'item_id', 'LABEL_FIELD': 'label', 'NEG_PREFIX': 'neg_',
		   'USER_ID_FIELD': 'user_id','field_separator': '\t','load_col': {'inter': ['user_id', 'item_id']},'seq_separator': ' '} )
		self.cfg["eval_args"]["split"] = {'LS': 'test_only'}
		self.cfg["user_inter_num_interval"] = '[5,inf]'
		self.cfg["item_inter_num_interval"] = '[1,inf]'
		ds = dataset.Dataset(self.cfg)
		df = ds.inter_feat.copy()
		df.rename(columns={'user_id': 'user', 'item_id': 'item'}, inplace=True)
		return ds, df

class DoubanMovieDataset(LastFM2kDataset):
	def __init__(self, root, train=False, num_negatives=4) -> None:
		super().__init__(root, train, num_negatives)
	
	def _get_df(self):
		self.cfg = Config(model="BPR", dataset='douban', 
		    config_dict={'ITEM_ID_FIELD': 'item_id', 'LABEL_FIELD': 'label', 'NEG_PREFIX': 'neg_',
		   'USER_ID_FIELD': 'user_id','field_separator': '\t','load_col': {'inter': ['user_id', 'item_id']},'seq_separator': ' '} )
		self.cfg["eval_args"]["split"] = {'LS': 'test_only'}
		self.cfg["user_inter_num_interval"] = '[5,inf]'
		self.cfg["item_inter_num_interval"] = '[1,inf]'
		ds = dataset.Dataset(self.cfg)
		df = ds.inter_feat.copy()
		df.rename(columns={'user_id': 'user', 'item_id': 'item'}, inplace=True)
		return ds, df

if __name__ == "__main__":
	# from recbole.data.dataset import Dataset
	# cfg = Config(model="BPR", dataset='lastfm', config_dict={'ITEM_ID_FIELD': 'artist_id', 'LABEL_FIELD': 'label', 'NEG_PREFIX': 'neg_','USER_ID_FIELD': 'user_id','field_separator': '\t','load_col': {'inter': ['user_id', 'artist_id']},'seq_separator': ' '} )
	# cfg["eval_args"]["split"] = {'LS': 'test_only'}
	# print(cfg["eval_args"]["group_by"])
	# ds = Dataset(cfg)
	# df = ds.inter_feat.copy()
	# # print(ds.inter_feat)
	# # train, val, test = ds.build()

	# grouped_inter_feat_index = ds._grouped_index(
    #         ds.inter_feat["user_id"].values
    #     )
	# next_index = ds._split_index_by_leave_one_out(
	# 	grouped_inter_feat_index, leave_one_num=1
	# )
	# train_ds, test_ds = df.iloc[next_index[0]], df.iloc[next_index[1]]
	# train_ds = train_ds[['user_id', 'artist_id', 'rating']].reset_index(drop=True)
	# test_ds = test_ds[['user_id', 'artist_id', 'rating']].reset_index(drop=True)
	ds = LastFM2kDataset('data/lastfm-2k', train=True)
		# next_index = [next_index[0], [], next_index[1]]
	import IPython
	IPython.embed()

