from pathlib import Path
import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data

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
		
		train_rating_df = pd.concat([self.rating_df, neg_rating_df], ignore_index=True)
		return train_rating_df
	
	def sample_negatives(self):
		train_rating_df = self._sample_nagatives()
		# train_rarting_df = train_rarting_df.sample(frac=1).reset_index(drop=True)
		# self.data = self.rating_df.values.tolist() + neg_rating_df.values.tolist()
		self.data = train_rating_df.values.tolist() 
	
	def __len__(self):
		if not self._train:
			return len(self.data)
		else:
			return len(self.rating_df) * (self.num_negatives + 1)

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