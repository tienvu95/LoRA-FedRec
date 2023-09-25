from pathlib import Path
import numpy as np 
import pandas as pd 
import torch

from torch.utils.data import TensorDataset

import random
# random = np.random.default_rng()

def get_neg_items(rating_df: pd.DataFrame, test_inter_dict=None):
    """return all negative items & 100 sampled negative items"""
    item_pool = set(rating_df['item'].unique())
    interactions = rating_df.groupby('user')['item'].apply(set).reset_index().rename(
        columns={'item': 'pos_items'})
    user_interaction_count = interactions[['user',]]
    user_interaction_count['num_interaction'] = interactions['pos_items'].apply(len)
    user_interaction_count = dict(zip(user_interaction_count['user'].values, user_interaction_count['num_interaction'].values.tolist()))

    pos_item_dict = dict(zip(interactions['user'].values, interactions['pos_items'].values))

    # interactions['neg_items'] = interactions['pos_items'].apply(lambda x: (list(item_pool - x)))
    # neg_item_dict = dict(zip(interactions['user'].values, interactions['neg_items'].values))
    if test_inter_dict is not None:
        for u, test_items in test_inter_dict.items():
            # neg_item_dict[u] = list(set(neg_item_dict[u]).difference(test_items))
            # print(pos_item_dict[u])
            # pos_item_dict[u] = list(pos_item_dict[u].update(test_items))
            pos_item_dict[u].update(test_items)
    return item_pool, pos_item_dict, user_interaction_count

class RecDataModule():
    def __init__(self, root, num_train_negatives=4) -> None:
        self.root = Path(root)
        self.num_train_negatives = num_train_negatives
    
    def process_test_data(self, test_df):
        test_data = []
        for index, row in test_df.iterrows():
            u = row['user']
            for i in row['neg_sample']:
                test_data.append((int(u), int(i), 0.0))
            test_data.append((int(u), int(row['pos_item']), 1.0))
        # test_data = np.array(test_data)
        return test_data

    def setup(self):
        self.post_train_df = pd.read_csv(self.root / 'train.csv')
        self.test_df = pd.read_csv(self.root / 'test.csv')

        self.num_users = max(self.post_train_df['user'].max(), self.test_df['user'].max()) + 1
        self.num_items = max(self.post_train_df['item'].max(), self.test_df['pos_item'].max()) + 1

        self.test_df['neg_sample'] = self.test_df['neg_sample'].apply(lambda x: [int(s) for s in x[1:-1].split(',')])
        test_inter_dict = self.test_df.apply(lambda x: x['neg_sample'] + [x['pos_item']], axis=1)
        test_inter_dict = dict(zip(self.test_df['user'].values, test_inter_dict.values))

        if "v2" in str(self.root):
            self.val_df = pd.read_csv(self.root / 'val.csv')
            self.val_df['neg_sample'] = self.val_df['neg_sample'].apply(lambda x: [int(s) for s in x[1:-1].split(',')])
            
            self.num_users = max(self.num_users, self.val_df['user'].max() + 1)
            self.num_items = max(self.num_items, self.val_df['pos_item'].max() + 1)
            
            val_inter_dict = self.val_df.apply(lambda x: x['neg_sample'] + [x['pos_item']], axis=1)
            val_inter_dict = dict(zip(self.val_df['user'].values, val_inter_dict.values))
            for u, test_items in val_inter_dict.items():
                test_inter_dict[u] = set(test_inter_dict[u]).union(test_items)
            self.val_data = self.process_test_data(self.val_df)

        else:
            self.val_data = None

        self.item_pool, self.pos_item_dict, self.user_interaction_count = get_neg_items(self.post_train_df, test_inter_dict)

        self.test_data = self.process_test_data(self.test_df)

        self.item_pool = tuple(self.item_pool)
    
    def _get_neg_items_of_user(self, u):
        return self.item_pool - self.pos_item_dict[u]

    def _sample_neg_per_user(self, u, num_negatives):
        # Sampling negative examples 
        sample_size = num_negatives*self.user_interaction_count[u]
        # num_pos = len(self.pos_item_dict[u])
        # num_neg = len(self.item_pool) - num_pos

        # neg_items = list(self._get_neg_items_of_user(u))
        neg_samples = []
        while len(neg_samples) < sample_size:
            # k = min(sample_size - len(neg_samples), len(self.item_pool))
            # if k == len(self.item_pool):
            #     sample = self.item_pool
            # else:
            k = num_negatives
            sample = random.sample(self.item_pool, k)
            sample = set(sample) - self.pos_item_dict[u]
            neg_samples.extend(sample)
        neg_samples = neg_samples[:sample_size]
        # while int(sample_size) >= len(neg_items):
        #     neg_samples += neg_items
        #     sample_size -= len(neg_items)
        # neg_samples += random.sample(neg_items, sample_size)
        return neg_samples
    
    def _sample_neg_traindf(self, num_negatives):
        # Sampling negative examples 
        neg_samples = {}
        for u in range(len(self.pos_item_dict)):
            neg_samples[u] = self._sample_neg_per_user(u, num_negatives)
        neg_rating_df = pd.DataFrame.from_records(list(neg_samples.items()), columns=['user', 'neg_sample'])
        neg_rating_df = neg_rating_df.explode('neg_sample').reset_index(drop=True)
        neg_rating_df['rating'] = 0.0
        neg_rating_df.columns = ['user', 'item', 'rating']
        neg_rating_df['user'] = neg_rating_df['user'].astype(int)
        neg_rating_df['item'] = neg_rating_df['item'].astype(int)
        neg_rating_df['rating'] = neg_rating_df['rating'].astype(float)

        return neg_rating_df

    def train_dataset(self, num_negatives=None):
        if num_negatives is None:
            num_negatives = self.num_train_negatives
        neg_train_df = self._sample_neg_traindf(num_negatives)
        train_rating_df = pd.concat([self.post_train_df, neg_train_df], ignore_index=True)
        # train_rating_df = train_rating_df.sample(frac=1).reset_index(drop=True)
        
        users_tensor = torch.tensor(train_rating_df['user'].values, dtype=torch.long)
        items_tensor = torch.tensor(train_rating_df['item'].values, dtype=torch.long)
        ratings_tensor = torch.tensor(train_rating_df['rating'].values, dtype=torch.float)
        dataset = TensorDataset(users_tensor, items_tensor, ratings_tensor)
        return dataset

    def test_dataset(self):
        # test_tensor = torch.tensor(self.test_data, dtype=torch.float)
        # users_tensor = torch.tensor(self.test_data[:, 0], dtype=torch.long)
        # items_tensor = torch.tensor(self.test_data[:, 1], dtype=torch.long)
        # ratings_tensor = torch.tensor(self.test_data[:, 2], dtype=torch.float)
        # dataset = TensorDataset(users_tensor, items_tensor, ratings_tensor)
        return self.test_data

    def val_dataset(self):
        if self.val_data is None:
            return None
        # users_tensor = torch.tensor(self.val_data[:, 0], dtype=torch.long)
        # items_tensor = torch.tensor(self.val_data[:, 1], dtype=torch.long)
        # ratings_tensor = torch.tensor(self.val_data[:, 2], dtype=torch.float)
        # dataset = TensorDataset(users_tensor, items_tensor, ratings_tensor)
        return self.val_data

def get_datamodule(cfg):
    if cfg.DATA.name == "lastfm":
        root = cfg.DATA.root + "/lastfm"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "movielens":
        root = cfg.DATA.root + "/ml-1m"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "movielens-v2":
        root = cfg.DATA.root + "/ml-1m-v2"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "pinterest":
        root = cfg.DATA.root + "/pinterest"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "pinterest-v2":
        root = cfg.DATA.root + "/pinterest-v2"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "amazon-video":
        root = cfg.DATA.root + "/Amazon_Instant_Video"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "amazon-industry-and-science":
        root = cfg.DATA.root + "/amz_ins"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "foursq-ny":
        root = cfg.DATA.root + "/4sq-ny"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "book-crossing":
        root = cfg.DATA.root + "/book-crossing"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    else:
        raise ValueError
    return dm


