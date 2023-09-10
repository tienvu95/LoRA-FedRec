from pathlib import Path
import numpy as np 
import pandas as pd 
import scipy.sparse as sp
import torch

import torch.utils.data as data
from recbole.data import dataset, create_dataset, data_preparation
from recbole.config import Config
from torch.utils.data import TensorDataset, DataLoader

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

    def __init__(self, root, train=False, num_negatives=4, sample_negative=True) -> None:
        super().__init__()
        self.root = Path(root)
        self._train = train
        self.num_negatives = num_negatives
        self.test_data = self._load_files(train=False)
        self.test_inter_dict = {}
        for sample in self.test_data:
            u = sample[0]
            i = sample[1]
            if u in self.test_inter_dict:
                self.test_inter_dict[u].update({i})
            else:
                self.test_inter_dict[u] = {i}
        

        if train:
            self.rating_df = self._load_files(train=True)
            self.num_users = self.rating_df['user'].max() + 1
            self.num_items = self.rating_df['item'].max() + 1
            self.neg_item_dict, self.user_interaction_count = self._get_neg_items(self.rating_df)
            if sample_negative:
                self.sample_negatives()
        else:
            self.data = self.test_data
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
        for u, test_items in self.test_inter_dict.items():
            neg_item_dict[u] = list(set(neg_item_dict[u]).difference(test_items))
        return neg_item_dict, user_interaction_count
    
    def _load_files(self, train):
        if train: 
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


class RecDataModule():
    def __init__(self, root, num_train_negatives=4) -> None:
        self.root = Path(root)
        self.num_train_negatives = num_train_negatives
    
    def setup(self):
        self.post_train_df = pd.read_csv(self.root / 'train.csv')
        self.test_df = pd.read_csv(self.root / 'test.csv')
        self.num_users = max(self.post_train_df['user'].max(), self.test_df['user'].max()) + 1
        self.num_items = max(self.post_train_df['item'].max(), self.test_df['pos_item'].max()) + 1

        self.test_df['neg_sample'] = self.test_df['neg_sample'].apply(lambda x: [int(s) for s in x[1:-1].split(',')])
        test_inter_dict = self.test_df.apply(lambda x: x['neg_sample'] + [x['pos_item']], axis=1)
        self.item_pool, self.pos_item_dict, self.user_interaction_count = get_neg_items(self.post_train_df, test_inter_dict)

        self.test_data = []
        for index, row in self.test_df.iterrows():
            u = row['user']
            self.test_data.append((u, row['pos_item'], 1.0))
            for i in row['neg_sample']:
                self.test_data.append((u, i, 0.0))
        self.test_data = np.array(self.test_data)
    
    def _get_neg_items_of_user(self, u):
        return self.item_pool - self.pos_item_dict[u]

    def _sample_neg_per_user(self, u, num_negatives):
        # Sampling negative examples 
        neg_items = list(self._get_neg_items_of_user(u))
        sample_size = num_negatives*self.user_interaction_count[u]
        neg_samples = []
        while int(sample_size) >= len(neg_items):
            neg_samples += neg_items
            sample_size -= len(neg_items)
        neg_samples += random.sample(neg_items, sample_size)
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
        users_tensor = torch.tensor(self.test_data[:, 0], dtype=torch.long)
        items_tensor = torch.tensor(self.test_data[:, 1], dtype=torch.long)
        ratings_tensor = torch.tensor(self.test_data[:, 2], dtype=torch.float)
        dataset = TensorDataset(users_tensor, items_tensor, ratings_tensor)
        return dataset

def get_datamodule(cfg):
    if cfg.DATA.name == "lastfm":
        root = cfg.DATA.root + "/lastfm"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "movielens":
        root = cfg.DATA.root + "/ml-1m"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "pinterest":
        root = cfg.DATA.root + "/pinterest"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "amazon-video":
        root = cfg.DATA.root + "/Amazon_Instant_Video"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "amazon-industry-and-science":
        root = cfg.DATA.root + "/amz_ins"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    elif cfg.DATA.name == "foursq-ny":
        root = cfg.DATA.root + "/foursquare_NYC"
        dm = RecDataModule(root=root, num_train_negatives=cfg.DATA.num_negatives)
    else:
        raise ValueError
    return dm


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
    ds = DoubanMovieDataset('data/lastfm-2k', train=True)
        # next_index = [next_index[0], [], next_index[1]]
    import IPython
    IPython.embed()

