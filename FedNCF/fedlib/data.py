import torch
import torch.utils.data as data
from rec.data import get_datamodule, RecDataModule
import pandas as pd

class FedDataModule(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.rec_datamodule: RecDataModule = get_datamodule(cfg)
    
    def setup(self):
        # self.train_dataset, self.test_dataset = get_dataset(self.cfg, sample_negative=False)
        # self.num_users = self.train_dataset.num_users
        # self.num_items = self.train_dataset.num_items
        # self.sample_negatives()
        self.rec_datamodule.setup()
        self.num_users = self.rec_datamodule.num_users
        self.num_items = self.rec_datamodule.num_items

    def train_dataset(self, cid_list):
        num_negatives = self.rec_datamodule.num_train_negatives
        all_post_train_df = self.rec_datamodule.post_train_df
        client_pos_traindf = all_post_train_df[all_post_train_df['user'].isin(cid_list)]
        
        client_neg_sample = {}
        for cid in cid_list:
            client_neg_sample[cid] = self.rec_datamodule._sample_neg_per_user(cid, num_negatives=num_negatives)
        # client_neg_sample = self.rec_datamodule._sample_neg_per_user(cid, num_negatives=num_negatives)
        client_neg_traindf = pd.DataFrame.from_records(list(client_neg_sample.items()), columns=['user', 'neg_sample'])
        client_neg_traindf = client_neg_traindf.explode('neg_sample').reset_index(drop=True)
        client_neg_traindf['rating'] = 0.0
        client_neg_traindf.columns = ['user', 'item', 'rating']
        client_neg_traindf['user'] = client_neg_traindf['user'].astype(int)
        client_neg_traindf['item'] = client_neg_traindf['item'].astype(int)
        client_neg_traindf['rating'] = client_neg_traindf['rating'].astype(float)

        train_rating_df = pd.concat([client_pos_traindf, client_neg_traindf], ignore_index=True)

        # print(train_rating_df.info())
        users_tensor = torch.tensor(train_rating_df['user'].values, dtype=torch.long)
        items_tensor = torch.tensor(train_rating_df['item'].values, dtype=torch.long)
        ratings_tensor = torch.tensor(train_rating_df['rating'].values, dtype=torch.float)
        dataset = data.TensorDataset(users_tensor, items_tensor, ratings_tensor)
        return dataset
    
    def test_dataloader(self):
        return data.DataLoader(self.rec_datamodule.test_dataset(), batch_size=1024, shuffle=False, num_workers=2)

    def train_dataloader(self, cid=None, for_eval=False):
        if cid is None:
            if for_eval:
                return data.DataLoader(self.rec_datamodule.train_dataset(), batch_size=1024, shuffle=False, num_workers=8)
            else:
                return data.DataLoader(self.rec_datamodule.train_dataset(), **self.cfg.DATALOADER)
        return data.DataLoader(self.train_dataset(cid), **self.cfg.DATALOADER)