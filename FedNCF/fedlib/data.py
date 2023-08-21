import torch.utils.data as data
from rec.data import get_dataset

class FedDataModule(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
    
    def setup(self):
        self.train_dataset, self.test_dataset = get_dataset(self.cfg, sample_negative=False)
        self.num_users = self.train_dataset.num_users
        self.num_items = self.train_dataset.num_items
        self.sample_negative()
    
    def sample_negative(self):
        self.train_rating_data = self.train_dataset._sample_nagatives()

    def set_cid(self, cid):
        self.cid = cid
        cid_data = self.train_rating_data[self.train_rating_data['user'] == self.cid].values.tolist()
        self.train_dataset.data = cid_data
    
    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=2048, shuffle=False, num_workers=0)

    def train_dataloader(self, cid):
        self.set_cid(cid)
        return data.DataLoader(self.train_dataset, **self.cfg.DATALOADER)