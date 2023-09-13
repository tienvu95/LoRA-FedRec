from pathlib import Path
import numpy as np
import pandas as pd
import random
from rec import data
from recbole.data import dataset, create_dataset, data_preparation
from recbole.config import Config


def get_neg_items(rating_df: pd.DataFrame):
    """return all negative items & 100 sampled negative items"""
    item_pool = set(rating_df['item'].unique())
    interactions = rating_df.groupby('user')['item'].apply(set).reset_index().rename(
        columns={'item': 'pos_items'})
    user_interaction_count = dict(zip(interactions['user'].values, interactions['pos_items'].apply(len).values.tolist()))
    
    item_interaction_df = rating_df.groupby('item')['user'].count().reset_index().rename({'user': 'count'}, axis=1)
    print(item_interaction_df)
    item_interaction_count = dict(zip(item_interaction_df['item'].values, item_interaction_df['count'].values.tolist()))
    
    # interactions['neg_items'] = interactions['pos_items'].apply(lambda x: (list(item_pool - x)))
    # print(4)
    pos_item_dict = dict(zip(interactions['user'].values, interactions['pos_items'].values))
    print(5)
    return item_pool, pos_item_dict, user_interaction_count, item_interaction_count

def get_recbole_cfg(data_root, dataset_name):
    if dataset_name == 'lastfm':
        cfg = Config(model="BPR", 
                            dataset='lastfm', 
                            config_dict={'ITEM_ID_FIELD': 'artist_id', 
                                         'LABEL_FIELD': 'label', 
                                         'NEG_PREFIX': 'neg_',
                                         'USER_ID_FIELD': 'user_id',
                                         'field_separator': '\t',
                                         'load_col': {'inter': ['user_id', 'artist_id']},
                                         'seq_separator': ' '} )
        cfg["eval_args"]["split"] = {'LS': 'test_only'}
        cfg["user_inter_num_interval"] = '[5,inf]'
        cfg["item_inter_num_interval"] = '[1,inf]'
        cfg['data_path']= data_root + "/" + 'lastfm'
        cfg["eval_args"]["order"] = 'RO' # Random order
    elif dataset_name == 'amv':
        cfg = Config(model="BPR", dataset='Amazon_Instant_Video', 
            config_dict={'ITEM_ID_FIELD': 'item_id', 'LABEL_FIELD': 'label', 'TIME_FIELD':'timestamp', 'NEG_PREFIX': 'neg_',
           'USER_ID_FIELD': 'user_id','field_separator': '\t','load_col': {'inter': ['user_id', 'item_id', 'timestamp']},'seq_separator': ' '} )
        cfg["eval_args"]["split"] = {'LS': 'test_only'}
        cfg["user_inter_num_interval"] = '[5,inf]'
        cfg["item_inter_num_interval"] = '[1,inf]'
        cfg['data_path']= data_root + "/" + 'Amazon_Instant_Video'
        cfg["eval_args"]["order"] = 'TO' # Time order
        # cfg["eval_args"]["order"] = 'RO'
    elif dataset_name == 'amz_ins':
        cfg = Config(model="BPR", dataset='Amazon_Industrial_and_Scientific', 
            config_dict={'ITEM_ID_FIELD': 'item_id', 'LABEL_FIELD': 'label', 'TIME_FIELD':'timestamp', 'NEG_PREFIX': 'neg_',
           'USER_ID_FIELD': 'user_id','field_separator': '\t','load_col': {'inter': ['user_id', 'item_id', 'timestamp']},'seq_separator': ' '} )
        cfg["eval_args"]["split"] = {'LS': 'test_only'}
        cfg["user_inter_num_interval"] = '[5,inf]'
        cfg["item_inter_num_interval"] = '[1,inf]'
        cfg['data_path']= data_root + "/" + 'Amazon_Industrial_and_Scientific'
        cfg["eval_args"]["order"] = 'TO' # Time order
        # cfg["eval_args"]["order"] = 'RO'
    elif dataset_name == '4sq-ny':
        cfg = Config(model="BPR", dataset='foursquare_NYC', 
            config_dict={'ITEM_ID_FIELD': 'venue_id', 'LABEL_FIELD': 'label', 'NEG_PREFIX': 'neg_',
           'USER_ID_FIELD': 'user_id','field_separator': '\t','load_col': {'inter': ['user_id', 'venue_id', 'timestamp']},'seq_separator': ' '} )
        cfg["eval_args"]["split"] = {'LS': 'test_only'}
        cfg["user_inter_num_interval"] = '[5,inf]'
        cfg["item_inter_num_interval"] = '[1,inf]'
        cfg['data_path']= data_root + '/foursquare_NYC'
        cfg["eval_args"]["order"] = 'TO' # Time order
        # cfg["eval_args"]["order"] = 'RO'
    return cfg

def get_df_from_recbole(data_root, dataset_name):
    cfg = get_recbole_cfg(data_root, dataset_name)
    ds = dataset.Dataset(cfg)
    df = ds.inter_feat.copy()

    ordering_args = cfg["eval_args"]["order"]
    if ordering_args == "RO":
        df = df.sample(frac=1).reset_index(drop=True)
    elif ordering_args == "TO":
        time_field = cfg["TIME_FIELD"]
        df.sort_values(by=time_field, ascending=True, inplace=True)
    else:
        raise NotImplementedError(
            f"The ordering_method [{ordering_args}] has not been implemented."
        )

    print(df)
    # exit(0)
    df.rename(columns={cfg['USER_ID_FIELD']: 'user', cfg['ITEM_ID_FIELD']: 'item'}, inplace=True)
    df['user'] = df['user'].apply(int)
    df['item'] = df['item'].apply(int)
    df['rating'] = 1.0
    user_reid_dict = {j: i for i,j in enumerate(df['user'].unique())}
    item_reid_dict = {j: i for i,j in enumerate(df['item'].unique())}
    df['user'] = df['user'].apply(lambda u: user_reid_dict[u])
    df['item'] = df['item'].apply(lambda i: item_reid_dict[i])

    print(df.item.max(), df.item.nunique())
    
    return ds, df, cfg

def _split_index_by_leave_one_out(grouped_index, leave_one_num, item_interaction_count, df):
    """Split indexes by strategy leave one out.

    Args:
        grouped_index (list of list of int): Index to be split.
        leave_one_num (int): Number of parts whose length is expected to be ``1``.

    Returns:
        list: List of index that has been split.
    """
    # TODO: handle leave_one_num > 1 case
    item_interaction_count = item_interaction_count.copy()
    next_index = [[] for _ in range(leave_one_num + 1)]
    for index in grouped_index:
        index = list(index)
        tot_cnt = len(index)
        assert tot_cnt > 4
        legal_leave_one_num = min(leave_one_num, tot_cnt - 1)
        # pr = tot_cnt - legal_leave_one_num
        pr = tot_cnt - 1
        single_interaction_item = []
        for i in range(legal_leave_one_num):
            inter_id = index[pr]
            item_id = int(df.iloc[inter_id]['item'])
            while item_interaction_count[item_id] <= 1:
                single_interaction_item.append(inter_id)
                pr -= 1
                inter_id = index[pr]
                item_id = int(df.iloc[inter_id]['item'])
            item_interaction_count[item_id] -= 1
            next_index[-legal_leave_one_num + i].append(inter_id)
            pr -= 1

        next_index[0].extend(index[:pr + 1])
        next_index[0].extend(single_interaction_item)
    assert len(next_index[1]) > 0 and len(set(next_index[0]).intersection(next_index[1])) == 0
    return next_index

def gen_ds(dataset_name='lastfm', test_num_negatives=99, seed=42):
    ds, df, cfg = get_df_from_recbole(DATA_ROOT, dataset_name=dataset_name)
    item_pool, pos_item_dict, user_interaction_count, item_interaction_count = get_neg_items(df)
    
    index = {}
    for i, key in enumerate(df['user'].values):
        if key not in index:
            index[key] = [i]
        else:
            index[key].append(i)
    grouped_inter_feat_index = index.values()

    next_index = _split_index_by_leave_one_out(
        grouped_inter_feat_index, leave_one_num=1, item_interaction_count=item_interaction_count, df=df
    )

    train_df = df.iloc[next_index[0]]
    train_df = train_df[['user', 'item', 'rating']].reset_index(drop=True)

    test_df = df.iloc[next_index[1]]
    test_df = test_df[['user', 'item', 'rating']].reset_index(drop=True)
    test_data = []
    random.seed(seed)
    for index, row in test_df.iterrows():
        u = int(row['user'])
        pos_item = int(row['item'])
        user_neg_item_dict = list(item_pool - set(pos_item_dict[u]))
        neg_items = random.sample(user_neg_item_dict, test_num_negatives)
        assert len(neg_items) == test_num_negatives
        test_data.append([u, pos_item, neg_items])
    
    return train_df, test_data, cfg

def gen_movielen_files(root, train, prefix='ml-1m'):
    file_names = {
        "train_ratings": f"{prefix}.train.rating",
        "test_ratings": f"{prefix}.test.rating",
        "test_negative": f"{prefix}.test.negative",
    }
    if train: 
        train_df = pd.read_csv(
            root / file_names['train_ratings'], 
            sep='\t', header=None, names=['user', 'item', 'rating'], 
            usecols=[0, 1, 2], dtype={0: np.int32, 1: np.int32}
        )
        # print("Num 0 rating", len(train_df[train_df['rating'] == 0]))
        train_df = train_df[train_df['rating'] > 0]
        train_df['rating'] = 1.0
        return train_df 
    else:
        test_data = []
        with open(root / file_names['test_negative'], 'r') as fd:
            line = fd.readline()
            while line != None and line != '':
                arr = line.split('\t')
                u, pos_item = eval(arr[0])
                neg_sample = []
                for i in arr[1:]:
                    neg_sample.append(int(i))
                test_data.append([u, pos_item, neg_sample])
                line = fd.readline()
        test_df = pd.DataFrame(data=test_data, columns=['user', 'pos_item', 'neg_sample'])
        return test_df


DATA_ROOT = '../dataset'
dataset_name = 'lastfm'

if dataset_name == 'amz_ins':
    train_df, test_data, cfg = gen_ds(dataset_name='amz_ins')
    test_df = pd.DataFrame(data=test_data, columns=("user", "pos_item", "neg_sample"))
elif dataset_name == 'lastfm':
    train_df, test_data, cfg = gen_ds(dataset_name='lastfm')
    test_df = pd.DataFrame(data=test_data, columns=("user", "pos_item", "neg_sample"))
elif dataset_name == '4sq-ny':
    train_df, test_data, cfg = gen_ds(dataset_name='lastfm')
    test_df = pd.DataFrame(data=test_data, columns=("user", "pos_item", "neg_sample"))

print(train_df.item.max(), train_df.item.nunique())
# print(test_df.item.max(), test_df.pos_item.nunique())

print(set(range(train_df.item.max() + 1)) - set(train_df.item.unique()))


# DATA_ROOT = Path('../data/Data')


# train_df = gen_movielen_files(DATA_ROOT, train=True, prefix='pinterest-20')
# test_df = gen_movielen_files(DATA_ROOT, train=False,prefix='pinterest-20')
# data_path = cfg['data_path']

data_path = Path(f"../dataset/{dataset_name}")
if not data_path.exists():
    data_path.mkdir(parents=True)
train_df.to_csv(data_path / "train.csv", index=False)
test_df.to_csv(data_path / "test.csv", index=False)

print(train_df.info())
print(train_df.head())
print(test_df.info())
print(test_df.head())