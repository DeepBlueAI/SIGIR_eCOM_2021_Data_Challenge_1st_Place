#%%
import json

import pandas as pd
import numpy as np

from constant import *

def preprocess(df):
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
    df['cur_user_idx'] = df.groupby('user')['user'].cumcount().astype(np.int32)+1

#%%
def init_product():
    # preprocess df_product
    df_product = pd.read_csv(f'{data_dir}train/sku_to_content.csv')
    df_product.columns = ['item', 'description_vector', 'category_hash', 'image_vector', 'price_bucket']
    df_product['category_hash'], _ = pd.factorize(df_product['category_hash'])
    df_product = df_product.astype({'category_hash': np.float32, 'price_bucket': np.float32})
    df_product.to_pickle(f'{init_data}df_product.pickle')

def gen_online_data():
    test_data_path = f'{data_dir}test'
    with open(f'{test_data_path}/intention_test_phase_2.json') as f:
        rec_test = json.loads(f.read())
    print('test session num: ', len(rec_test))
    sequences = []
    sequences_2 = []

    null_seq_num = 0
    for items in rec_test:
        session_id = items['query'][0]['session_id_hash']
        nb_after_add = items['nb_after_add']
        purchase_num = 0

        first_add_product = ''
        cal_nb = 0
        for i, seq in enumerate(items['query']):
            if (seq['product_sku_hash'] is None) and (seq['query_vector'] is None):
                null_seq_num += 1
            if seq['product_action'] == 'purchase':
                purchase_num += 1
            if (first_add_product != ''): # and (seq['product_action'] is not None):
                cal_nb += 1
            if (first_add_product == '') and (seq['product_action']=='add'):
                first_add_product = seq['product_sku_hash']
                t = seq['server_timestamp_epoch_ms']
            sequences.append(seq)
            assert seq['session_id_hash'] == session_id
        if purchase_num > 0:
            print(items)
            print(1/0)
        sequences_2.append([session_id, first_add_product, t, nb_after_add]) # first_add_product

    print('test interaction num: ', len(sequences))
    df_train = pd.DataFrame(sequences)
    df_train.columns = [
        'user', 'query_vector', 'clicked_list', 'resp_list', 'time',
        'event_type', 'action', 'item', 'url', 'is_search'
    ]
    main_col = [
        'user', 'event_type', 'action', 'item',
        'time', 'url', 'query_vector',
        'clicked_list', 'resp_list', 'cur_user_idx',
    ]
    df_train['cur_user_idx'] = df_train.groupby('user')['user'].cumcount().astype(np.int32)+1
    df = pd.DataFrame(sequences_2, columns=['user', 'item', 'time', 'nb_after_add']) # 'item',
    df['nb_after_add'] = df['nb_after_add'].astype(np.float32)
    df_train[main_col].to_pickle(f'{init_dir}df_train_online.pickle')
    df.to_pickle(f'{init_dir}df_online.pickle')

def get_all_data_and_init():
    data_path = f'{data_dir}train'
    train_browsing = pd.read_csv(f'{data_path}/browsing_train.csv')
    train_browsing.columns = ['user', 'event_type', 'action', 'item', 'time', 'url']
    train_search = pd.read_csv(f'{data_path}/search_train.csv')
    train_search.columns = ['user', 'query_vector', 'clicked_list', 'resp_list', 'time']
    print('train_search info: \n', train_search.info())

    train = train_browsing.append(train_search)
    print(train.shape)
    train = train.sort_values(by=['user', 'time'])

     # take have add action session
    add_session = train[train['action']=='add']['user'].unique()
    train = train[train['user'].isin(add_session)].copy()

    # set session action idx, first add idx, first purchase idx.
    train['cur_user_idx'] = train.groupby('user')['user'].cumcount().astype(np.int32)+1

    user_first_add_idx_dict = train[train['action']=='add'].groupby('user')['cur_user_idx'].min()
    train['user_first_add_idx']=train['user'].map(user_first_add_idx_dict)
    print('first add mean idx: ', user_first_add_idx_dict.mean())

    user_first_purchase_idx_dict = train[train['action']=='purchase'].groupby('user')['cur_user_idx'].min()
    train['user_first_purchase_idx']=train['user'].map(user_first_purchase_idx_dict)
    print('first purchase mean idx: ', user_first_purchase_idx_dict.mean())
    # build cart-abandonment label
    userfirst_product_dict = train[train['action']=='add'].groupby('user')['item'].first()
    train['user_first_add_product']=train['user'].map(userfirst_product_dict)
    return train

def gen_local_data_extend_data():
    train = get_all_data_and_init()

    lab1_sess = train[
        (train['user_first_add_product']==train['item'])&(
         train['action']=='purchase')]['user'].unique()

    # get before first purchase , consider test data have not any purchase action
    train = train[train['user_first_purchase_idx'].fillna(10000)>train['cur_user_idx']]
    train = train[train['user_first_add_idx']<train['user_first_purchase_idx'].fillna(10000)]

    # statis after first add, rest action num
    tmp = train[train['user_first_add_idx']<train['cur_user_idx']]
    user_cal_nb_dict = tmp.groupby('user')['user'].count()
    train['cal_nb_after_add'] = train['user'].map(user_cal_nb_dict).fillna(0)
    train['nb_after_add'] = train['cal_nb_after_add'] + 1
    train['nb_after_add'] -= (train['nb_after_add']%2)

    # data extend
    data = []
    new_lab1_user = []
    train['cur_nb_after_add'] = train['cur_user_idx']-train['user_first_add_idx']
    for nb in range(0, 11, 2):
        tmp = train[(train['cur_nb_after_add']<=nb)&(train['nb_after_add']>=nb)].copy()
        tmp['user'] = tmp['user']+'_'+str(nb)
        new_lab1_user.extend([i+'_'+str(nb) for i in lab1_sess])
        tmp['nb_after_add'] = nb
        data.append(tmp)
    train = pd.concat(data, ignore_index=True)
    print('after data extend train shape: ', train.shape)

    df = train[
        train['user_first_add_idx']==train['cur_user_idx']][
        ['user', 'item', 'time', 'nb_after_add']].copy()

    df['label'] = 0
    df.loc[df['user'].isin(new_lab1_user), 'label'] = 1

    dtypes = {'nb_after_add': np.int32, 'label': np.float32}
    df = df.astype(dtypes)
    train = train.sort_values(by=['user', 'time'])
    main_col = [
        'user', 'event_type', 'action', 'item',
        'time', 'url', 'query_vector',
        'clicked_list', 'resp_list', 'cur_user_idx', # 'next_item',
    ]
    train[main_col].to_pickle(f'{init_dir}df_train_local.pickle')
    df.to_pickle(f'{init_dir}df_local.pickle')

def gen_his_data():
    data_path = f'{data_dir}train'
    train_browsing = pd.read_csv(f'{data_path}/browsing_train.csv')
    train_browsing.columns = ['user', 'event_type', 'action', 'item', 'time', 'url']
    train_search = pd.read_csv(f'{data_path}/search_train.csv')
    train_search.columns = ['user', 'query_vector', 'clicked_list', 'resp_list', 'time']
    train_browsing = train_browsing.sort_values(by=['user', 'time']).reset_index(drop=True)
    train_search = train_search.sort_values(by=['user', 'time']).reset_index(drop=True)
    train_browsing.to_pickle(f'{init_dir}browsing_his.pickle')
    train_search.to_pickle(f'{init_dir}search_his.pickle')

if __name__ == '__main__':
    init_product()
    gen_local_data_extend_data()
    gen_online_data()
    gen_his_data()
