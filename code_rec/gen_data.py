# %%

import json
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from constant import *

# %%
def print_info(df):
    print('shape: ', df.shape)
    print('user nunique: ', df['user'].nunique())
    print('user item nunique mean: ', df.groupby('user')['item'].nunique().mean())
    print('user item count mean: ', df.groupby('user')['item'].count().mean())
    print('user sess size mean: ', df.groupby('user')['user'].size().mean())
   
def print_detail(df):
    print('user item nunique value counts: \n', df.groupby('user')['item'].nunique().value_counts().head(5))
    print('df info: ')
    print_info(df)
    df_zero = get_not_infor(df)
    print('df zeros info: ')
    print_info(df_zero)
    df_merit = df[~df['user'].isin(df_zero['user'].unique())]
    print('df merit info: ')
    print_info(df_merit)

def get_not_infor(df):
    judge = (df['query_vector'].isna())&(df['item'].isna())
    # print(df['user'].nunique())
    # df[~judge]['user'].nunique()
    a = set(df['user'].unique())
    b = set(df[~judge]['user'].unique())
    c = list(a-b)
    return df[df['user'].isin(c)].copy()

def gen_local_data_by_testdata():
    data_path = data_dir + 'train'
    train_browsing = pd.read_csv(f'{data_path}/browsing_train.csv')
    train_browsing.columns = ['user', 'event_type', 'action', 'item', 'time', 'url']
    train_search = pd.read_csv(f'{data_path}/search_train.csv')
    train_search.columns = ['user', 'query_vector', 'clicked_list', 'resp_list', 'time']
    train_browsing = train_browsing.sort_values(by=['user', 'time']).reset_index(drop=True)
    train_search = train_search.sort_values(by=['user', 'time']).reset_index(drop=True)

    test_data_path = data_dir + 'test'
    with open(f'{test_data_path}/rec_test_phase_2.json') as f:
        rec_test = json.loads(f.read())
    print('test session num: ', len(rec_test))
    sequences = []
    null_seq_num = 0
    for items in rec_test:
        session_id = items['query'][0]['session_id_hash']
        for seq in items['query']:
            if (seq['product_sku_hash'] is None) and (seq['query_vector'] is None):
                # print(items)
                null_seq_num += 1
                # print(1/0)
            sequences.append(seq)
            assert seq['session_id_hash'] == session_id
    print('test interaction num: ', len(sequences))
    df_test = pd.DataFrame(sequences)
    df_test.columns = [
        'user', 'query_vector', 'clicked_list', 'resp_list', 'time',
        'event_type', 'action', 'item', 'url', 'is_search'
    ]
    df_test = df_test.sort_values(by=['user', 'time']).reset_index(drop=True)
    df = df_test.copy()

    df['cur_user_idx'] = df.groupby('user')['user'].cumcount().astype(np.int32)+1
    
    df, user2unsee = gen_next_item_sample(df)
    df['unsee_list'] = df['user'].map(user2unsee)
 
    # 为了保证做特征一致。
    train_browsing.to_pickle(f'{init_dir}browsing_his_online.pickle')
    train_search.to_pickle(f'{init_dir}search_his_online.pickle')
    
    df_test = df_test[~df_test['user'].isin(df['user'].unique())]
    extra_browsing = df_test[df_test['event_type'].notna()]
    extra_search = df_test[df_test['event_type'].isna()]
    train_browsing = train_browsing.append(extra_browsing, ignore_index=True)
    train_search = train_search.append(extra_search, ignore_index=True)
    train_browsing = train_browsing.sort_values(by=['user', 'time']).reset_index(drop=True)
    train_search = train_search.sort_values(by=['user', 'time']).reset_index(drop=True)
    train_browsing.to_pickle(f'{init_dir}browsing_his_local.pickle')
    train_search.to_pickle(f'{init_dir}search_his_local.pickle')

    main_col = [
        'user', 'event_type', 'action', 'item',
        'time', 'url', 'query_vector',
        'clicked_list', 'resp_list', 'cur_user_idx', 'next_item', 'unsee_list', 
    ]
    assert (df['item']==df['next_item']).sum() == 0
    df[main_col].to_pickle(f'{init_dir}df_local.pickle')
    print_detail(df)

def gen_next_item_sample(df):
    df = df.reset_index(drop=True)
    data = []
    user2unsee = {}
    user_idx = 0
    pre_user = df['user'].iloc[0]
    activate_search_list = []
    cur_items_idx = []
    for cnt, (user, item, clicks) in tqdm(enumerate(df[['user', 'item', 'clicked_list']].values)):
        if user != pre_user:
            his_item_set = []
            for i in range(len(activate_search_list)):
                if cur_items_idx[i] == -1:
                    for j in activate_search_list[i]:
                        his_item_set.append(j)
                else:
                    his_item_set.append(activate_search_list[i])
            unsee_item_list = []

            for i in range(len(activate_search_list)-1, -1, -1):
                if cur_items_idx[i] != -1:
                    unsee_item_list.append(activate_search_list[i])
                    his_item_set.pop(-1)
                    if (len(unsee_item_list)>0) and (cur_items_idx[i]>user_idx) and (len(set(his_item_set)&set(unsee_item_list))==0):
                        tmp = df[user_idx: cur_items_idx[i]].copy()
                        assert len(tmp)>0
                        tmp['user'] = pre_user
                        tmp['next_item'] = unsee_item_list[-1]
                        user2unsee[pre_user] = tuple(set(unsee_item_list[:]))
                        data.append(tmp)
                        break
                else:
                    for _ in range(len(activate_search_list[i])):
                        his_item_set.pop(-1)
                    if (i>0) and (cur_items_idx[i-1] != -1):
                        assert his_item_set[-1] == activate_search_list[i-1]

            user_idx = cnt
            activate_search_list.clear()
            cur_items_idx.clear()
        if pd.notna(item):
            activate_search_list.append(item)
            cur_items_idx.append(cnt)
        if isinstance(clicks, list):
            activate_search_list.append(clicks)
            cur_items_idx.append(-1)
        pre_user = user
    data = pd.concat(data, ignore_index=True)
    assert data['user'].nunique()==len(user2unsee)
    return data, user2unsee

def gen_online_data():
    test_data_path = f'{data_dir}test'
    with open(f'{test_data_path}/rec_test_phase_2.json') as f:
        rec_test = json.loads(f.read())
    print('test session num: ', len(rec_test))
    sequences = []
    null_seq_num = 0
    for items in rec_test:
        session_id = items['query'][0]['session_id_hash']
        for seq in items['query']:
            if (seq['product_sku_hash'] is None) and (seq['query_vector'] is None):
                # print(items)
                null_seq_num += 1
                # print(1/0)
            sequences.append(seq)
            assert seq['session_id_hash'] == session_id
    print('test interaction num: ', len(sequences))
    df_rec = pd.DataFrame(sequences)
    df_rec.columns = [
        'user', 'query_vector', 'clicked_list', 'resp_list', 'time',
        'event_type', 'action', 'item', 'url', 'is_search'
    ]
    
    df_rec = df_rec.sort_values(by=['user', 'time']).reset_index(drop=True)
    df_rec['cur_user_idx'] = df_rec.groupby('user')['user'].cumcount().astype(np.int32)+1
    main_col = [
        'user', 'event_type', 'action', 'item',
        'time', 'url', 'query_vector',
        'clicked_list', 'resp_list', 'cur_user_idx', # 'next_item',
    ]
    df_rec[main_col].to_pickle(f'{init_dir}df_online.pickle')
    print_detail(df_rec)

def vec_preprocess():
    df_product = pd.read_csv(f'{data_dir}train/sku_to_content.csv')
    df_product.columns = ['item', 'description_vector', 'category_hash', 'image_vector', 'price_bucket']
    df_product['category_hash'], _ = pd.factorize(df_product['category_hash'])
    df_product = df_product.astype({'category_hash': np.float32, 'price_bucket': np.float32})
    df_product.to_pickle(f'{init_dir}df_product.pickle')
    df_product = pd.read_pickle(f'{init_dir}df_product.pickle')
    print(df_product.info())
    item2desp = {}
    item2image = {}
    for item, des, image in df_product[['item', 'description_vector', 'image_vector']].values:
        if pd.notna(des):
            des_vec = eval(des)
            item2desp[item] = des_vec
        if pd.notna(image):
            image_vec = eval(image)
            item2image[item] = image_vec

    pickle.dump(item2desp, open(f'{user_data_dir}item2desp.pickle', 'wb'))
    pickle.dump(item2desp, open(f'{user_data_dir}item2image.pickle', 'wb'))

    for k, v in item2desp.items():
        assert len(v) == 50

    for k, v in item2image.items():
        assert len(v) == 50

    from gensim.models import Word2Vec

    entities = list(item2desp.keys())
    weights = list(item2desp.values())
    model_desp = Word2Vec(vector_size=50)
    wv_desp = model_desp.wv
    wv_desp.add_vectors(entities, weights)
    wv_desp.save(f'{user_data_dir}wv_desp.kv')
    entities = list(item2image.keys())
    weights = list(item2image.values())
    model_image = Word2Vec(vector_size=50)
    wv_image = model_image.wv
    wv_image.add_vectors(entities, weights)
    wv_image.save(f'{user_data_dir}wv_image.kv')

if __name__ == '__main__':
    gen_local_data_by_testdata()
    gen_online_data()
    vec_preprocess()
