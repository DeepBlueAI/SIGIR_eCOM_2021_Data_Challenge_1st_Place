#%%
import math
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from constant import *
from evaluation import mrr_at_k

mode = cur_mode

def print_recall_info(df):
    print('recall df shape: ', df.shape)
    print('user num: ', df['user'].nunique(), 'each user mean item: ', len(df)/df['user'].nunique())

# %%
i2i_sim_limit = 150
max_sim_num = 200
get_sum = False
big_or_small ='small'

def get_whole_time_wei(df, base_time):
    time_list = df['time'].copy()
    base_time = pd.to_datetime(base_time, unit='ms')
    time_list = pd.to_datetime(time_list, unit='ms')
    time_wei = ((base_time-time_list) // pd.Timedelta('1day'))+1
    time_wei[time_wei<0] = 0
    time_wei = 1 - time_wei/100
    time_wei = np.log(time_wei)+1
    time_wei[time_wei<0.1] = 0.1
    # time_wei **= 2
    assert ((time_wei>1)|(time_wei<=0)).sum() == 0
    return time_wei

def get_ban_item(df):
    search = df[df['clicked_list'].notna()].copy()
    interact = df[df['item'].notna()].copy()
    interact = interact.drop_duplicates(subset=['user', 'item'], keep='last')
    cur_sess_product_dict = interact.groupby('user')['item'].agg(list).to_dict()
    # get ban products
    ban_product = cur_sess_product_dict.copy()
    for sess, prods in search[['user', 'clicked_list']].values:
        if sess in ban_product:
            new_bans = ban_product[sess]
        else:
            new_bans = []
        # print(type(prods))
        new_bans = list(set(new_bans)|set(prods)) # eval
        ban_product[sess] = new_bans
    return ban_product

def u2i_interact_rec(sim_item_corr, user_item_dict, user_time_dict, ban_item_dict, user_id, loc_coff=0.7, recall_type=None):
    interacted_items = user_item_dict[user_id]
    ban_items = ban_item_dict[user_id]
    interacted_times = user_time_dict[user_id]
    qtime_loc = len(interacted_items)
    qtime = interacted_times[-1]
    # while qtime_loc<len(interacted_times) and qtime >= interacted_times[qtime_loc]:
    #     qtime_loc+=1
    if big_or_small == 'big':
        if recall_type== 'i2i_itemcf_w02':
            recall_left_max_num_each_road = 200
            recall_max_road_num = 20
        elif recall_type == 'i2i_cossim':
            recall_left_max_num_each_road = 200
            recall_max_road_num = 20
        elif recall_type == 'i2i2i_new':
            recall_left_max_num_each_road = 150
            recall_max_road_num = 20
    else:
        if recall_type== 'i2i_itemcf_w02':
            recall_left_max_num_each_road = 100
            recall_max_road_num = 10
        elif recall_type == 'i2i_cossim':
            recall_left_max_num_each_road = 100
            recall_max_road_num = 10
        elif recall_type == 'i2i2i_new':
            recall_left_max_num_each_road = 100
            recall_max_road_num = 10

    if get_sum:
        multi_road_result = {}
    else:
        multi_road_result = []
    for j, item in enumerate(interacted_items[:-recall_max_road_num-1:-1]):
        # item = interacted_items[ l_cans_loc[i] ]
        item_loc = len(interacted_items)-j-1
        time = interacted_times[ item_loc]
        each_road_result = []

        loc_weight = 0.7**j
        if loc_weight<=0.1:
            loc_weight = 0.1
        time_weight = (1 - abs( qtime - time ) / 60000)
        if time_weight<=0.1:
            time_weight = 0.1
        
        if item not in sim_item_corr:
            continue
        for j, wij in sim_item_corr[ item ].items(): 
            if j not in ban_items:
                sim_weight = wij
                rank_weight = sim_weight * loc_weight * time_weight
                each_road_result.append( ( j, sim_weight, loc_weight, time_weight, rank_weight, item,
                                       item_loc, time, qtime_loc, qtime, user_id ) )
        each_road_result.sort(key=lambda x:x[1], reverse=True)
        each_road_result = each_road_result[0:recall_left_max_num_each_road]
        
        if get_sum:
            for idx,k in enumerate(each_road_result):
                if k[0] not in multi_road_result:
                    multi_road_result[k[0]] = k[1:]
                else:
                    t1 = multi_road_result[k[0]]
                    t2 = k[1:]
                    multi_road_result[k[0]] = ( t1[0]+t2[0] , t1[1], t1[2], t1[3]+t2[3], t1[4],
                                                t1[5], t1[6], t1[7], t1[8], t1[9] )
        else:
            multi_road_result += each_road_result

    if get_sum:
        multi_road_result_t = sorted(multi_road_result.items(), key=lambda i: i[1][3], reverse=True)
        multi_road_result = []
        for q in multi_road_result_t:
            multi_road_result.append( (q[0],)+q[1] )
    else:
        multi_road_result.sort(key=lambda x:x[4], reverse=True)

    return multi_road_result

def u2i_interact_i2i_itemcf_w02_recall(df_train, df):
    df_train = df.append(df_train)
    df_train = df_train[df_train['item'].notna()].copy()
    df_train['time_wei'] = get_whole_time_wei(df_train, df['time'].min()) 
    user2time_wei = df_train.groupby('user')['time_wei'].max().to_dict()

    df_train = df_train.drop_duplicates(subset=['user', 'item'], keep='last') # 
    print(df_train.shape)

    user_item_ = df_train.groupby('user')['item'].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_['user'], user_item_['item']))
    user_time_ = df_train.groupby('user')['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_['user'], user_time_['time']))
    all_pair_num = 0
    sim_item = {}
    item_cnt = defaultdict(int)
    for user, items in user_item_dict.items():
        times = user_time_dict[user]
        time_wei = user2time_wei[user]
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):  
                if item == relate_item:
                    continue
                all_pair_num += 1               
                t1 = times[loc1]
                t2 = times[loc2]
                sim_item[item].setdefault(relate_item, 0)
                if loc1-loc2>0:
                    time_weight = (1 - (t1 - t2) / 60000)
                    if time_weight<=0.2:
                        time_weight = 0.2
                        
                    loc_diff = loc1-loc2-1
                    loc_weight = (0.9**loc_diff)
                    if loc_weight <= 0.2:
                        loc_weight = 0.2

                    sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight*time_wei / math.log(1 + len(items))
                          
                else:
                    time_weight = (1 - (t2 - t1) / 60000)
                    if time_weight<=0.2:
                        time_weight = 0.2
                    
                    loc_diff =  loc2-loc1-1
                    loc_weight =  (0.9**loc_diff)
                    
                    if loc_weight <= 0.2:
                        loc_weight = 0.2
                    
                    sim_item[item][relate_item] += 1 * 1.0 * loc_weight * time_weight*time_wei / math.log(1 + len(items)) 
                                        
    for i, related_items in sim_item.items():  
        for j, cij in related_items.items():  
            sim_item[i][j] = cij / ((item_cnt[i] * item_cnt[j]) ** 0.2) #  

    print('all_pair_num',all_pair_num)
    for key in sim_item.keys():
        t = sim_item[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = {}
        for i in t[0:i2i_sim_limit]:
            res[i[0]]=i[1]
        sim_item[key] = res
    # get ban products
    ban_product = get_ban_item(df)
    user2recall = []
    user2qtime = df.groupby('user')['time'].last().to_dict()
    user2qloc = df.groupby('user')['cur_user_idx'].last().to_dict()
    df = df[df['item'].notna()]
    df = df.drop_duplicates(subset=['user', 'item'], keep='last')
    cur_sess_product_dict = df.groupby('user')['item'].agg(list).to_dict()
    cur_sess_time_dict = df.groupby('user')['time'].agg(list).to_dict()
    
    for user in tqdm(df['user'].unique()):
        user2recall.extend(u2i_interact_rec(sim_item,cur_sess_product_dict,cur_sess_time_dict,ban_product,user,0.7,'i2i_itemcf_w02'))

    col_nams = [
        'item', 'sim_weight', 'loc_weight', 'time_weight', 'rank_weight',
        'road_item', 'road_item_loc', 'road_item_time',
        'query_item_loc', 'query_item_time', 'user',
    ]
    dtype = {
        'sim_weight': np.float32, 'loc_weight': np.float32,
        'time_weight': np.float32, 'rank_weight': np.float32,
        'road_item_loc': np.float32, 'road_item_time': np.int64,
        'query_item_loc': np.float32, 'query_item_time': np.int64,
    }
    df_pred = pd.DataFrame(user2recall, columns=col_nams)
    df_pred = df_pred.astype(dtype)
    print_recall_info(df_pred)
    df_pred.to_pickle(recall_dir+f'u2i_interact_i2i_itemcf_w02_{mode}.pickle')

def u2url_url2i_urlcf_recall(df_train, df):
    assert df_train['url'].isna().sum() == 0
    df_train = df.append(df_train, ignore_index=True)
    df_train = df_train.sort_values(by=['user', 'time'])
    df_train['time_wei'] = get_whole_time_wei(df_train, df['time'].min())
    user2time_wei = df_train.groupby('user')['time_wei'].max().to_dict()
    # params
    max_shift = 3
    url2i_loc_wei_reduce = 0.7
    u2url_loc_wei_reduce = 0.2
    each_url_max_sim_item = 200
    each_road_url_max_item = [150, 75, 25]

    shift_col = []
    judge = False
    for shift in range(1, max_shift+1):
        shift_col.append(f'cur_next_{shift}_item')
        df_train[f'cur_next_{shift}_item'] = df_train.groupby('user')['item'].shift(-shift)
        judge |= df_train[f'cur_next_{shift}_item'].notna()
    df_train = df_train[judge]
    url2items = {}
    url2cnt = defaultdict(int)
    for v in df_train[['user', 'url']+shift_col].values:
        user, url = v[0], v[1]
        url2items.setdefault(url, {})
        items = v[2:]
        loc_wei = 1
        time_wei = user2time_wei[user]
        for item in items:
            if pd.notna(item) and (item is not None):
                # url2cnt[url] += 1
                url2items[url].setdefault(item, 0)
                url2items[url][item] += 1.0*time_wei*loc_wei
                url2cnt[url] += 1.0*time_wei*loc_wei
            loc_wei *= url2i_loc_wei_reduce

    for i, related_items in url2items.items():  
        url_cnt = url2cnt[i]
        for j, cij in related_items.items():  
            url2items[i][j] = cij / (url_cnt * 1.0)  # 对多的减少惩罚
    
    for key in url2items.keys():
        t = url2items[key]
        t = sorted(t.items(), key=lambda d:d[1], reverse = True )
        res = {}
        for i in t[0:each_url_max_sim_item]:
            res[i[0]]=i[1]
        url2items[key] = res

    # get ban products
    ban_product = get_ban_item(df)
    user2recall = []

    df = df.copy()
    df = df.sort_values(by=['user', 'time'], ascending=[True, False])
    df = df.groupby('user', as_index=False).head(max_shift) # 1
    user2url = df.groupby('user')['url'].agg(list).to_dict()
    user2time = df.groupby('user')['time'].agg(list).to_dict()
    # df['cur_max_idx'] = df.groupby('user')['cur_user_idx'].transform('max')
    # df = df[df['cur_user_idx']==df['cur_max_idx']]
    
    for user, urls in tqdm(user2url.items()):
        ban_items = ban_product[user] if user in ban_product else []
        loc_wei = 1.0
        tts = user2time[user]
        qtime = tts[0]
        for i, url in enumerate(urls):
            if url not in url2items:
                continue
            cur_recall_num = 0
            assert qtime >= tts[i]
            time_weight = (1 - (qtime - tts[i]) / 60000) # params
            if time_weight<=0.1:
                time_weight = 0.1
            for item, wij in url2items[url].items():
                if item not in ban_items:
                    rank_weight = 1.0*loc_wei*wij*time_weight # time_weight
                    user2recall.append([item, wij, loc_wei, time_weight, rank_weight, user])
                    cur_recall_num += 1
                    if cur_recall_num >= each_road_url_max_item[i]:
                        break
            loc_wei *= u2url_loc_wei_reduce

    col_nams = [
        'item', 'sim_weight', 'loc_weight', 'time_weight', 'rank_weight', 'user',
    ]
    
    dtype = {'rank_weight': np.float32}

    df_pred = pd.DataFrame(user2recall, columns=col_nams)
    df_pred = df_pred.astype(dtype)
    print_recall_info(df_pred)
    df_pred.to_pickle(recall_dir+f'u2url_url2i_urlcf_{mode}.pickle')

def recall_score(df_pred, df):
    print_recall_info(df_pred)
    tmp = df_pred.groupby(['user', 'item'], as_index=False)['rank_weight'].sum()
    tmp = tmp.sort_values(by=['user', 'rank_weight'], ascending=[True, False])
    print('del duplicate (user, item)')
    print_recall_info(tmp)
    if mode == 'local':
        sess_lab_dict = df.drop_duplicates(subset=['user']).set_index('user')['next_item']
        print('all user num: ', len(sess_lab_dict))
        tmp['label_item'] = tmp['user'].map(sess_lab_dict)
        assert tmp['label_item'].isna().sum() == 0
        print('recall rate: ', (tmp['label_item']==tmp['item']).sum()/df['user'].nunique())
        print('有召回的user mean recall rate: ', (tmp['label_item']==tmp['item']).sum()/tmp['user'].nunique())
        tmp = tmp.groupby('user').head(20)
        print('@20recall rate: ', (tmp['label_item']==tmp['item']).sum()/df['user'].nunique())
        print('有召回的user mean @20recall rate: ', (tmp['label_item']==tmp['item']).sum()/tmp['user'].nunique())
        preds = tmp.groupby('user')['item'].agg(list).reindex(sess_lab_dict.index, fill_value=[])
        labels = sess_lab_dict.reset_index().groupby('user')['next_item'].agg(list).to_list()
        print('mrr: ' ,mrr_at_k(preds, labels, 20))
        preds = tmp.groupby('user')['item'].agg(list).to_list()
        labels = tmp.groupby('user')['label_item'].agg(list).to_list()
        print('有召回的user mean mrr: ' ,mrr_at_k(preds, labels, 20))

recall_diff_road_func_list = [
    u2i_interact_i2i_itemcf_w02_recall,
    u2url_url2i_urlcf_recall,
]

print('cur mode: ', mode)

df_train = pd.read_pickle(f'{init_data}browsing_his_{mode}.pickle')
df = pd.read_pickle(f'{init_data}df_{mode}.pickle')
for func in recall_diff_road_func_list:
    func(df_train, df)

# %%
def recall_online_score(df_pred, df):
    tmp = df_pred.groupby(['user', 'item'], as_index=False)['rank_weight'].sum()
    tmp = tmp.sort_values(by=['user', 'rank_weight'], ascending=[True, False])
    print('del duplicate (user, item)')
    print_recall_info(tmp)
    if mode == 'local':
        sess_lab_dict = df.drop_duplicates(subset=['user']).set_index('user')['next_item']
        print('all user num: ', len(sess_lab_dict))
        tmp['label_item'] = tmp['user'].map(sess_lab_dict)
        assert tmp['label_item'].isna().sum() == 0
        print('recall rate: ', (tmp['label_item']==tmp['item']).sum()/df['user'].nunique())
        tmp = tmp.groupby('user').head(20)
        print('@20recall rate: ', (tmp['label_item']==tmp['item']).sum()/df['user'].nunique())
        preds = tmp.groupby('user')['item'].agg(list).reindex(sess_lab_dict.index, fill_value=[])
        labels = sess_lab_dict.reset_index().groupby('user')['next_item'].agg(list).to_list()
        print(mrr_at_k(preds, labels, 20))
    else:
        tmp = tmp.groupby(['user']).head(20)
        sess_preds_dict = tmp.groupby('user')['item'].agg(list).to_dict()
        utils.upload_res(sess_preds_dict)

def merge_recall_weight(dfs, weights):
    data = dfs[0]
    data = data.groupby(['user', 'item'], as_index=False)['rank_weight'].sum()
    data['rank_weight'] = data['rank_weight']*weights[0]
    for i, df in enumerate(dfs[1:]):
        tmp = df.groupby(['user', 'item'], as_index=False)['rank_weight'].sum()
        tmp = tmp.rename(columns={'rank_weight': 'rank_weight_tmp'})
        data = data.merge(tmp, how='outer', on=['user', 'item'])
        data['rank_weight'] = data['rank_weight'].fillna(0)
        data['rank_weight_tmp'] = data['rank_weight_tmp'].fillna(0)
        data['rank_weight'] += (data['rank_weight_tmp']*weights[i+1])
        data.pop('rank_weight_tmp')
    return data

df = pd.read_pickle(f'{init_data}df_{mode}.pickle')
df_pred = pd.read_pickle(recall_dir+f'u2url_url2i_urlcf_{mode}.pickle')
# recall_online_score(df_pred, df) # df_pred
df_pred_1 = pd.read_pickle(recall_dir+f'u2i_interact_i2i_itemcf_w02_{mode}.pickle')
data = merge_recall_weight([df_pred, df_pred_1], [0.98, 0.02])
recall_online_score(data, df) # df_pred
