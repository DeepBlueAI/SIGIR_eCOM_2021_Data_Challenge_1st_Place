#%%

import json

import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

import utils
from constant import *
from model import LGBBinary

train = pd.read_pickle(f'{init_dir}df_local.pickle')
test = pd.read_pickle(f'{init_dir}df_online.pickle')
df_main = train.append(test, ignore_index=True)
df_train_local = pd.read_pickle(f'{init_dir}df_train_local.pickle')
df_train_online = pd.read_pickle(f'{init_dir}df_train_online.pickle')
df_train = df_train_local.append(df_train_online, ignore_index=True)
df_product = pd.read_pickle(f'{init_dir}df_product.pickle')

# %%
# params
print('df user num: ', len(df_main))
assert len(df_main) == df_main['user'].nunique()
test_min_time = test['time'].min()

# proprecess
# set session action idx, first add idx, first purchase idx.
df_train['cur_user_idx'] = df_train.groupby('user')['user'].cumcount().astype(np.int32)+1

user_first_add_idx_dict = df_train[df_train['action']=='add'].groupby('user')['cur_user_idx'].min()
df_train['user_first_add_idx']=df_train['user'].map(user_first_add_idx_dict)
print('first add mean idx: ', user_first_add_idx_dict.mean())

user_first_item_dict = df_train[df_train['action']=='add'].groupby(
    'user')['item'].first()
df_train['user_first_add_item']=df_train['user'].map(user_first_item_dict)

df_train.loc[(df_train['event_type']=='pageview')&(df_train['action'].isna()), 'action'] = 'view'

# df_train['time'] = 1000
df_train = df_train.sort_values(by=['user', 'time'])
df_train['browing_time'] = -df_train.groupby('user')['time'].diff(-1)

# %%
# feat_engine 

def feat_detail_action(df, df_train):
    feat = df[['user']].copy()

    def func(df_train_):
        df_interactive = df_train_[df_train_['item']==df_train_['user_first_add_item']]
        detail_count = df_interactive[df_interactive['action']=='detail'].groupby(
            'user')['user'].count()
        return detail_count

    df_train_ = df_train[df_train['cur_user_idx']<df_train['user_first_add_idx']]
    feat['detail_count_left'] = feat['user'].map(func(df_train_)).fillna(0)
    df_train_ = df_train[df_train['cur_user_idx']>df_train['user_first_add_idx']]
    feat['detail_count_right'] = feat['user'].map(func(df_train_)).fillna(0)

    def func1(df_train_):
        df_interactive = df_train_[df_train_['item']!=df_train_['user_first_add_item']]
        detail_count = df_interactive[df_interactive['action']=='detail'].groupby(
            'user')['user'].count()
        return detail_count

    df_train_ = df_train[df_train['cur_user_idx']<df_train['user_first_add_idx']]
    feat['other_detail_count_left'] = feat['user'].map(func1(df_train_)).fillna(0)
    df_train_ = df_train[df_train['cur_user_idx']>df_train['user_first_add_idx']]
    feat['other_detail_count_right'] = feat['user'].map(func1(df_train_)).fillna(0)

    def func2(df_train_):
        detail_count = df_train_.groupby('user')['user'].count()
        return detail_count

    df_train_ = df_train[df_train['cur_user_idx']<df_train['user_first_add_idx']]
    feat['nb_before_add'] = feat['user'].map(func2(df_train_)).fillna(0)

    # ratio
    eplison = 1e-5
    feat['detail_count_ratio_left'] = feat['detail_count_left']/(feat['detail_count_left']+feat['other_detail_count_left']+eplison)
    feat['detail_count_ratio_right'] = feat['detail_count_right']/(feat['detail_count_right']+feat['other_detail_count_right']+eplison)

    return feat[[
        'detail_count_left', 'detail_count_right',
        'other_detail_count_left', 'other_detail_count_right',
        'detail_count_ratio_left', 'detail_count_ratio_right',
        'nb_before_add',
        ]]

def feat_detail_action_time(df, df_train):
    df_train = df_train.copy()
    
    feat = df[['user']].copy()

    def func(df_train_):
        df_interactive = df_train_[df_train_['item']==df_train_['user_first_add_item']]
        detail_tot_time = df_interactive[df_interactive['action']=='detail'].groupby(
            'user')['browing_time'].sum()
        return detail_tot_time

    df_train_ = df_train[df_train['cur_user_idx']<df_train['user_first_add_idx']]
    feat['detail_tot_time_left'] = feat['user'].map(func(df_train_)).fillna(0)
    df_train_ = df_train[df_train['cur_user_idx']>df_train['user_first_add_idx']]
    feat['detail_tot_time_right'] = feat['user'].map(func(df_train_)).fillna(0)

    def func1(df_train_):
        df_interactive = df_train_[df_train_['item']==df_train_['user_first_add_item']]
        detail_last_time = df_interactive[df_interactive['action']=='detail'].groupby(
            'user')[['time', 'browing_time']].last()
        first_add_time = df_interactive.loc[
            df_interactive['cur_user_idx']==df_interactive['user_first_add_idx']].set_index(
            'user')['time']
        pre_detail_interval = first_add_time-detail_last_time['time']-detail_last_time['browing_time']
        assert pre_detail_interval.min()>=0
        return pre_detail_interval

    df_train_ = df_train[df_train['cur_user_idx']<=df_train['user_first_add_idx']]
    feat['pre_detail_interval'] = feat['user'].map(func1(df_train_)).fillna(-1)

    def func2(df_train_):
        df_interactive = df_train_[df_train_['item']==df_train_['user_first_add_item']]
        detail_last_time = df_interactive[df_interactive['action']=='detail'].groupby(
            'user')[['cur_user_idx', 'user_first_add_idx']].last()
        pre_detail_loc_diff = detail_last_time['user_first_add_idx']-detail_last_time['cur_user_idx']
        assert pre_detail_loc_diff.min()>0
        return pre_detail_loc_diff

    df_train_ = df_train[df_train['cur_user_idx']<df_train['user_first_add_idx']]
    feat['pre_detail_loc_diff'] = feat['user'].map(func2(df_train_)).fillna(-1)

    def func3(df_train_):
        df_interactive = df_train_[df_train_['item']!=df_train_['user_first_add_item']]
        detail_tot_time = df_interactive[df_interactive['action']=='detail'].groupby(
            'user')['browing_time'].sum()
        return detail_tot_time

    df_train_ = df_train[df_train['cur_user_idx']<df_train['user_first_add_idx']]
    feat['other_detail_tot_time_left'] = feat['user'].map(func3(df_train_)).fillna(0)
    df_train_ = df_train[df_train['cur_user_idx']>df_train['user_first_add_idx']]
    feat['other_detail_tot_time_right'] = feat['user'].map(func3(df_train_)).fillna(0)

    def func4(df_train_):
        last_time = df_train_.groupby('user')['time'].last()
        first_add_time = df_train_.loc[
            df_train_['cur_user_idx']==df_train_['user_first_add_idx']].set_index(
            'user')['time']
        pre_detail_interval = last_time-first_add_time
        assert pre_detail_interval.min()>=0
        return pre_detail_interval

    df_train_ = df_train[df_train['cur_user_idx']>=df_train['user_first_add_idx']]
    feat['interval_after_add'] = feat['user'].map(func4(df_train_)).fillna(-1)

    def func5(df_train_):
        first_time = df_train_.groupby('user')['time'].first()
        first_add_time = df_train_.loc[
            df_train_['cur_user_idx']==df_train_['user_first_add_idx']].set_index(
            'user')['time']
        pre_detail_interval = first_add_time-first_time
        assert pre_detail_interval.min()>=0
        return pre_detail_interval

    df_train_ = df_train[df_train['cur_user_idx']<=df_train['user_first_add_idx']]
    feat['interval_before_add'] = feat['user'].map(func5(df_train_)).fillna(-1)

    # ratio
    eplison = 1e-5
    feat['detail_tot_time_ratio_left'] = feat['detail_tot_time_left']/(feat['detail_tot_time_left']+feat['other_detail_tot_time_left']+eplison)
    feat['detail_tot_time_ratio_right'] = feat['detail_tot_time_right']/(feat['detail_tot_time_right']+feat['other_detail_tot_time_right']+eplison)

    return feat[[
        'detail_tot_time_left', 'detail_tot_time_right', 'pre_detail_interval', 'pre_detail_loc_diff',
        'other_detail_tot_time_left', 'other_detail_tot_time_right',
        'detail_tot_time_ratio_left', 'detail_tot_time_ratio_right',
        'interval_after_add', 'interval_before_add',
    ]]

def feat_add_action(df, df_train):
    feat = df[['user']].copy()
    def func(df_train_):
        df_interactive = df_train_[df_train_['item']==df_train_['user_first_add_item']]
        add_count = df_interactive[df_interactive['action']=='add'].groupby(
            'user')['user'].count()
        return add_count

    df_train_ = df_train[df_train['cur_user_idx']>df_train['user_first_add_idx']]
    feat['add_count_right'] = feat['user'].map(func(df_train_)).fillna(0)

    def func1(df_train_):
        df_interactive = df_train_[df_train_['item']!=df_train_['user_first_add_item']]
        add_count = df_interactive[df_interactive['action']=='add'].groupby(
            'user')['user'].count()
        return add_count

    df_train_ = df_train[df_train['cur_user_idx']>df_train['user_first_add_idx']]
    feat['not_first_item_add_count_right'] = feat['user'].map(func1(df_train_)).fillna(0)
    return feat[['add_count_right', 'not_first_item_add_count_right']]

def feat_action_statis(df, df_train):
    df_train = df_train.copy()
    feat = df[['user']].copy()
    df_train['action'] = df_train['action'].fillna('search')
    df_train_ = df_train[df_train['cur_user_idx']<df_train['user_first_add_idx']]
    all_action_cnt = df_train_.groupby('user')['action'].size()
    for action in ['detail', 'view', 'remove', 'search']:
        tmp = df_train_[df_train_['action']==action]
        cur_action_cnt = tmp.groupby('user')['action'].size()
        cur_action_ratio = cur_action_cnt/all_action_cnt
        feat[f'user_{action}_cnt_left'] = feat['user'].map(cur_action_cnt).fillna(0)
        feat[f'user_{action}_ratio_left'] = feat['user'].map(cur_action_ratio).fillna(0)
    
    df_train_ = df_train[df_train['cur_user_idx']>df_train['user_first_add_idx']]
    all_action_cnt = df_train_.groupby('user')['action'].size()
    for action in ['detail', 'add', 'view', 'remove', 'search']:
        tmp = df_train_[df_train_['action']==action]
        cur_action_cnt = tmp.groupby('user')['action'].size()
        cur_action_ratio = cur_action_cnt/all_action_cnt
        feat[f'user_{action}_cnt_right'] = feat['user'].map(cur_action_cnt).fillna(0)
        feat[f'user_{action}_ratio_right'] = feat['user'].map(cur_action_ratio).fillna(0)
    feat.pop('user')
    return feat

def feat_action_item_nunique(df, df_train):
    df_train = df_train.copy()
    feat = df[['user']].copy()

    df_train_ = df_train[df_train['cur_user_idx']<df_train['user_first_add_idx']]
    for action in ['detail', 'remove']:
        tmp = df_train_[df_train_['action']==action]
        mp = tmp.groupby('user')['item'].nunique()
        feat[f'user_{action}_item_nunique_left'] = feat['user'].map(mp)

    df_train_ = df_train[df_train['cur_user_idx']>df_train['user_first_add_idx']]
    for action in ['detail', 'add', 'remove']:
        tmp = df_train_[df_train_['action']==action]
        mp = tmp.groupby('user')['item'].nunique()
        feat[f'user_{action}_item_nunique_right'] = feat['user'].map(mp)
    
    feat.pop('user')
    return feat

def feat_item_meta(df, df_train):
    df_train = df_train.copy()
    feat = df[['user', 'item']].copy()
    use_feats = ['category_hash', 'price_bucket']
    df_train = df_train.join(df_product.set_index('item')[use_feats], how='left', on='item')
    feat = feat.join(df_product.set_index('item')[use_feats], how='left', on='item')
    
    def func(df_train_):
        res = df_train_[df_train_['action']=='detail'].groupby(
            'user')['price_bucket'].mean()
        return res
    feat['all_item_price_mean'] = feat['user'].map(func(df_train)).fillna(-1)

    def func1(df_train_):
        res = df_train_[df_train_['action']=='detail'].groupby(
            'user')['price_bucket'].std()
        return res
    feat['all_item_price_std'] = feat['user'].map(func1(df_train)).fillna(-1)

    def func2(df_train_):
        res = df_train_[df_train_['action']=='detail'].groupby(
            'user')['category_hash'].nunique()
        return res
    feat['all_item_category_nunique'] = feat['user'].map(func2(df_train)).fillna(-1)

    feat['price_minus_mean_price'] = feat['price_bucket']-feat['all_item_price_mean']

    feat = feat[[
        'category_hash', 'price_bucket',
        'all_item_price_mean', 'all_item_price_std', 'all_item_category_nunique',
        'price_minus_mean_price',
    ]]
    return feat

feat_list = [df_main]
feat_funcs = [
    feat_detail_action,
    feat_detail_action_time,
    feat_add_action,
    feat_action_statis,
    feat_action_item_nunique,
    feat_item_meta,  
] 
for func in feat_funcs:
    feat = func(df_main, df_train)
    for col in feat.columns:
        feat[col] = feat[col].astype(np.float32)
    feat_list.append(feat)
data = pd.concat( feat_list, axis=1 ).reset_index(drop=True)

def weight_accuracy(y, preds, nb_after_add):
    weights = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
    score = 0
    for nb in range(0, 11, 2):
        judge = (nb_after_add==nb)
        y_tmp = y[judge]
        preds_tmp = preds[judge]
        score += (metrics.accuracy_score(y_tmp, preds_tmp)*weights[nb//2])
    return score

# %%

def adjust_threshold(y_true, preds, nb_after_add):
    best_thre = 0.5
    best_sc = -1
    for i in range(40, 80, 1):
        thre = i/100
        y_preds = (preds>thre).astype(np.int16)
        cur_score = weight_accuracy(y_true, y_preds, nb_after_add) # metrics.accuracy_score
        # print(f'cur thre: {thre}, cur score: {cur_score}, preds 1 num: {(y_preds==1).sum()}')
        if cur_score>best_sc:
            best_sc = cur_score
            best_thre = thre
    return best_sc, np.mean((preds>best_thre).astype(np.int32)), best_thre

def adjust_threshold_each_nb(y_true, preds, nb_after_add):
    best_score = 0
    best_thres = []
    wei = 1.0
    for nb in range(0, 11, 2):
        judge = (nb_after_add==nb)
        y_tmp = y_true[judge]
        preds_tmp = preds[judge]
        best_thre = 0.5
        best_sc = -1
        for i in range(101):
            thre = i/100
            y_preds_tmp = (preds_tmp>thre).astype(np.int16)
            cur_score = metrics.accuracy_score(y_tmp, y_preds_tmp) 
            # print(f'cur thre: {thre}, cur score: {cur_score}, preds 1 num: {(y_preds==1).sum()}')
            if cur_score>best_sc:
                best_sc = cur_score
                best_thre = thre
        print('nb: ', nb, 'positive num: ', (preds_tmp>best_thre).sum())
        best_score += (best_sc*wei)
        wei -= 0.1
        best_thres.append(best_thre)
    return best_score, best_thres

def post_process_by_thres(preds, thres, nb_after_add):
    y_preds_bin = np.zeros(len(preds))
    for nb in range(0, 11, 2):
        judge = (nb_after_add==nb)
        y_preds_bin[judge&(preds>thres[nb//2])] = 1
    return y_preds_bin


test = data[data['label'].isna()].copy()
train = data[data['label'].notna()].copy()

use_feats = list(data.columns)
rm_feats = [
    'user', 'item', 'label', 'time', 'user_remove_item_nunique_right', 
]
for col in rm_feats:
    if col in use_feats:
        use_feats.remove(col)
with open('use_feat.txt', 'w') as f:
    f.write(json.dumps(use_feats))
    
print('use feat num: ', len(use_feats))
dt = data[use_feats].dtypes
print('64bit feat: ', dt[(dt=='float64')|(dt=='int64')])

# time split train valid set
split_time = test_min_time-30*24*3600*1000 
X_train = train[train['time']<split_time]
X_valid = train[train['time']>=split_time]
y_valid = X_valid['label']
y_train = X_train['label']

print(X_valid['user'].nunique(), X_valid.shape)
print("[INFO] y_train value counts: \n", y_train.value_counts())
print("[INFO] y_valid value counts: \n", y_valid.value_counts())
lgb_model = LGBBinary()
preds, imp = lgb_model.valid_fit(X_train[use_feats], X_valid[use_feats], y_train, y_valid)

print('all zeros score: ', weight_accuracy(y_valid, np.zeros(len(y_valid)), X_valid['nb_after_add'].values))
best_sc, tn_ratio, best_thre = adjust_threshold(y_valid, preds, X_valid['nb_after_add'].values)
print('generl adjust_threshold')
print(f'valid score: {best_sc}, positive num ratio: {tn_ratio}', f'best thre: {best_thre}')
print('positive num: ', (preds>best_thre).sum())
best_score, best_thres = adjust_threshold_each_nb(y_valid, preds, X_valid['nb_after_add'].values)
print('adjust_threshold_each_nb')
print(f'valid score: {best_score}', f'best thre: {best_thres}')
preds_bin = post_process_by_thres(preds, best_thres, X_valid['nb_after_add'].values)
print('positive num: ', (preds_bin==1).sum())

# %%
y = train['label'].astype(np.int32)
lgb_model.fit(train[use_feats], y)
test_preds = lgb_model.predict(test[use_feats])
preds_bin = post_process_by_thres(test_preds, best_thres, test['nb_after_add'].values)
print('predict test label distribution: \n', pd.Series(preds_bin).value_counts())

def pred_analyse(nb_list, preds):
    nb_list = np.array(nb_list)
    preds = pd.Series(preds)
    for i in range(0, 11, 2):
        preds_tmp = preds[nb_list==i]
        print(f'cur nb {i} \n', preds_tmp.value_counts())
pred_analyse(test['nb_after_add'].values, preds_bin)


# %%
sub = True
if sub: 
    user_label_dict = dict(zip(test['user'].values, preds_bin))
    utils.upload_res(user_label_dict)
