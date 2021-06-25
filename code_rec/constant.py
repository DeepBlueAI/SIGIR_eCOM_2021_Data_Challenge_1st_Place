import os 

data_dir = '../data/'

cur_mode = 'online'
cur_used_recall_source = 'u2i_interact_i2i_itemcf_w02-u2url_url2i_urlcf'


prediction_result = '../prediction_result/'
if not os.path.exists(prediction_result):
    os.makedirs(prediction_result)

user_data_dir = '../user_data_rec/'
if not os.path.exists(user_data_dir):
    os.makedirs(user_data_dir)

lgb_model_dir = '../user_data_rec/lgb_model/'
if not os.path.exists(lgb_model_dir):
    os.makedirs(lgb_model_dir)

feat_dir = '../user_data_rec/feat_data/'
if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

init_dir = '../user_data_rec/init_data/'
if not os.path.exists(init_dir):
    os.makedirs(init_dir)

recall_dir = '../user_data_rec/recall_data/'
if not os.path.exists(recall_dir):
    os.makedirs(recall_dir)

merge_dir = '../user_data_rec/merge_data/'
if not os.path.exists(merge_dir):
    os.makedirs(merge_dir)

SEED = 2021