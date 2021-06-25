import os 

data_dir = '../data/'

prediction_result = '../prediction_result/'
if not os.path.exists(prediction_result):
    os.makedirs(prediction_result)

user_data_dir = '../user_data/'
if not os.path.exists(user_data_dir):
    os.makedirs(user_data_dir)

lgb_model_dir = '../user_data/lgb_model/'
if not os.path.exists(lgb_model_dir):
    os.makedirs(lgb_model_dir)

feat_dir = '../user_data/feat_data/'
if not os.path.exists(feat_dir):
    os.makedirs(feat_dir)

init_dir = '../user_data/init_data/'
if not os.path.exists(init_dir):
    os.makedirs(init_dir)

merge_dir = '../user_data/merge_data/'
if not os.path.exists(merge_dir):
    os.makedirs(merge_dir)

SEED = 2021