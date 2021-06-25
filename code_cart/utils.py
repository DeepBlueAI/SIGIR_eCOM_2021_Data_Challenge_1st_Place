import os
import json
import time

import pandas as pd
import numpy as np
import pytz

from uploader import upload_submission


def build_bin_time(ss, rounding=False, unit='1d', shift_time=None):
    if shift_time is not None:
        ss = ss+shift_time
    if rounding:
        return ((ss-pd.Timestamp("2021-03-22 00:00:00", tzinfo=pytz.FixedOffset(480))
        ) // pd.Timedelta(unit)).astype(np.float32)
    else:
        return ((ss-pd.Timestamp("2021-03-22 00:00:00", tzinfo=pytz.FixedOffset(480))
        ) / pd.Timedelta(unit)).astype(np.float32)

def gen_combine_cats(df, cols):
    category = df[cols[0]].astype('float64')
    for col in cols[1:]:
        assert (df[col].min()>-1).all()
        mx = df[col].max()
        category *= mx
        category += df[col]
    return category
    
def gen_combine_cats_str(df, cols):
    category = df[cols[0]].astype('str')
    for col in cols[1:]:
        category = category + '-' + df[col].astype('str')
    return category

def upload_res(sess2res):

    # user_label_dict = dict(zip(test['user'].values, preds_bin))
    # make submission file
    test_data_path = '../data/test'
    with open(f'{test_data_path}/intention_test_phase_2.json') as f:
        test_queries = json.load(f)
    cnt_preds = 0
    my_predictions = []
    for t in test_queries:
        # copy the test case
        _pred = dict(t)
        cur_user = t['query'][0]['session_id_hash']
        _pred["label"] = int(sess2res[cur_user])
        cnt_preds += 1
        # append prediction to the final list
        my_predictions.append(_pred)
        
    # check for consistency
    assert len(my_predictions) == len(test_queries)
    # print out some "coverage"
    print("Predictions made in {} out of {} total test cases".format(cnt_preds, len(test_queries)))

    upload = True
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='upload.env.local')
    EMAIL = os.getenv('EMAIL', None) # the e-mail you used to sign up
    assert EMAIL is not None
    local_prediction_file = '{}_{}.json'.format(EMAIL.replace('@', '_'), round(time.time() * 1000))
    with open(local_prediction_file, 'w') as outfile:
        json.dump(my_predictions, outfile, indent=2)
    if upload:
        upload_submission(local_file=local_prediction_file, task='cart')
