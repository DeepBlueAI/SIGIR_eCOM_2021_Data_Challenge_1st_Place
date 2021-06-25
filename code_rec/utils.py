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

def downcast(series,accuracy_loss = True, min_float_type='float16'):
    if series.dtype == np.int64:
        ii8 = np.iinfo(np.int8)
        ii16 = np.iinfo(np.int16)
        ii32 = np.iinfo(np.int32)
        max_value = series.max()
        min_value = series.min()
        
        if max_value <= ii8.max and min_value >= ii8.min:
            return series.astype(np.int8)
        elif  max_value <= ii16.max and min_value >= ii16.min:
            return series.astype(np.int16)
        elif max_value <= ii32.max and min_value >= ii32.min:
            return series.astype(np.int32)
        else:
            return series
        
    elif series.dtype == np.float64:
        fi16 = np.finfo(np.float16)
        fi32 = np.finfo(np.float32)
        
        if accuracy_loss:
            max_value = series.max()
            min_value = series.min()
            if np.isnan(max_value):
                max_value = 0
            
            if np.isnan(min_value):
                min_value = 0
                
            if min_float_type=='float16' and max_value <= fi16.max and min_value >= fi16.min:
                return series.astype(np.float16)
            elif max_value <= fi32.max and min_value >= fi32.min:
                return series.astype(np.float32)
            else:
                return series
        else:
            tmp = series[~pd.isna(series)]
            if(len(tmp)==0):
                return series.astype(np.float16)
            
            if (tmp == tmp.astype(np.float16)).sum() == len(tmp):
                return series.astype(np.float16)
            elif (tmp == tmp.astype(np.float32)).sum() == len(tmp):
                return series.astype(np.float32)
           
            else:
                return series
            
    else:
        return series

def upload_res(sess2res):
    cnt_preds = 0
    my_predictions = []
    # get all possible SKUs in the model, as a back-up choice
    test_file = '../data/test/rec_test_phase_2.json'
    with open(test_file) as json_file:
        # read the test cases from the provided file
        test_queries = json.load(json_file)
    # loop over the records and predict the next event
    all_sub_num = 0
    ban_sub_num = 0
    for t in test_queries:
        # append the label - which needs to be a list
        session_id = t['query'][0]['session_id_hash']
        _products_in_clicked = []
        for _ in t['query']:
            if _["clicked_skus_hash"]:
                _products_in_clicked.extend(_["clicked_skus_hash"])
        _products_in_session = [_["product_sku_hash"] for _ in t['query'] if _["product_sku_hash"]]
        ban_product = _products_in_clicked+_products_in_session
        if session_id in sess2res:
            _pred = {'label': sess2res[session_id]}
            for i in sess2res[session_id]:
                if i in ban_product:
                    ban_sub_num += 1
            all_sub_num += len(sess2res[session_id])
            cnt_preds += 1
        else:
            _pred = {'label': []}
        
        # append prediction to the final list
        my_predictions.append(_pred)

    print('sub product num: ', all_sub_num, ban_sub_num)
    # check for consistency
    assert len(my_predictions) == len(test_queries)
    # print out some "coverage"
    print("Predictions made in {} out of {} total test cases".format(cnt_preds, len(test_queries)))
    from dotenv import load_dotenv
    load_dotenv(verbose=True, dotenv_path='upload.env.local')
    EMAIL = os.getenv('EMAIL', None) # the e-mail you used to sign up
    assert EMAIL is not None
    local_prediction_file = '../prediction_result/{}_{}.json'.format(EMAIL.replace('@', '_'), round(time.time() * 1000))
    # dump to file
    with open(local_prediction_file, 'w') as outfile:
        json.dump(my_predictions, outfile, indent=2)
    # finally, upload the test file using the provided script
    upload_submission(local_file=local_prediction_file, task='rec')