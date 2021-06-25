#%%
import collections
import gc
import json
import os
import pickle
import random
import time
from collections import Counter, defaultdict, deque
from os.path import exists

import lightgbm as lgb
import matplotlib as mlb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# from CONSTANT import *


class LGBBinary:
    def __init__(self, model=None):
        self.num_boost_round = 1000
        self.grow_boost_round = 200
        self.model = model
        self.params = {
            "objective": "binary",
            'n_jobs': 30,
            'metric': 'auc',
            'max_depth': 7,
            'eta': 0.03,
            'max_bin': 255,
            'min_child_samples': 20,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            # 'bagging_freq': 10,
            # 'scale_pos_weight': 1.5,
            'num_leaves':32,
            # 'neg_bagging_fraction': 0.1,
            # 'lambda_l1': 3.8,
            # 'num_leaves': 60,
            # 'first_metric_only': True,
            # 'metric_freq': 10,
            'verbose':-1
        }
        
    def load_model(self, model_path):
        self.model = lgb.Booster(model_file=model_path)
        
    def run_simple_model(self,
                         X_train,
                         y_train,
                         params=None,
                         categorical_feature=None):
        if params is None:
            params = {
                'objective': 'binary',
                'n_jobs': 20,
                'metric': 'None',
                'max_depth': 5,
                'eta': 0.015,
                'num_leaves': 31,
                'verbosity': -1,
            }
        lgb_train = lgb.Dataset(X_train, label=y_train)
        train_params = {
            'train_set': lgb_train,
            'num_boost_round': 50,
        }
        if categorical_feature is not None:
            train_params['categorical_feature'] = categorical_feature
        model = lgb.train(params, **train_params)
        return model
    
    # def mrr_score(self, preds, train_data):
    #     y_true = train_data.get_label()
    #     if len(y_true) == len(self.train_query):
    #         query_id = self.train_query
    #     else:
    #         query_id = self.valid_query
    #     df = pd.DataFrame({'query_id': query_id, 'y_true': y_true, 'preds': preds})
    #     df = df.sort_values(by='preds', ascending=False)
    #     df['rank'] = df.groupby('query_id')['preds'].cumcount()+1
    #     df = df[df['y_true']==1]
    #     df = df.drop_duplicates(subset='query_id', keep='first')
    #     mrr = (1/df['rank']).mean()
    #     return 'mrr', mrr, True

    def micro_f1(self, preds, train_data):
        y_true = train_data.get_label()
        # 先就用f1
        return 'f1', metrics.f1_score(y_true, preds), True
    
    def valid_fit(self, X_train, X_valid, y_train, y_valid, weight=None):

        self.feature_name = X_train.columns
        if weight is not None:
            lgb_train = lgb.Dataset(X_train, label=y_train, weight=weight)
        else:
            lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_eval = lgb.Dataset(X_valid, label=y_valid)
        
        self.model = lgb.train(
            self.params, lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=[lgb_train, lgb_eval], valid_names=['train', 'valid'],
            # feval=self.micro_f1,
            early_stopping_rounds=60,
            verbose_eval=50,
        )
        
        self.grow_boost_round = int(self.model.num_trees()*1.2)
        df_imp = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                               'importances': self.model.feature_importance('gain')})
        
        df_imp = df_imp.sort_values('importances', ascending=False)
        print('importants: ', df_imp.head(50))
        preds = self.model.predict(X_valid)
        # self.model.save_model(f'../user_data/model/{int(time.time()*1000)}.model')
        return preds, df_imp
    
    def get_importance(self):
        df_imp = pd.DataFrame({'features': [i for i in self.model.feature_name()],
                               'importances': self.model.feature_importance('gain')})
        df_imp = df_imp.sort_values('importances', ascending=False)
        return df_imp
    
    def fit(self, X, y, weight=None):
        self.feature_name = X.columns
        if weight is not None:
            lgb_train = lgb.Dataset(X, label=y, weight=weight)
        else:
            lgb_train = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            self.params, lgb_train,
            num_boost_round=self.grow_boost_round,
            valid_sets=[lgb_train], valid_names=['train'],
            verbose_eval=50,
        )
        
    def predict(self, test):
        return self.model.predict(test)


class TimeConsistency:
    """
    One interesting trick called "time consistency" is to train a single model
    using a single feature (or small group of features).
    """
    def __init__(self, threshold=0.9):
        self.params = {
            "objective": "binary",
            'n_jobs': 20,
            'metric': 'None',
            'max_depth': 6,
            'eta': 0.015,
            'num_leaves': 40,
            'verbose':-1, 
            'num_boost_round': 50,
        }
        self.col2auc = {}
        self.threshold = threshold

    def _fit(self, X_train, X_valid, y_train, y_valid, verbose=True):
        """train single model for every feature
        Parameters
        ----------
        X_train, X_valid: {pd.DataFrame}, shape(n_samples, n_features)
                          data of train data and valid data.
        y_train, y_valid: {array-like}, shape(n_sample, 1)
                          label of train data and valid data.
        Returns
        -------
        self: TimeConsistency
        """
        feats = X_train.columns
        if verbose:
            print("--------TimeConsistency---------")
        lgb_model = LGBBinary()
        for feat in tqdm(feats) if verbose else feats:
            # lgb_train = lgb.Dataset(X_train[[feat]], label=y_train)
            model, _ = lgb_model.run_simple_model(X_train[[feat]], y_train, self.params)

            train_preds = model.predict(X_train[[feat]])
            fpr, tpr, _ = metrics.roc_curve(y_train, train_preds)
            train_auc = metrics.auc(fpr, tpr)
            
            valid_preds = model.predict(X_valid[[feat]])
            fpr, tpr, _ = metrics.roc_curve(y_valid, valid_preds)
            test_auc = metrics.auc(fpr, tpr)
            
            self.col2auc[feat] = (train_auc, test_auc)
        return self
    
    def fit(self, X, y, sample=True):
        # TODO(wangjin@deepblueai.com): To fit API.
        pass

    def transform(self, X, y=None, verbose=True, max_ratio=0.2):
        """
        select feature according to self.col2auc.
        Parameters
        ----------
        max_ratio: max drop ratio of column num.
        """
        if not self.col2auc:
            raise ValueError(
                "Make sure to call _fit before transform"
            )
        col2score = {}
        drop_col = []
        for col, auc in self.col2auc.items():
            score = max(0, (auc[0]-auc[1])) / (auc[0]-0.5)
            col2score[col] = score
            if score > self.threshold:
                # print(f"maybe to drop column {col} auc is {auc} score is {score}")
                drop_col.append((score, col))
        drop_col.sort()
        max_drop_num = int(X.shape[1]*max_ratio)
        
        drop_col = list(map(lambda x: x[1], drop_col[:max_drop_num]))
        if verbose:
            print(f'drop_col: {drop_col}')
        return X.drop(columns=drop_col)
