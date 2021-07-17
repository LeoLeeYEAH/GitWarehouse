import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from xgboost import XGBClassifier

import data_set_preprocess


# 编程期间辅助设置
# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
# pd.set_option('display.max_rows', None)


# XGBoost
def xgb_classify(xgb_para, classify_dataset, attack_types):
    # 获取XGBoost参数
    # 基本的6个参数
    learning_rate = xgb_para[0]
    gamma = xgb_para[1]
    max_depth = int(xgb_para[2])
    min_child_weight = xgb_para[3]
    subsample = xgb_para[4]
    colsample_bytree = xgb_para[5]
    # 默认的额外参数
    colsample_bylevel = 1
    colsample_bynode = 1
    max_delta_step = 0
    reg_lambda = 1
    reg_alpha = 0
    # 优化的额外参数
    if len(xgb_para) > 6:
        colsample_bylevel = xgb_para[6]
        colsample_bynode = xgb_para[7]
        max_delta_step = xgb_para[8]
    if len(xgb_para) > 9:
        reg_lambda = xgb_para[9]
        reg_alpha = xgb_para[10]
    # 设置XGBoost参数
    xgb = XGBClassifier(tree_method='gpu_hist',
                        learning_rate=learning_rate,
                        gamma=gamma,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        colsample_bylevel=colsample_bylevel,
                        colsample_bynode=colsample_bynode,
                        max_delta_step=max_delta_step,
                        reg_lambda=reg_lambda,
                        reg_alpha=reg_alpha)

    # 获取要进行分类的数据集
    # NSL-KDD
    # classify_dataset, attack_types = data_set_preprocess.get_nsl_kdd()

    # 设置训练集和测试集
    train_set_x = classify_dataset.train_set_x
    train_set_y = classify_dataset.train_set_y
    test_set_x = classify_dataset.test_set_x
    test_set_y = classify_dataset.test_set_y

    # 开始XGBoost拟合
    xgb.fit(train_set_x, train_set_y)
    # 利用XGBoost分类
    xgb_predict = xgb.predict(test_set_x)

    # 获取本次分类的各项指标
    macro_precision = precision_score(test_set_y, xgb_predict, average='macro')
    macro_recall = recall_score(test_set_y, xgb_predict, average='macro')
    macro_f1_score = f1_score(test_set_y, xgb_predict, average='macro')
    # Classification Report
    # print('XGBoost Classification Report:\n')
    # print(classification_report(test_set_y, xgb_predict, target_names=attack_types))
    # print('\n')

    # 返回设定的fitness值
    return macro_f1_score


# xgb_classify([0.3, 0, 6, 1, 1, 1])


# 查看程序运行时间
# print(time.perf_counter())













