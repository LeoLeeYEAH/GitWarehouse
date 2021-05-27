import numpy as np
import pandas as pd
import random
import time

import matplotlib.pyplot as plt


# 随机生成XGBoost的每个参数
def xgb_para_init(pos):
    # learning_rate
    pos[0] = random.uniform(0, 1)
    # gamma
    pos[1] = random.uniform(0, 1)
    # max_depth
    pos[2] = random.randint(1, 20)
    # min_child_weight
    pos[3] = random.uniform(0, 10)
    # subsample
    pos[4] = random.uniform(0, 1)
    # colsample_bytree
    pos[5] = random.uniform(0, 1)

    return pos


# XGBoost参数取值范围越界检查
def xgb_bound_check(pos):
    # learning_rate
    if pos[0] < 0:
        pos[0] = 0
    if pos[0] > 1:
        pos[0] = 1
    # gamma
    if pos[1] < 0:
        pos[1] = 0
    # max_depth
    pos[2] = round(pos[2])  # max_depth必须是整数
    if pos[2] < 1:
        pos[2] = 1
    if pos[2] > 20:
        pos[2] = 20
    # min_child_weight
    if pos[3] < 0:
        pos[3] = 0
    # subsample
    if pos[4] <= 0:
        pos[4] = 0.01
    if pos[4] > 1:
        pos[4] = 1
    # colsample_bytree
    if pos[5] <= 0:
        pos[5] = 0.01
    if pos[5] > 1:
        pos[5] = 1

    return pos


# 绘制提升曲线
def curve(max_iter, fit_record, algo_name):
    # 为储存结果的文件生成时间戳
    ts = time.strftime("%Y%m%d%H%M%S")
    fig_name = 'result' + '_' + ts + '_' + algo_name + '.png'
    # 设置绘图参数
    plt.title("Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.plot(np.arange(max_iter + 1), [v for v in fit_record])
    # 图片存到本地
    plt.savefig(fig_name)


# 将数据转存到CSV
def to_csv(fit_record, pos_record, algo_name):
    # 为储存结果的文件生成时间戳
    ts = time.strftime("%Y%m%d%H%M%S")
    file_name = 'result' + '_' + ts + '_' + algo_name + '.csv'
    # 将需要转存的数据转为DataFrame
    fit = pd.DataFrame(fit_record)
    pos = pd.DataFrame(pos_record)
    # 拼接适应度值和参数信息
    data = pd.concat([fit, pos], axis=1)
    # 转存到CSV文件
    columns = ['fitness', 'learning_rate', 'gamma', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree']
    data.to_csv(file_name, header=columns)






















