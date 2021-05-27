import numpy as np
import pandas as pd
import random
import time
import math

import intrusion_classify
import classify_common


# 创建优化XGBoost的PSO类
class PSOForXGBoost:
    # 构造方法
    def __init__(self, w_max, w_min, c1, c2, max_iter, ps_size, dim):
        # 惯性权重
        self.w_max = w_max
        self.w_min = w_min
        # 加速度因子
        self.c1 = c1
        self.c2 = c2
        # 迭代次数
        self.max_iter = max_iter
        # 种群规模
        self.ps_size = ps_size
        # 维度
        self.dim = dim
        # 粒子群位置和速度
        self.p_pos = np.zeros((ps_size, dim))
        self.p_vel = np.zeros((ps_size, dim))
        # 粒子最优位置
        self.p_best_pos = np.zeros((ps_size, dim))
        # 全局最优位置
        self.g_best_pos = None
        # 粒子最优适应度值
        self.p_best_fit = np.zeros(ps_size)
        # 全局最优适应度值
        self.g_best_fit = None
        # 记录每一代的最优位置
        self.pos_record = np.zeros((max_iter + 1, dim))
        # 记录每一代的最优适应度值
        self.fit_record = np.zeros(max_iter + 1)
        # 最后的收敛结果
        self.final_result = None

    # 种群初始化
    def initial(self, classify_dataset, attack_types):
        # 生成每个粒子的初始位置和初始速度（初始速度为0）
        for i in range(self.ps_size):
            # 随机生成XGBoost的每个参数
            self.p_pos[i] = classify_common.xgb_para_init(self.p_pos[i])
            # 记录粒子初始位置和适应度值
            self.p_best_pos[i] = self.p_pos[i]
            self.p_best_fit[i] = intrusion_classify.xgb_classify(self.p_pos[i], classify_dataset, attack_types)
        # 记录全局初始最优位置和适应度值
        max_index = np.where(self.p_best_fit == np.max(self.p_best_fit))[0][0]
        self.g_best_fit = np.max(self.p_best_fit)
        self.g_best_pos = self.p_pos[max_index]
        self.pos_record[0] = self.g_best_pos.copy()    # deep copy
        self.fit_record[0] = self.g_best_fit
        print('初始最优位置和适应度值为：')
        print(self.g_best_pos)
        print(self.g_best_fit)

    # 迭代寻优
    def optimal(self, classify_dataset, attack_types):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 计算当前惯性权重w的值
            w = (self.w_max + (self.max_iter - iter_count) * (self.w_max - self.w_min)) / self.max_iter
            # w = random.uniform(self.w_min, self.w_max)

            # 更新粒子位置和速度
            for i in range(self.ps_size):
                # 粒子速度更新
                self.p_vel[i] = w * self.p_vel[i] + \
                                self.c1 * random.uniform(0, 1) * (self.p_best_pos[i] - self.p_pos[i]) + \
                                self.c2 * random.uniform(0, 1) * (self.g_best_pos - self.p_pos[i])

                # 粒子位置更新
                self.p_pos[i] = self.p_pos[i] + self.p_vel[i]

                # 判断各参数是否越界
                self.p_pos[i] = classify_common.xgb_bound_check(self.p_pos[i])

                # 计算当前粒子的适应度值
                curr_fit = intrusion_classify.xgb_classify(self.p_pos[i], classify_dataset, attack_types)

                # 根据粒子适应度值判断是否更新粒子以及全局的最优位置和适应度值
                if curr_fit > self.p_best_fit[i]:
                    # 更新粒子最优位置和适应度值
                    self.p_best_fit[i] = curr_fit
                    self.p_best_pos[i] = self.p_pos[i]
                    if self.p_best_fit[i] > self.g_best_fit:
                        # 更新全局最优位置和适应度值
                        self.g_best_fit = self.p_best_fit[i]
                        self.g_best_pos = self.p_best_pos[i]

            # 输出本次迭代的全局最优位置和适应度值
            print('当前迭代次数：', iter_count + 1)
            print(self.g_best_pos)
            print(self.g_best_fit)
            # 记录本次迭代的最优位置和适应度值
            self.pos_record[iter_count + 1] = self.g_best_pos.copy()    # deep copy
            self.fit_record[iter_count + 1] = self.g_best_fit
































