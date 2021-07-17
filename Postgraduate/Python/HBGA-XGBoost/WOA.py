import numpy as np
import pandas as pd
import random
import time
import math

import intrusion_classify
import classify_common


# 创建WOA类
class WOAForXGBoost:
    # 构造函数
    def __init__(self, size, dim, max_iter, a, b):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 迭代次数
        self.max_iter = max_iter
        # 线性递减参数a
        self.a = a
        # 螺旋更新参数b
        self.b = b
        # 个体位置信息
        self.pos = np.zeros((size, dim))
        # 个体适应度信息
        self.fit = np.zeros(size)
        # 全局最优位置
        self.g_best_pos = None
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
        # 随机生成每个个体的初始位置
        for i in range(self.size):
            self.pos[i] = classify_common.xgb_para_init(self.pos[i])
            # 记录个体的初始适应度值
            self.fit[i] = intrusion_classify.xgb_classify(self.pos[i], classify_dataset, attack_types)
        # 记录初始全局最优下标、位置和适应度值
        max_index = np.argsort(-self.fit)[0]
        self.g_best_pos = self.pos[max_index].copy()    # deep copy
        self.g_best_fit = self.fit[max_index]
        self.pos_record[0] = self.g_best_pos.copy()  # deep copy
        self.fit_record[0] = self.g_best_fit
        print('初始最优位置和适应度值为：')
        print(self.g_best_pos)
        print(self.g_best_fit)

    # 迭代寻优
    def optimal(self, classify_dataset, attack_types):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 确定线性递减参数a1 a2
            a1 = self.a - iter_count * (self.a / self.max_iter)
            a2 = -1 + iter_count * (-1 / self.max_iter)

            # 利用每个个体开始寻优
            for i in range(self.size):
                # 计算当前个体的参数A
                A = 2 * a1 * random.uniform(0, 1) - a1
                # 计算当前个体的参数C
                C = 2 * random.uniform(0, 1)
                # 生成随机数p
                p = random.uniform(0, 1)

                # 判断当前所要进行的操作
                if p < 0.5:
                    # Encircling Prey 或 Search for Prey
                    if abs(A) < 1:
                        # Encircling Prey
                        # 对个体中的每个位置进行操作
                        for j in range(self.dim):
                            # 计算参数D
                            D = abs(C * self.g_best_pos[j] - self.pos[i, j])
                            # 更新后的位置
                            self.pos[i, j] = self.g_best_pos[j] - A * D
                    else:
                        # Search for Prey
                        # 随机选择一个个体
                        rand_index = random.randint(0, self.size - 1)
                        # 对个体中的每个位置进行操作
                        for j in range(self.dim):
                            # 计算参数D
                            D = abs(C * self.pos[rand_index, j] - self.pos[i, j])
                            # 更新后的位置
                            self.pos[i, j] = self.pos[rand_index, j] - A * D
                else:
                    # Attacking
                    # 生成随机数l
                    l = (a2 - 1) * random.uniform(0, 1) + 1
                    # 对个体中的每个位置进行操作
                    for j in range(self.dim):
                        # 计算参数D
                        D = abs(self.g_best_pos[j] - self.pos[i, j])
                        # 更新后的位置
                        self.pos[i, j] = D * np.exp(self.b * l) * np.cos(2 * np.pi * l) + self.g_best_pos[j]

                # 判断新生成的个体位置是否越界
                self.pos[i] = classify_common.xgb_bound_check(self.pos[i])

                # 计算当前个体的适应度值
                curr_fit = intrusion_classify.xgb_classify(self.pos[i], classify_dataset, attack_types)

                # 如果当前个体的适应度值优于全局最优适应度值
                if curr_fit > self.g_best_fit:
                    # 替换全局最优位置和最优适应度值
                    self.g_best_pos = self.pos[i].copy()    # deep copy
                    self.g_best_fit = curr_fit

            # 本次迭代结束
            # 输出本次迭代的全局最优位置和适应度值
            print('当前迭代次数：', iter_count + 1)
            print(self.g_best_pos)
            print(self.g_best_fit)
            # 记录本次迭代的最优位置和适应度值
            self.pos_record[iter_count + 1] = self.g_best_pos.copy()  # deep copy
            self.fit_record[iter_count + 1] = self.g_best_fit


















