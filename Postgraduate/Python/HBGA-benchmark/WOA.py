import numpy as np
import ea_common
import random

import math


# 创建WOA类
class WOA:
    # 构造函数
    def __init__(self, size, dim, pos_max, pos_min, max_iter, func_no, a, b):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 搜索域上限
        self.pos_max = pos_max
        self.pos_min = pos_min
        # 迭代次数
        self.max_iter = max_iter
        # benchmark函数编号
        self.func_no = func_no
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
        # 记录每一代的最优适应度值
        self.fit_record = np.zeros(max_iter + 1)
        # 最后的收敛结果
        self.final_result = None

    # 种群初始化
    def initial(self):
        # 随机生成每个个体的初始位置
        for i in range(self.size):
            for j in range(self.dim):
                self.pos[i, j] = random.uniform(self.pos_min, self.pos_max)
            # 记录个体的初始适应度值
            self.fit[i] = ea_common.func_eval(self.pos[i], self.func_no)
        # 记录初始全局最优下标、位置和适应度值
        min_index = np.argsort(self.fit)[0]
        self.g_best_pos = self.pos[min_index].copy()    # deep copy
        self.g_best_fit = self.fit[min_index]
        self.fit_record[0] = self.g_best_fit
        # print('初始最优位置和适应度值为：')
        # print(self.g_best_pos)
        # print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
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

                # 判断新生成的位置是否越界
                for j in range(self.dim):
                    self.pos[i, j] = ea_common.bound_check(self.pos[i, j], self.pos_max, self.pos_min)

                # 计算当前个体的适应度值
                curr_fit = ea_common.func_eval(self.pos[i], self.func_no)

                # 如果当前个体的适应度值优于全局最优适应度值
                if curr_fit < self.g_best_fit:
                    # 替换全局最优位置和最优适应度值
                    self.g_best_pos = self.pos[i].copy()    # deep copy
                    self.g_best_fit = curr_fit

            # 本次迭代结束
            # 本次迭代结束，判断是否提前收敛
            if self.g_best_fit < 1e-8:
                # 若最优值小于1e-8则认为函数已经收敛
                print('--------本次迭代提前收敛于：', iter_count)
                break
            # 输出本次迭代的全局最优位置和适应度值
            # print('当前迭代次数：', iter_count + 1)
            # print(self.g_best_pos)
            # print(self.g_best_fit)
            # 记录本次迭代的最优适应度值
            self.fit_record[iter_count + 1] = self.g_best_fit

        # 迭代寻优结束，记录最终结果
        self.final_result = self.fit_record[-1]


# woa = WOA(size=100, dim=10, pos_max=100, pos_min=-100, max_iter=500, func_no=1, a=2, b=1)
# woa.initial()
# woa.optimal()
# ea_common.curve(woa.max_iter, woa.fit_record)

















