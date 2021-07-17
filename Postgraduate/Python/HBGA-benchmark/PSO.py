import numpy as np
import pandas as pd
import random
import ea_common
import matplotlib.pyplot as plt
import time
import math


# 创建PSO类
class PSO:
    # 构造方法
    def __init__(self, w_max, w_min, c1, c2, pos_max, pos_min, vel_max, vel_min, max_iter, ps_size, dim, func_no):
        # 惯性权重
        self.w_max = w_max
        self.w_min = w_min
        # 加速度因子
        self.c1 = c1
        self.c2 = c2
        # 搜索域
        self.pos_max = pos_max
        self.pos_min = pos_min
        # 速度域
        self.vel_max = vel_max
        self.vel_min = vel_min
        # 迭代次数
        self.max_iter = max_iter
        # 种群规模
        self.ps_size = ps_size
        # 维度
        self.dim = dim
        # benchmark函数编号
        self.func_no = func_no
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
        # 记录每一代的最优适应度值
        self.fit_record = np.zeros(max_iter + 1)
        # 最后的收敛结果
        self.final_result = None

    # 种群初始化
    def initial(self):
        # 随机生成每个粒子的初始位置和初始速度
        for i in range(self.ps_size):
            for j in range(self.dim):
                self.p_pos[i, j] = random.uniform(self.pos_min, self.pos_max)
                self.p_vel[i, j] = random.uniform(self.vel_min, self.vel_max)
            # 记录粒子初始位置和适应度值
            self.p_best_pos[i] = self.p_pos[i]
            self.p_best_fit[i] = ea_common.func_eval(self.p_pos[i], self.func_no)
        # 记录全局初始最优位置和适应度值
        min_index = np.where(self.p_best_fit == np.min(self.p_best_fit))[0][0]
        self.g_best_fit = np.min(self.p_best_fit)
        self.g_best_pos = self.p_pos[min_index]
        self.fit_record[0] = self.g_best_fit
        # print('初始最优位置和适应度值为：')
        # print(self.g_best_pos)
        # print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 计算当前惯性权重w的值
            # w = (self.w_max + (self.max_iter - iter_count) * (self.w_max - self.w_min)) / self.max_iter
            w = random.uniform(self.w_min, self.w_max)

            # 更新粒子位置和速度
            for i in range(self.ps_size):
                # 粒子速度更新
                self.p_vel[i] = w * self.p_vel[i] + \
                                self.c1 * random.uniform(0, 1) * (self.p_best_pos[i] - self.p_pos[i]) + \
                                self.c2 * random.uniform(0, 1) * (self.g_best_pos - self.p_pos[i])
                # 判断粒子速度是否超过边界
                for j in range(self.dim):
                    self.p_vel[i, j] = ea_common.bound_check(self.p_vel[i, j], self.vel_max, self.vel_min)
                # 粒子位置更新
                self.p_pos[i] = self.p_pos[i] + self.p_vel[i]
                # 判断粒子位置是否超过边界
                for j in range(self.dim):
                    self.p_pos[i, j] = ea_common.bound_check(self.p_pos[i, j], self.pos_max, self.pos_min)
                # 计算当前粒子的适应度值
                curr_fit = ea_common.func_eval(self.p_pos[i], self.func_no)
                # 根据粒子适应度值判断是否更新粒子以及全局的最优位置和适应度值
                if curr_fit < self.p_best_fit[i]:
                    # 更新粒子最优位置和适应度值
                    self.p_best_fit[i] = curr_fit
                    self.p_best_pos[i] = self.p_pos[i]
                    if self.p_best_fit[i] < self.g_best_fit:
                        # 更新全局最优位置和适应度值
                        self.g_best_fit = self.p_best_fit[i]
                        self.g_best_pos = self.p_best_pos[i]

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


# # 设置PSO的各项参数
# pso = PSO(w_max=0.9, w_min=0.4, c1=2, c2=2, pos_max=100, pos_min=-100, vel_max=1, vel_min=-1,
#           max_iter=10, ps_size=100, dim=10, func_no=1)
#
# for i in range(1):
#     # 初始化PSO
#     pso.initial()
#     # 开始迭代
#     pso.optimal()
#     # 收敛曲线
#     # pso.curve()
#     # 收敛结果
#     pso.result()






































