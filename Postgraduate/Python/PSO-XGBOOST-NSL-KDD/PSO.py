import numpy as np
import random


# 创建PSO类
class PSO:
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
        # 粒子最佳位置
        self.p_best_pos = np.zeros((ps_size, dim))
        # 全局最佳位置
        self.g_best_pos = np.zeros((1, dim))
        # 粒子最佳适应度
        self.p_best_fit = np.zeros(ps_size)
        # 全局最佳适应度
        self.g_best_fit = 0

    # 种群初始化
    def initial(self):
        for i in range(self.ps_size):
            for j in range(self.dim):
                if j == 0:
                    self.p_pos[i][j] = random.uniform(-2, 2)
                if j == 1:
                    self.p_pos[i][j] = random.uniform(-2, 2)
                self.p_vel[i][j] = random.uniform(-0.5, 0.5)
            # 计算初始适应度
            temp = self.fit_function(self.p_pos[i])
            # 记录粒子初始位置和适应度
            self.p_best_pos[i] = self.p_pos[i]
            self.p_best_fit[i] = temp
            # 记录全局初始最优位置和适应度
            if temp > self.g_best_fit:
                self.g_best_fit = temp
                self.g_best_pos = self.p_pos[i]

    # 迭代寻优
    def optimal(self):
        # 迭代寻优
        for iter_count in range(self.max_iter):
            # 计算当前惯性权重w的值
            w = (self.w_max + (self.max_iter - iter_count) * (self.w_max - self.w_min)) / self.max_iter
            # 更新粒子位置和速度
            for i in range(self.ps_size):
                # 粒子速度更新
                self.p_vel[i] = w * self.p_vel[i] + \
                                self.c1 * random.uniform(0, 1) * (self.p_best_pos[i] - self.p_pos[i]) + \
                                self.c2 * random.uniform(0, 1) * (self.g_best_pos - self.p_pos[i])
                # 判断粒子速度是否超过边界
                for j in range(self.dim):
                    if self.p_vel[i][j] > 0.5:
                        self.p_vel[i][j] = 0.5
                    if self.p_vel[i][j] < -0.5:
                        self.p_vel[i][j] = -0.5
                # 粒子位置更新
                self.p_pos[i] = self.p_pos[i] + self.p_vel[i]
                # 判断粒子位置是否超过边界
                for j in range(self.dim):
                    if self.p_pos[i][j] > 2:
                        self.p_pos[i][j] = 2
                    if self.p_pos[i][j] < -2:
                        self.p_pos[i][j] = -2
                # 计算当前粒子的适应度
                temp = self.fit_function(self.p_pos[i])
                # 根据粒子适应度判断是否更新粒子以及全局的最优位置和适应度
                if temp > self.p_best_fit[i]:
                    # 更新粒子最优位置和适应度
                    self.p_best_fit[i] = temp
                    self.p_best_pos[i] = self.p_pos[i]
                    if self.p_best_fit[i] > self.g_best_fit:
                        # 更新全局最优位置和适应度
                        self.g_best_fit = self.p_best_fit[i]
                        self.g_best_pos = self.p_best_pos[i]
                        print(self.g_best_pos)
                        print(self.g_best_fit)

    # 适应度函数
    @staticmethod
    def fit_function(x):
        result = (np.sin(np.sqrt(x[0] * x[0] + x[1] * x[1]))) / np.sqrt(x[0] * x[0] + x[1] * x[1]) + \
                 np.exp((np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])) / 2) - 2.71289
        return result


pso = PSO(w_max=0.9, w_min=0.4, c1=2, c2=2, max_iter=80, ps_size=100, dim=2)

pso.initial()
pso.optimal()











































