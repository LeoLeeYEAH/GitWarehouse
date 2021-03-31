import numpy as np
import random
import matlabFE
import time


# 创建PSO类
class PSO:
    def __init__(self, w_max, w_min, c1, c2, max_iter, ps_size, dim, func_no):
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
        # benchmark函数编号
        self.func_no = func_no
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
        self.g_best_fit = None

    # 种群初始化
    def initial(self):
        for i in range(self.ps_size):
            for j in range(self.dim):
                self.p_pos[i][j] = random.uniform(-100, 100)
                self.p_vel[i][j] = random.uniform(-1, 1)
            # 计算初始适应度
            # temp = self.fit_function(self.p_pos[i])
            # 记录粒子初始位置和适应度
            self.p_best_pos[i] = self.p_pos[i]
            self.p_best_fit[i] = func_eval(self.p_pos[i], self.func_no)
        # 记录全局初始最优位置和适应度
        min_index = int(np.where(self.p_best_fit == np.min(self.p_best_fit))[0])
        self.g_best_fit = np.min(self.p_best_fit)
        self.g_best_pos = self.p_pos[min_index]
        print('初始最优位置和适应度值为：')
        print(self.g_best_pos)
        print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 迭代寻优
        for iter_count in range(self.max_iter):
            # 判断是否提前收敛
            if self.g_best_fit < 1e-8:
                # 若最优值小于1e-8则认为函数已经收敛
                break
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
                    if self.p_vel[i][j] > 1:
                        self.p_vel[i][j] = 1
                    if self.p_vel[i][j] < -1:
                        self.p_vel[i][j] = -1
                # 粒子位置更新
                self.p_pos[i] = self.p_pos[i] + self.p_vel[i]
                # 判断粒子位置是否超过边界
                for j in range(self.dim):
                    if self.p_pos[i][j] > 100:
                        self.p_pos[i][j] = 100
                    if self.p_pos[i][j] < -100:
                        self.p_pos[i][j] = -100
                # 计算当前粒子的适应度
                curr_fit = func_eval(self.p_pos[i], self.func_no)
                # 根据粒子适应度判断是否更新粒子以及全局的最优位置和适应度
                if curr_fit < self.p_best_fit[i]:
                    # 更新粒子最优位置和适应度
                    self.p_best_fit[i] = curr_fit
                    self.p_best_pos[i] = self.p_pos[i]
                    if self.p_best_fit[i] < self.g_best_fit:
                        # 更新全局最优位置和适应度
                        self.g_best_fit = self.p_best_fit[i]
                        self.g_best_pos = self.p_best_pos[i]
                        print('当前迭代次数：', iter_count + 1)
                        print(self.g_best_pos)
                        print(self.g_best_fit)

    # 利用benchmark函数计算适应度值
    # @staticmethod
    # def fit_function(x):
    #     result = matlabFE.func_eval(x, 1)
    #     return result


# Function Evaluation 适应度值评估
def func_eval(pos, func_no):
    # 若 func_no > 0 则是利用CEC2015 benchmark函数进行计算
    if func_no > 0:
        result = matlabFE.func_eval(pos, func_no)
    # 若 func_no = 0 则说明是其他适应度值计算方式
    else:
        result = 0
    return result


# 设置PSO的各项参数
pso = PSO(w_max=0.9, w_min=0.4, c1=2, c2=2, max_iter=50, ps_size=10, dim=10, func_no=1)
# 初始化PSO
pso.initial()
# 开始迭代
pso.optimal()







































