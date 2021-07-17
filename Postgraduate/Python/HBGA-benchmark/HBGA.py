import numpy as np
import random
import ea_common
import matplotlib.pyplot as plt
import math


# 创建HBGA类
class HBGA:
    # 构造方法
    def __init__(self, size, dim, max_self, r_max, r_min, max_iter, func_no):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 自交阈值
        self.max_self = max_self
        # 搜索域上限
        self.r_max = r_max
        # 搜索域下限
        self.r_min = r_min
        # 最大迭代次数
        self.max_iter = max_iter
        # benchmark函数编号
        self.func_no = func_no
        # 个体位置、自交次数和适应度值
        self.pos_time_fit = np.zeros((size, dim + 2))
        '''
            说明：
            self.pos_time_fit为储存个体位置、自交次数和适应度值的二维数组
            根据数组下标，具体储存安排如下
            0 ~ dim-1   --> 问题的解
            dim         --> 自交次数
            dim+1 or -1 --> 适应度值
        '''
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
                self.pos_time_fit[i, j] = random.uniform(self.r_min, self.r_max)
            # 记录个体的初始适应度值
            self.pos_time_fit[i, -1] = ea_common.func_eval(self.pos_time_fit[i, :self.dim], self.func_no)
        # 根据适应度值对种群进行排序
        self.pos_time_fit = self.pos_time_fit[np.argsort(self.pos_time_fit[:, -1])]
        # 记录全局初始最优位置和适应度
        self.g_best_pos = self.pos_time_fit[0, :self.dim].copy()    # deep copy
        self.g_best_fit = self.pos_time_fit[0, -1]
        self.fit_record[0] = self.g_best_fit
        # print('初始最优位置和适应度值为：')
        # print(self.g_best_pos)
        # print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 创建保持系 恢复系 不育系索引
            maintainer_index = np.arange(0, int(self.size/3))
            restorer_index = np.arange(int(self.size/3), int(self.size/3)*2)
            sterile_index = np.arange(int(self.size/3)*2, self.size)

            # 保持系与不育系进行杂交
            for index in sterile_index:
                # 初始化杂交后产生的新不育个体
                new_sterile = np.zeros(self.dim + 2)
                # 随机选择一个保持系个体
                selected_maintainer = self.pos_time_fit[random.choice(maintainer_index)]
                # 随机选择一个不育系个体
                selected_sterile = self.pos_time_fit[random.choice(sterile_index)]
                # 开始杂交过程
                for i in range(self.dim):
                    # 生成随机数r1 r2
                    # r1 = random.uniform(-1, 1)    # 原始
                    r1 = random.uniform(0, 1)    # 改进
                    # r2 = random.uniform(-1, 1)
                    # 根据所定义的公式进行杂交
                    # new_sterile[i] = (r1 * selected_maintainer[i] + r2 * selected_sterile[i]) / (r1 + r2)    # 原始
                    new_sterile[i] = (r1 * selected_maintainer[i] + (1 - r1) * selected_sterile[i])   # 改进1
                    # 判断个体位置是否会越界
                    new_sterile[i] = ea_common.bound_check(new_sterile[i], self.r_max, self.r_min)
                # 计算新个体的适应度值
                new_sterile[-1] = ea_common.func_eval(new_sterile[:self.dim], self.func_no)
                # 如果新个体的适应度值优于当前不育系个体，则替换之
                if new_sterile[-1] < self.pos_time_fit[index, -1]:
                    self.pos_time_fit[index] = new_sterile
            # 杂交结束后更新全局最优位置和最优适应度值
            best_index = np.where(self.pos_time_fit == np.min(self.pos_time_fit[:, -1]))[0][0]
            self.g_best_pos = self.pos_time_fit[best_index, :self.dim].copy()    # deep copy
            self.g_best_fit = self.pos_time_fit[best_index, -1]

            # 恢复系自交或重置
            for index in restorer_index:
                # 判断当前个体自交次数是否已达上限
                if self.pos_time_fit[index, self.dim] < self.max_self:
                    # 若自交次数未达上限
                    # 初始化自交后产生的新恢复个体
                    new_restorer = np.zeros(self.dim + 2)
                    # 开始自交过程
                    for i in range(self.dim):
                        # 随机选择一个恢复系个体（与当前个体不重复）
                        selected_restorer = self.pos_time_fit[random.choice(restorer_index[restorer_index != index])]
                        # 生成随机数r3
                        r3 = random.uniform(0, 1)
                        # 根据所定义的公式进行自交
                        new_restorer[i] = r3 * (self.g_best_pos[i] - selected_restorer[i]) + self.pos_time_fit[index, i]
                        # 判断个体位置是否会越界
                        new_restorer[i] = ea_common.bound_check(new_restorer[i], self.r_max, self.r_min)
                    # 计算新个体的适应度值
                    new_restorer[-1] = ea_common.func_eval(new_restorer[:self.dim], self.func_no)
                    # 判断新生成的个体适应度值是否优于之前的个体
                    if new_restorer[-1] < self.pos_time_fit[index, -1]:
                        # 如若优于，则替换之
                        self.pos_time_fit[index] = new_restorer
                        # 同时该个体自交次数置0
                        self.pos_time_fit[index, self.dim] = 0
                    else:
                        # 如若未优于，则个体自交次数+1
                        self.pos_time_fit[index, self.dim] = self.pos_time_fit[index, self.dim] + 1
                else:
                    # 若自交次数已达上限
                    # 进行重置操作
                    for i in range(self.dim):
                        # 生成随机数r3
                        r3 = random.uniform(0, 1)
                        # 根据所定义的公式进行重置
                        # 原始
                        # self.pos_time_fit[index, i] = self.pos_time_fit[index, i] + random.uniform(self.r_min, self.r_max)
                        # 改进
                        self.pos_time_fit[index, i] = random.uniform(self.r_min, self.r_max)
                    # 重新计算该个体的适应度值
                    self.pos_time_fit[index, -1] = ea_common.func_eval(self.pos_time_fit[index, :self.dim], self.func_no)
                    # 将该个体自交次数置0
                    self.pos_time_fit[index, self.dim] = 0
                # 针对当前个体的操作完成后，更新全局最优位置和最优适应度值
                best_index = np.where(self.pos_time_fit == np.min(self.pos_time_fit[:, -1]))[0][0]
                self.g_best_pos = self.pos_time_fit[best_index, :self.dim].copy()    # deep copy
                self.g_best_fit = self.pos_time_fit[best_index, -1]

            # 当前迭代完成，根据适应度值对种群重新排序
            self.pos_time_fit = self.pos_time_fit[np.argsort(self.pos_time_fit[:, -1])]
            # 更新全局最优位置和最优适应度值
            # self.g_best_pos = self.pos_time_fit[0, :self.dim].copy()    # deep copy
            # self.g_best_fit = self.pos_time_fit[0, -1]
            # 本次迭代结束，判断是否提前收敛
            if self.g_best_fit < 1e-8:
                # 若最优值小于1e-8则认为函数已经收敛
                print('--------本次迭代提前收敛于：', iter_count)
                break
            # 输出全局最优位置和最优适应度值
            # print('当前迭代次数：', iter_count + 1)
            # print(self.g_best_pos)
            # print(self.g_best_fit)
            # 记录本次迭代后的最优适应度值
            self.fit_record[iter_count + 1] = self.g_best_fit

        # 迭代寻优结束，记录最终结果
        self.final_result = self.fit_record[-1]


# 设置HBGA的各项参数
# hbga = HBGA(size=30, dim=10, max_self=60, r_max=100, r_min=-100, max_iter=2500, func_no=1)
























