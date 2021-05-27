import numpy as np
import ea_common
import random
import matplotlib.pyplot as plt
import math


# 创建GA类
class GA:
    # 构造方法
    def __init__(self, size, dim, pos_max, pos_min, max_iter, func_no,
                 select_type, cross_type, cross_rate, mutation_type, mutation_rate, keep_elite):
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
        # 选择类型
        self.select_type = select_type
        # 交叉类型
        self.cross_type = cross_type
        # 交叉概率
        self.cross_rate = cross_rate
        # 变异类型
        self.mutation_type = mutation_type
        # 变异概率
        self.mutation_rate = mutation_rate
        # 保留的精英数量
        self.keep_elite = keep_elite
        # 保留的精英个体下标
        self.keep_elite_index = np.zeros(keep_elite)
        # 个体位置信息
        self.pos = np.zeros((size, dim))
        # 个体适应度信息
        self.fit = np.zeros(size)
        # 全局最优下标
        self.g_best_index = None
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
        self.g_best_index = min_index
        self.g_best_pos = self.pos[min_index].copy()    # deep copy
        self.g_best_fit = self.fit[min_index]
        self.fit_record[0] = self.g_best_fit
        # 记录初始保留精英及其下标
        self.keep_elite_index = np.argsort(self.fit)[0:self.keep_elite]
        # print('初始最优位置和适应度值为：')
        # print(self.g_best_pos)
        # print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 开始迭代
        for iter_count in range(self.max_iter):
            # 执行选择操作
            if self.select_type == 'rws':
                # 轮盘赌选择
                self.roulette_wheel_selection()

            # 执行交叉操作
            if self.cross_type == 'spc':
                # 单点交叉
                self.single_point_crossover()

            # 执行变异操作
            if self.mutation_type == 'rm':
                # 单点变异
                self.random_mutation()

            # 重新计算适应度值
            for i in range(self.size):
                self.fit[i] = ea_common.func_eval(self.pos[i], self.func_no)

            # 更新全局最优下标、位置和适应度值
            min_index = np.argsort(self.fit)[0]
            self.g_best_index = min_index
            self.g_best_pos = self.pos[min_index].copy()  # deep copy
            self.g_best_fit = self.fit[min_index]
            # 更新需要保留的精英个体下标
            self.keep_elite_index = np.argsort(self.fit)[0:self.keep_elite]
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

    # 轮盘赌选择
    def roulette_wheel_selection(self):
        # 因为求最小值所以需要颠倒适应度数组
        # 先求出当前最大最小适应度值
        max_fit = np.max(self.fit)
        min_fit = np.min(self.fit)
        # 颠倒适应度数组
        select_fit = max_fit + min_fit - self.fit
        # 计算每个个体的选择概率
        select_prob = select_fit / np.sum(select_fit)
        # 创建数组储存新被选中的个体
        selected_pos = np.zeros((self.size, self.dim))
        # 开始轮盘赌选择循环
        for i in range(self.size):
            # 判断当前个体是否是需要保留的精英个体
            if i in self.keep_elite_index:
                # 若当前个体是需要保留的精英个体
                # 直接将该个体添加到被选中个体的数组中
                selected_pos[i] = self.pos[i]
            else:
                # 若当前个体不是最优个体
                # 随机一个0到1之间的随机数
                random_num = random.uniform(0, 1)
                for j in range(self.size):
                    # 通过累计概率判断随机数落到了哪个区间
                    add_prob = np.sum(select_prob[:j + 1])
                    # 如果随机数小于当前累计概率，则说明随机数落在了当前区间
                    if random_num < add_prob:
                        # 添加新选中个体
                        selected_pos[i] = self.pos[j]
                        # 跳出当前循环
                        break
        # 选择过程结束后，用新位置信息数组替换原位置信息数组
        self.pos = selected_pos.copy()  # deep copy

    # 单点交叉
    def single_point_crossover(self):
        # 创建数组储存尚未参与杂交个体的下标
        uncross_index = list(range(0, self.size))
        # 为保留精英个体，移除精英个体的下标
        for index in self.keep_elite_index:
            uncross_index.remove(index)
        # 开始单点交叉循环
        while len(uncross_index) > 1:
            # 随机选择两个尚未参与杂交的个体
            chosen = random.sample(uncross_index, 2)
            # 将选中的个体移除出尚未参与杂交的数组
            uncross_index.remove(chosen[0])
            uncross_index.remove(chosen[1])
            # 根据交叉概率判断本次是否进行交叉
            cross_prob = random.uniform(0, 1)
            if cross_prob < self.cross_rate:
                # 随机要交叉的单点下标
                cross_index = random.randint(0, self.dim - 1)
                # 执行单点交叉
                self.pos[chosen[0], cross_index], self.pos[chosen[1], cross_index] = \
                self.pos[chosen[1], cross_index], self.pos[chosen[0], cross_index]

    # 单点变异
    def random_mutation(self):
        # 开始单点变异循环
        for i in range(self.size):
            # 需要保留的精英个体不参与变异
            if i in self.keep_elite_index:
                continue
            else:
                # 根据变异概率判断本个体是否进行交叉
                mutation_prob = random.uniform(0, 1)
                if mutation_prob < self.mutation_rate:
                    # 随机要变异的单点下标
                    mutation_index = random.randint(0, self.dim - 1)
                    # 执行单点变异
                    self.pos[i, mutation_index] = random.uniform(self.pos_min, self.pos_max)

    # 收敛曲线
    def curve(self):
        # 利用math.log转换适应度值
        fit_record_log = np.zeros(self.max_iter + 1)
        for i in range(self.max_iter + 1):
            # 判断是否提前收敛
            if self.fit_record[i] > 0:
                # 若未提前收敛
                fit_record_log[i] = math.log(self.fit_record[i])
            else:
                # 若已提前收敛
                fit_record_log[fit_record_log == 0] = math.log(1e-8)
        # 绘制收敛曲线
        plt.title("Convergence Curve")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.plot(np.arange(self.max_iter + 1), [v for v in fit_record_log])
        plt.show()

    # 输出收敛结果等各项信息
    def result(self):
        self.final_result = self.fit_record[-1]
        return self.final_result


# # 设置GA的各项参数
# ga = GA(size=100, dim=10, pos_max=100, pos_min=-100, max_iter=500, func_no=1,
#         select_type='rws', cross_type='spc', cross_rate=0.8, mutation_type='rm', mutation_rate=0.05, keep_elite=10)
#
# for i in range(1):
#     # 初始化GA
#     ga.initial()
#     # 开始迭代
#     ga.optimal()
#     # 收敛曲线
#     ga.curve()
#     # 收敛结果
#     # ga.result()












