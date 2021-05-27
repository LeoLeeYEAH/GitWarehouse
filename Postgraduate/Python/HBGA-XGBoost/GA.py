import numpy as np
import random
import math

import intrusion_classify
import classify_common


# 创建优化XGBoost的GA类
class GAForXGBoost:
    # 构造方法
    def __init__(self, size, dim, max_iter,
                 select_type, cross_type, cross_rate, mutation_type, mutation_rate, keep_elite):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 迭代次数
        self.max_iter = max_iter
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
            # 随机生成XGBoost的每个参数
            self.pos[i] = classify_common.xgb_para_init(self.pos[i])
            # 记录个体的初始适应度值
            self.fit[i] = intrusion_classify.xgb_classify(self.pos[i], classify_dataset, attack_types)
        # 记录初始全局最优下标、位置和适应度值
        max_index = np.argsort(-self.fit)[0]
        self.g_best_index = max_index
        self.g_best_pos = self.pos[max_index].copy()    # deep copy
        self.g_best_fit = self.fit[max_index]
        self.pos_record[0] = self.g_best_pos.copy()    # deep copy
        self.fit_record[0] = self.g_best_fit
        # 记录初始保留精英及其下标
        self.keep_elite_index = np.argsort(-self.fit)[0:self.keep_elite]
        print('初始最优位置和适应度值为：')
        print(self.g_best_pos)
        print(self.g_best_fit)

    # 迭代寻优
    def optimal(self, classify_dataset, attack_types):
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
                # 判断各参数是否越界
                self.pos[i] = classify_common.xgb_bound_check(self.pos[i])
                # 计算适应度值
                self.fit[i] = intrusion_classify.xgb_classify(self.pos[i], classify_dataset, attack_types)

            # 更新全局最优下标、位置和适应度值
            max_index = np.argsort(-self.fit)[0]
            self.g_best_index = max_index
            self.g_best_pos = self.pos[max_index].copy()    # deep copy
            self.g_best_fit = self.fit[max_index]
            # 更新需要保留的精英个体下标
            self.keep_elite_index = np.argsort(-self.fit)[0:self.keep_elite]
            # 输出本次迭代的全局最优位置和适应度值
            print('当前迭代次数：', iter_count + 1)
            print(self.g_best_pos)
            print(self.g_best_fit)
            # 记录本次迭代的最优适应度值
            self.pos_record[iter_count + 1] = self.g_best_pos.copy()    # deep copy
            self.fit_record[iter_count + 1] = self.g_best_fit

    # 轮盘赌选择
    def roulette_wheel_selection(self):
        # 计算每个个体的选择概率
        select_prob = self.fit / np.sum(self.fit)
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
        # 创建数组储存尚未参与交叉个体的下标
        uncross_index = list(range(0, self.size))
        # 为保留精英个体，移除精英个体的下标
        for index in self.keep_elite_index:
            uncross_index.remove(index)
        # 开始单点交叉循环
        while len(uncross_index) > 1:
            # 随机选择两个尚未参与交叉的个体
            chosen = random.sample(uncross_index, 2)
            # 将选中的个体移除出尚未参与交叉的数组
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
                    # 判断是哪个参数需要变异
                    if mutation_index == 0:
                        # learning_rate
                        self.pos[i, 0] = random.uniform(0, 1)
                    if mutation_index == 1:
                        # gamma
                        self.pos[i, 1] = random.uniform(0, 1)
                    if mutation_index == 2:
                        # max_depth
                        self.pos[i, 2] = random.randint(1, 20)
                    if mutation_index == 3:
                        # min_child_weight
                        self.pos[i, 3] = random.uniform(0, 10)
                    if mutation_index == 4:
                        # subsample
                        self.pos[i, 4] = random.uniform(0, 1)
                    if mutation_index == 5:
                        # colsample_bytree
                        self.pos[i, 5] = random.uniform(0, 1)










