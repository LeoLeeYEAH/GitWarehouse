import numpy as np
import random
import math

import intrusion_classify
import classify_common


# 创建优化XGBoost的DE类
class DEForXGBoost:
    # 构造方法
    def __init__(self, size, dim, max_iter, F, CR):
        # 种群大小
        self.size = size
        # 维度
        self.dim = dim
        # 迭代次数
        self.max_iter = max_iter
        # 缩放比例
        self.F = F
        # 交叉概率
        self.CR = CR
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
            # 对每个个体进行变异、交叉、选择操作
            for i in range(self.size):
                # Mutation
                # 记录target_vector
                target_vector = self.pos[i].copy()    # deep copy
                # 随机选取r1,r2,r3，需与i不同
                r_list = []
                # 循环选取
                while len(r_list) < 3:
                    # 随机选取一个数
                    r_temp = random.randint(0, self.size-1)
                    # 若该数不与i相同
                    if r_temp != i:
                        # 则将该数添加进被选数组
                        r_list.append(r_temp)
                # r1,r2,r3
                r1 = r_list[0]
                r2 = r_list[1]
                r3 = r_list[2]
                # 生成mutant_vector
                mutant_vector = self.pos[r1] + self.F * (self.pos[r2] - self.pos[r3])

                # Crossover
                # 创建trial_vector
                trial_vector = np.zeros(self.dim)
                # 随机生成 rnbr
                rnbr = random.randint(0, self.dim-1)
                # 开始交叉过程
                for j in range(self.dim):
                    # 生成决定是否交叉的随机数
                    randb = random.uniform(0, 1)
                    # 判断是否进行交叉操作
                    if randb <= self.CR or j == rnbr:
                        # 进行交叉操作
                        trial_vector[j] = mutant_vector[j]
                    else:
                        # 不进行交叉操作
                        trial_vector[j] = target_vector[j]

                # Selection
                # 记录target_vector的适应度值
                target_vector_fit = self.fit[i]
                # 判断trial_vector各参数是否越界
                trial_vector = classify_common.xgb_bound_check(trial_vector)
                # 计算trial_vector的适应度值
                trial_vector_fit = intrusion_classify.xgb_classify(trial_vector, classify_dataset, attack_types)
                # 比较双方的适应度值
                if trial_vector_fit > target_vector_fit:
                    # 若trial_vector适应度值优于target_vector，则替换之
                    self.pos[i] = trial_vector.copy()    # deep copy
                    # 并同时替换适应度值
                    self.fit[i] = trial_vector_fit

            # 更新全局最优下标、位置和适应度值
            max_index = np.argsort(-self.fit)[0]
            self.g_best_pos = self.pos[max_index].copy()  # deep copy
            self.g_best_fit = self.fit[max_index]
            # 输出本次迭代的全局最优位置和适应度值
            print('当前迭代次数：', iter_count + 1)
            print(self.g_best_pos)
            print(self.g_best_fit)
            # 记录本次迭代的最优位置和适应度值
            self.pos_record[iter_count + 1] = self.g_best_pos
            self.fit_record[iter_count + 1] = self.g_best_fit


# de_xgb = DEForXGBoost(size=5, dim=6, max_iter=3, F=1, CR=0.5)
# de_xgb.initial()
# de_xgb.optimal()

























