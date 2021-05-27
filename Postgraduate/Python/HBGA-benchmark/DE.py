import numpy as np
import ea_common
import random
import matplotlib.pyplot as plt
import math


# 创建DE类
class DE:
    # 构造方法
    def __init__(self, size, dim, pos_max, pos_min, max_iter, func_no, F, CR):
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
                # 计算trial_vector的适应度值
                trial_vector_fit = ea_common.func_eval(trial_vector, self.func_no)
                # 比较双方的适应度值
                if trial_vector_fit < target_vector_fit:
                    # 若trial_vector适应度值优于target_vector，则替换之
                    self.pos[i] = trial_vector.copy()    # deep copy
                    # 并同时替换适应度值
                    self.fit[i] = trial_vector_fit

            # 更新全局最优下标、位置和适应度值
            min_index = np.argsort(self.fit)[0]
            self.g_best_pos = self.pos[min_index].copy()  # deep copy
            self.g_best_fit = self.fit[min_index]
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


# de = DE(size=100, dim=10, pos_max=100, pos_min=-100, max_iter=500, func_no=1, F=1, CR=0.5)
# de.initial()
# de.optimal()
# de.curve()























