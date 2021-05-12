import numpy as np
import pandas as pd
import math
import time

from GA import GA
from HBGA import HBGA
from PSO import PSO


# 待测试算法数量
optimizers = ['PSO', 'HBGA']
# benchmark函数数量
benchmark = 5
# 每个算法运行次数
times = 50
# 创建numpy数组储存运行结果
# results = np.zeros((benchmark * times, len(optimizers)))
# 为了方便审阅数据建立专用索引
data_index = [0] * benchmark * times
for i in range(benchmark * times):
    data_index[i] = 'F' + str(math.floor(i/times)+1)
# 创建DataFrame储存运行结果
data = pd.DataFrame(index=data_index, columns=optimizers)
# 将数据转存到CSV文件中
data.to_csv('results.csv')

# 开始测试
for opt in optimizers:
    print('==============================')
    print('开始测试', opt)
    print('==============================')

    # 当前算法的索引
    # opt_index = optimizers.index(opt)

    if opt == 'PSO':
        # PSO
        for func_no in range(benchmark):
            # 输出当前的benchmark函数
            print('----当前benchmark函数为：', func_no+1)
            # 设置各项参数
            pso = PSO(w_max=0.9, w_min=0.4, c1=2, c2=2, pos_max=100, pos_min=-100, vel_max=1, vel_min=-1,
                      max_iter=1000, ps_size=100, dim=10, func_no=func_no+1)
            # 多次运行
            for time in range(times):
                # 初始化
                pso.initial()
                # 开始迭代
                pso.optimal()
                # 收敛结果
                print('--------第', time+1, '次收敛结果为：', pso.result())
                # 运行结果的索引
                result_index = func_no * times + time
                # 储存运行结果
                # results[result_index, opt_index] = pso.result()
                data['PSO'].iloc[result_index] = pso.result()
                # 转存到CSV文件中
                data.to_csv('results.csv')

    if opt == 'HBGA':
        # HBGA
        for func_no in range(benchmark):
            # 输出当前的benchmark函数
            print('----当前benchmark函数为：', func_no+1)
            # 设置各项参数
            hbga = HBGA(size=60, dim=10, max_self=60, r_max=100, r_min=-100, max_iter=2500, func_no=func_no+1)
            # 多次运行
            for time in range(times):
                # 初始化
                hbga.initial()
                # 开始迭代
                hbga.optimal()
                # 收敛结果
                print('--------第', time+1, '次收敛结果为：', hbga.result())
                # 运行结果的索引
                result_index = func_no * times + time
                # 储存运行结果
                # results[result_index, opt_index] = hbga.result()
                data['HBGA'].iloc[result_index] = hbga.result()
                # 转存到CSV文件中
                data.to_csv('results.csv')

    if opt == 'GA':
        # GA
        for func_no in range(benchmark):
            # 输出当前的benchmark函数
            print('----当前benchmark函数为：', func_no+1)
            # 设置各项参数
            ga = GA(size=100, dim=10, pos_max=100, pos_min=-100, max_iter=1000, func_no=func_no+1,
                    select_type='rws', cross_type='spc', cross_rate=0.8, mutation_type='rm', mutation_rate=0.05, keep_elite=10)
            # 多次运行
            for time in range(times):
                # 初始化
                ga.initial()
                # 开始迭代
                ga.optimal()
                # 收敛结果
                print('--------第', time+1, '次收敛结果为：', ga.result())
                # 运行结果的索引
                result_index = func_no * times + time
                # 储存运行结果
                # results[result_index, opt_index] = ga.result()
                data['GA'].iloc[result_index] = ga.result()
                # 转存到CSV文件中
                data.to_csv('results.csv')

    print('==============================')
    print(opt, '测试结束')
    print('==============================')




















