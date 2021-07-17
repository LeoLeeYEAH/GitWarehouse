import numpy as np
import math
import matlab_fe
import matplotlib.pyplot as plt


# Function Evaluation 适应度值评估
def func_eval(pos, func_no):
    # 若 func_no > 0 则是利用CEC2015 benchmark函数进行计算
    if func_no > 0:
        result = matlab_fe.func_eval(pos, func_no)
    # 若 func_no = 0 则说明是其他适应度值计算方式
    else:
        result = 0
    return result


# Boundary Check 越界检查
def bound_check(item, upper_bound, lower_bound):
    if item > upper_bound:
        item = upper_bound
    if item < lower_bound:
        item = lower_bound
    return item


# 绘制收敛曲线
def curve(max_iter, fit_record):
    # 利用math.log转换适应度值
    fit_record_log = np.zeros(max_iter + 1)
    for i in range(max_iter + 1):
        # 判断是否提前收敛
        if fit_record[i] > 0:
            # 若未提前收敛
            fit_record_log[i] = math.log(fit_record[i])
        else:
            # 若已提前收敛
            fit_record_log[fit_record_log == 0] = math.log(1e-8)
    # 绘制收敛曲线
    plt.title("Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.plot(np.arange(max_iter + 1), [v for v in fit_record_log])
    plt.show()






















