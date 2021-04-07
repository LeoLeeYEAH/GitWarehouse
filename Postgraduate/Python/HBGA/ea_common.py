import matlab_fe


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
























