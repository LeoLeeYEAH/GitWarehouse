import matlab.engine
import time


# 调用matlab.engine（因每次调用都需要打开Matlab会话非常耗时，因此最好在程序执行期间只调用一次）
startME = matlab.engine.start_matlab()


# 计算适应度值的方法
def func_eval(pos, func_no):
    # 进行数据格式转换
    pos = matlab.double(pos.tolist())
    func_no = str(func_no)
    # 计算适应度值
    result = startME.test(pos, func_no)

    # 减去相应的附加值
    if func_no == '1':
        result = result - 100
    elif func_no == '2':
        result = result - 200
    elif func_no == '3':
        result = result - 300
    elif func_no == '4':
        result = result - 400
    elif func_no == '5':
        result = result - 500
    elif func_no == '6':
        result = result - 600
    elif func_no == '7':
        result = result - 700
    elif func_no == '8':
        result = result - 800
    elif func_no == '9':
        result = result - 900
    elif func_no == '10':
        result = result - 1200
    elif func_no == '11':
        result = result - 1100
    elif func_no == '12':
        result = result - 1200
    elif func_no == '13':
        result = result - 1600
    elif func_no == '14':
        result = result - 1400
    elif func_no == '15':
        result = result - 1700

    # 返回适应度值
    return result


# 计算适应度值（示例）
# pos = [4.42978163577759, 38.9001364720917, 53.5380018967597, 4.21556282220163, -54.1771326694329,
#        -7.36327648112782, -56.0254271699207, -51.7274333740953, -49.9579113664428, -12.5193627717176]
# func_no = '15'
# for i in range(10000):
#     fit_value = func_eval(pos, func_no)
#     print(fit_value)
# print(time.perf_counter())





