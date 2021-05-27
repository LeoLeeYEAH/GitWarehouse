% 借用Matlab调用C++底层的benchmark函数
function val = test(pos, func_no)
    % 将Python传递过来的数据进行转换
    func_no = str2num(func_no);
    % 调用Matlab写好的方法计算适应度值并返回
    val = cec15_func(pos', func_no);