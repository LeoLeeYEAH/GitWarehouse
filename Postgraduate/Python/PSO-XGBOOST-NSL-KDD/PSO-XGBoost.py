import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from xgboost import XGBClassifier


# 引入time模块
import time
# 程序运行开始时间
start_time = time.perf_counter()


# 数据预处理
def data_preprocessing(train_set_file, test_set_file):
    # 读取数据
    train_df = pd.read_csv(train_set_file)
    test_df = pd.read_csv(test_set_file)

    # 为数据集添加列索引
    columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
                'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
                'is_host_login',
                'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                'srv_rerror_rate',
                'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate', 'attack', 'level'])

    train_df.columns = columns
    test_df.columns = columns

    # 去掉数据集最后一列（攻击程度 level）
    train_df.drop('level', axis=1, inplace=True)
    test_df.drop('level', axis=1, inplace=True)

    # 判断是否被攻击
    def attack_flag(attack):
        if attack == 'normal':
            return 0
        else:
            return 1

    # 判断是哪一类攻击
    def attack_type(attack):
        # 攻击类型
        dos_attacks = ['apache2', 'back', 'land', 'mailbomb', 'neptune', 'pod', 'processtable', 'smurf', 'teardrop',
                       'udpstorm']
        probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
        u2r_attacks = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm', 'httptunnel']
        r2l_attacks = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                       'snmpgetattack', 'spy', 'snmpguess', 'warezclient', 'warezmaster', 'xlock', 'xsnoop', 'worm']

        if attack in dos_attacks:
            return 1
        elif attack in probe_attacks:
            return 2
        elif attack in u2r_attacks:
            return 3
        elif attack in r2l_attacks:
            return 4
        else:
            return 0

    # 为数据添加是否被攻击以及攻击类型
    train_df['attack_flag'] = train_df['attack'].apply(attack_flag)
    train_df['attack_type'] = train_df['attack'].apply(attack_type)
    test_df['attack_flag'] = test_df['attack'].apply(attack_flag)
    test_df['attack_type'] = test_df['attack'].apply(attack_type)

    # 对数据集中的连续型特征进行预处理
    # 所有连续型特征
    continuous_features = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
                           'num_compromised', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files',
                           'num_outbound_cmds', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
                           'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
                           'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
                           'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
                           'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']
    # 对数据进行标准化和归一化
    for feature in continuous_features:
        # 标准化
        train_df[feature] = StandardScaler().fit_transform(train_df[feature].values.reshape(-1, 1))
        test_df[feature] = StandardScaler().fit_transform(test_df[feature].values.reshape(-1, 1))
        # 归一化
        train_df[feature] = MinMaxScaler().fit_transform(train_df[feature].values.reshape(-1, 1))
        test_df[feature] = MinMaxScaler().fit_transform(test_df[feature].values.reshape(-1, 1))

    # 对离散型特征进行处理
    # land logged_in root_shell su_attempted is_host_login is_guest_login 也是离散型数据，但是取值都是0或1，因此不必独热编码
    discrete_features = ['protocol_type', 'service', 'flag']
    train_discrete_features_encode = pd.get_dummies(train_df[discrete_features])
    test_discrete_features_encode = pd.get_dummies(test_df[discrete_features])

    # 对训练集和测试集中的离散特征进行热编码后，测试集中有部分离散特征缺失，因此需要进行补充
    # 根据测试集的数据量生成index
    test_index = np.arange(len(test_df))
    # 找出测试集中缺失的离散特征
    diff_columns = list(
        set(train_discrete_features_encode.columns.values) - set(test_discrete_features_encode.columns.values))
    # 构建缺失的离散特征的DataFrame
    diff_df = pd.DataFrame(0, index=test_index, columns=diff_columns)
    # 将缺失的特征补到测试集的独热编码上
    test_discrete_features_encode = pd.concat([test_discrete_features_encode, diff_df], axis=1)
    # 对离散特征数据重新排序
    train_discrete_features_encode = train_discrete_features_encode.sort_index(axis=1)
    test_discrete_features_encode = test_discrete_features_encode.sort_index(axis=1)

    # 将整理好的离散特征加入到数据集中
    train_df = pd.concat([train_df, train_discrete_features_encode], axis=1)
    test_df = pd.concat([test_df, test_discrete_features_encode], axis=1)

    # 将原本的离散特征从数据集中删除
    train_df.drop(['protocol_type', 'service', 'flag'], axis=1, inplace=True)
    test_df.drop(['protocol_type', 'service', 'flag'], axis=1, inplace=True)

    # 记录数据集中与攻击类型相关的数据
    train_df_attack = train_df['attack']
    test_df_attack = test_df['attack']
    train_df_attack_flag = train_df['attack_flag']
    test_df_attack_flag = test_df['attack_flag']
    train_df_attack_type = train_df['attack_type']
    test_df_attack_type = test_df['attack_type']

    # 将原本的与攻击类型相关的数据从数据集中删除
    train_df.drop(['attack', 'attack_flag', 'attack_type'], axis=1, inplace=True)
    test_df.drop(['attack', 'attack_flag', 'attack_type'], axis=1, inplace=True)

    return train_df, train_df_attack_type, test_df, test_df_attack_type


# XGBoost
def XGBoost(x_train, y_train, x_test, y_test,
            learning_rate=0.3, max_depth=6, min_child_weight=1.0, gamma=0.0, subsample=1.0, colsample_bytree=1.0):

    # 设置XGBoost各项参数
    xgb = XGBClassifier(learning_rate=learning_rate,
                        max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        gamma=gamma,
                        subsample=subsample,
                        colsample_bytree=colsample_bytree,
                        # GPU加速
                        tree_method='gpu_hist')

    # 开始训练
    xgb.fit(x_train, y_train)
    # 继续利用训练集进行预测
    train_predict = xgb.predict(x_train)
    # 利用测试集进行预测
    test_predict = xgb.predict(x_test)

    # 攻击类型标签
    attack_types = ['Normal', 'Dos', 'Probe', 'U2R', 'R2L']

    # 多分类评价标准
    # train_classification_report = classification_report(y_train, train_predict, target_names=attack_types)
    # test_classification_report = classification_report(y_test, test_predict, target_names=attack_types)
    # print('Train Set Classification Report:\n')
    # print(train_classification_report)
    # print('Test Set Classification Report:\n')
    # print(test_classification_report)

    # 计算各攻击类型的AP值
    # 将测试集预测目标转化为Numpy Array格式
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # 获取样本数量
    train_samples_num = len(y_train)
    test_samples_num = len(y_test)
    # 利用自定义的方法计算AP值
    # 训练集
    normal_ap_score_train = calculate_ap_score('Normal', y_train.copy(), train_predict.copy(), train_samples_num)
    dos_ap_score_train = calculate_ap_score('Dos', y_train.copy(), train_predict.copy(), train_samples_num)
    probe_ap_score_train = calculate_ap_score('Probe', y_train.copy(), train_predict.copy(), train_samples_num)
    u2r_ap_score_train = calculate_ap_score('U2R', y_train.copy(), train_predict.copy(), train_samples_num)
    r2l_ap_score_train = calculate_ap_score('R2L', y_train.copy(), train_predict.copy(), train_samples_num)
    # 测试集
    normal_ap_score_test = calculate_ap_score('Normal', y_test.copy(), test_predict.copy(), test_samples_num)
    dos_ap_score_test = calculate_ap_score('Dos', y_test.copy(), test_predict.copy(), test_samples_num)
    probe_ap_score_test = calculate_ap_score('Probe', y_test.copy(), test_predict.copy(), test_samples_num)
    u2r_ap_score_test = calculate_ap_score('U2R', y_test.copy(), test_predict.copy(), test_samples_num)
    r2l_ap_score_test = calculate_ap_score('R2L', y_test.copy(), test_predict.copy(), test_samples_num)

    # 返回所设置的适应度目标
    return u2r_ap_score_train


# 计算各攻击类型的AP值
def calculate_ap_score(attack_type, y_true, y_score, samples_num):
    # 判断要计算哪个攻击类型的AP值
    if attack_type == 'Normal':
        # Normal
        for i in range(samples_num):
            if y_true[i] == 0:
                y_true[i] = 1
            else:
                y_true[i] = 0
            if y_score[i] == 0:
                y_score[i] = 1
            else:
                y_score[i] = 0
        normal_ap_score = average_precision_score(y_true, y_score)
        return normal_ap_score
    elif attack_type == 'Dos':
        # Dos
        for i in range(samples_num):
            if y_true[i] != 1:
                y_true[i] = 0
            if y_score[i] != 1:
                y_score[i] = 0
        dos_ap_score = average_precision_score(y_true, y_score)
        return dos_ap_score
    elif attack_type == 'Probe':
        # Probe
        for i in range(samples_num):
            if y_true[i] == 2:
                y_true[i] = 1
            else:
                y_true[i] = 0
            if y_score[i] == 2:
                y_score[i] = 1
            else:
                y_score[i] = 0
        probe_ap_score = average_precision_score(y_true, y_score)
        return probe_ap_score
    elif attack_type == 'U2R':
        # U2R
        for i in range(samples_num):
            if y_true[i] == 3:
                y_true[i] = 1
            else:
                y_true[i] = 0
            if y_score[i] == 3:
                y_score[i] = 1
            else:
                y_score[i] = 0
        u2r_ap_score = average_precision_score(y_true, y_score)
        return u2r_ap_score
    elif attack_type == 'R2L':
        # R2L
        for i in range(samples_num):
            if y_true[i] == 4:
                y_true[i] = 1
            else:
                y_true[i] = 0
            if y_score[i] == 4:
                y_score[i] = 1
            else:
                y_score[i] = 0
        r2l_ap_score = average_precision_score(y_true, y_score)
        return r2l_ap_score
    else:
        return 0

# 创建PSO类
class PSO:
    def __init__(self, w_max, w_min, c1, c2, max_iter, ps_size, dim, x_train, y_train, x_test, y_test):
        # 惯性权重
        self.w_max = w_max
        self.w_min = w_min
        # 加速度因子
        self.c1 = c1
        self.c2 = c2
        # 迭代次数
        self.max_iter = max_iter
        # 种群规模
        self.ps_size = ps_size
        # 维度
        self.dim = dim
        # 粒子群位置和速度
        self.p_pos = np.zeros((ps_size, dim))
        self.p_vel = np.zeros((ps_size, dim))
        # 粒子最佳位置
        self.p_best_pos = np.zeros((ps_size, dim))
        # 全局最佳位置
        self.g_best_pos = np.zeros((1, dim))
        # 粒子最佳适应度
        self.p_best_fit = np.zeros(ps_size)
        # 全局最佳适应度
        self.g_best_fit = 0
        # 训练集和测试集
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    # 种群初始化
    def initial(self):
        for i in range(self.ps_size):
            # 初始化XGBoost模型的六个重要参数的位置
            # eta
            self.p_pos[i][0] = random.uniform(0, 1)
            # max_depth
            self.p_pos[i][1] = random.randint(3, 10)
            # min_child_weight
            self.p_pos[i][2] = random.uniform(0, 10)
            # gamma
            self.p_pos[i][3] = random.uniform(0, 1)
            # subsample
            self.p_pos[i][4] = random.uniform(0, 1)
            # colsample_bytree
            self.p_pos[i][5] = random.uniform(0, 1)
            # 初始化XGBoost模型的六个重要参数的速度
            for j in range(self.dim):
                self.p_vel[i][j] = random.uniform(-0.5, 0.5)
            # 计算初始适应度
            temp = self.fit_function(self.x_train, self.y_train, self.x_test, self.y_test, self.p_pos[i])
            # 记录粒子初始位置和适应度
            self.p_best_pos[i] = self.p_pos[i]
            self.p_best_fit[i] = temp
            # 记录全局初始最优位置和适应度
            if temp > self.g_best_fit:
                self.g_best_fit = temp
                self.g_best_pos = self.p_pos[i]
        print(self.g_best_pos)
        print(self.g_best_fit)

    # 迭代寻优
    def optimal(self):
        # 迭代寻优
        for iter_count in range(self.max_iter):
            # 输出迭代次数
            global iter_counter
            iter_counter += 1
            print('迭代次数：%s' % iter_counter)
            # 计算当前惯性权重w的值
            w = (self.w_max + (self.max_iter - iter_count) * (self.w_max - self.w_min)) / self.max_iter
            # 更新粒子位置和速度
            for i in range(self.ps_size):
                # 粒子速度更新
                self.p_vel[i] = w * self.p_vel[i] + \
                                self.c1 * random.uniform(0, 1) * (self.p_best_pos[i] - self.p_pos[i]) + \
                                self.c2 * random.uniform(0, 1) * (self.g_best_pos - self.p_pos[i])
                # 判断粒子速度是否超过边界
                for j in range(self.dim):
                    if self.p_vel[i][j] > 0.5:
                        self.p_vel[i][j] = 0.5
                    if self.p_vel[i][j] < -0.5:
                        self.p_vel[i][j] = -0.5
                # 粒子位置更新
                self.p_pos[i] = self.p_pos[i] + self.p_vel[i]
                # max_depth必须为整数
                self.p_pos[i][1] = round(self.p_pos[i][1])
                # 判断粒子位置是否超过边界
                # eta
                if self.p_pos[i][0] < 0:
                    self.p_pos[i][0] = 0
                if self.p_pos[i][0] > 1:
                    self.p_pos[i][0] = 1
                # max_depth
                if self.p_pos[i][1] < 1:
                    self.p_pos[i][1] = 1
                # min_child_weight
                if self.p_pos[i][2] < 0:
                    self.p_pos[i][2] = 0
                # gamma
                if self.p_pos[i][3] < 0:
                    self.p_pos[i][3] = 0
                # subsample
                if self.p_pos[i][4] < 0:
                    self.p_pos[i][4] = 0.01
                if self.p_pos[i][4] > 1:
                    self.p_pos[i][4] = 1
                # colsample_bytree
                if self.p_pos[i][5] < 0:
                    self.p_pos[i][5] = 0.01
                if self.p_pos[i][5] > 1:
                    self.p_pos[i][5] = 1
                # 计算当前粒子的适应度
                temp = self.fit_function(self.x_train, self.y_train, self.x_test, self.y_test, self.p_pos[i])
                # 根据粒子适应度判断是否更新粒子以及全局的最优位置和适应度
                if temp > self.p_best_fit[i]:
                    # 更新粒子最优位置和适应度
                    self.p_best_fit[i] = temp
                    self.p_best_pos[i] = self.p_pos[i]
                    if self.p_best_fit[i] > self.g_best_fit:
                        # 更新全局最优位置和适应度
                        self.g_best_fit = self.p_best_fit[i]
                        self.g_best_pos = self.p_best_pos[i]
                        print(self.g_best_pos)
                        print(self.g_best_fit)

    # 适应度函数
    @staticmethod
    def fit_function(x_train, y_train, x_test, y_test, parameter):
        # 利用XGBoost进行训练，返回所需的评价参数
        fitness = XGBoost(x_train, y_train, x_test, y_test,
                          learning_rate=parameter[0],
                          max_depth=round(parameter[1]),
                          min_child_weight=parameter[2],
                          gamma=parameter[3],
                          subsample=parameter[4],
                          colsample_bytree=parameter[5])
        return fitness


if __name__ == '__main__':
    # 对数据集进行预处理
    train_set_file_origin = 'input/KDDTrain+.txt'
    test_set_file_origin = 'input/KDDTest+.txt'

    # 划分训练集和测试集
    train_set, train_set_attack_type, test_set, test_set_attack_type = data_preprocessing(train_set_file_origin,
                                                                                          test_set_file_origin)
    # 利用XGBoost进行训练
    # XGBoost(train_set, train_set_attack_type, test_set, test_set_attack_type,
    #         learning_rate=1.0, max_depth=8, min_child_weight=1.1867, gamma=0.0045, subsample=0.6145, colsample_bytree=0.7250)

    # 设置PSO参数
    pso = PSO(w_max=0.9, w_min=0.4, c1=2, c2=2, max_iter=80, ps_size=100, dim=6,
              x_train=train_set, y_train=train_set_attack_type, x_test=test_set, y_test=test_set_attack_type)

    # 记录迭代次数
    global iter_counter
    iter_counter = 0

    # 初始化粒子群
    pso.initial()

    # 开始优化
    pso.optimal()





























# 程序运行结束时间
end_time = time.perf_counter()

# 程序运行时间
print('Running Time: %s Seconds' % (end_time - start_time))

