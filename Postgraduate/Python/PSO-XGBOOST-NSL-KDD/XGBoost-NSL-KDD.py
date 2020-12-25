import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# 引入time模块
import time
# 程序运行开始时间
start_time = time.perf_counter()


# 读取数据
train_df = pd.read_csv('input/KDDTrain+.txt')
test_df = pd.read_csv('input/KDDTest+.txt')

# 为数据集添加列索引
columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
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
# 攻击类型标签
attack_types = ['Normal', 'Dos', 'Probe', 'U2R', 'R2L']


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
diff_columns = list(set(train_discrete_features_encode.columns.values) - set(test_discrete_features_encode.columns.values))
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

# 开始训练
# 划分训练集和测试集
train_set_x = train_df
train_set_y = train_df_attack_type
test_set_x = test_df
test_set_y = test_df_attack_type


# XGBoost
def XGBoost(x_train, y_train, x_test, y_test):
    # 设置XGBoost各项参数
    xgb = XGBClassifier(learning_rate=0.3,
                        max_depth=6,
                        min_child_weight=1,
                        gamma=0,
                        subsample=1,
                        colsample_bytree=1)
    # 开始训练
    xgb.fit(x_train, y_train)
    train_predict = xgb.predict(x_train)
    test_predict = xgb.predict(x_test)

    # 评价标准
    # train_classification_report = classification_report(y_train, train_predict, target_names=attack_types)
    # test_classification_report = classification_report(y_test, test_predict, target_names=attack_types)
    # print('Train Set Classification Report:\n')
    # print(train_classification_report)
    # print('Test Set Classification Report:\n')
    # print(test_classification_report)

    # 计算各攻击类型的AP值
    # 将测试集预测目标转化为Numpy Array格式
    y_train = np.array(y_train)
    # 获取样本数量
    samples_num = len(y_train)
    # Normal
    y_train_normal = y_train.copy()
    train_predict_normal = train_predict.copy()
    normal_ap_score(y_train_normal, train_predict_normal, samples_num)
    # Dos
    y_train_dos = y_train.copy()
    train_predict_dos = train_predict.copy()
    dos_ap_score(y_train_dos, train_predict_dos, samples_num)
    # Probe
    y_train_probe = y_train.copy()
    train_predict_probe = train_predict.copy()
    probe_ap_score(y_train_probe, train_predict_probe, samples_num)
    # U2R
    y_train_u2r = y_train.copy()
    train_predict_u2r = train_predict.copy()
    u2r_ap_score(y_train_u2r, train_predict_u2r, samples_num)
    # R2L
    y_train_r2l = y_train.copy()
    train_predict_r2l = train_predict.copy()
    r2l_ap_score(y_train_r2l, train_predict_r2l, samples_num)


# 计算Normal类型的AP值
def normal_ap_score(y_true, y_score, samples_num):
    for i in range(samples_num):
        if y_true[i] == 0:
            y_true[i] = 1
        else:
            y_true[i] = 0
        if y_score[i] == 0:
            y_score[i] = 1
        else:
            y_score[i] = 0
    print('Normal AP Score:%s' % average_precision_score(y_true, y_score))


# 计算Dos类型的AP值
def dos_ap_score(y_true, y_score, samples_num):
    for i in range(samples_num):
        if y_true[i] != 1:
            y_true[i] = 0
        if y_score[i] != 1:
            y_score[i] = 0
    print('Dos AP Score:%s' % average_precision_score(y_true, y_score))


# 计算Probe类型的AP值
def probe_ap_score(y_true, y_score, samples_num):
    for i in range(samples_num):
        if y_true[i] == 2:
            y_true[i] = 1
        else:
            y_true[i] = 0
        if y_score[i] == 2:
            y_score[i] = 1
        else:
            y_score[i] = 0
    print('Probe AP Score:%s' % average_precision_score(y_true, y_score))


# 计算U2R类型的AP值
def u2r_ap_score(y_true, y_score, samples_num):
    for i in range(samples_num):
        if y_true[i] == 3:
            y_true[i] = 1
        else:
            y_true[i] = 0
        if y_score[i] == 3:
            y_score[i] = 1
        else:
            y_score[i] = 0
    print('U2R AP Score:%s' % average_precision_score(y_true, y_score))


# 计算R2L类型的AP值
def r2l_ap_score(y_true, y_score, samples_num):
    for i in range(samples_num):
        if y_true[i] == 4:
            y_true[i] = 1
        else:
            y_true[i] = 0
        if y_score[i] == 4:
            y_score[i] = 1
        else:
            y_score[i] = 0
    print('R2L AP Score:%s' % average_precision_score(y_true, y_score))


XGBoost(train_set_x, train_set_y, test_set_x, test_set_y)


# pd.set_option('display.max_columns', None)





























# 程序运行结束时间
end_time = time.perf_counter()

# 程序运行时间
print('Running Time: %s Seconds' % (end_time - start_time))

