import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import random
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import  XGBClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# 程序运行开始时间
start_time = time.perf_counter()

# read data
train_df = pd.read_csv('input/KDDTrain+.txt')
test_df = pd.read_csv('input/KDDTest+.txt')

# add the column labels
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

train_df['attack_flag'] = train_df.attack.map(lambda a: 0 if a == 'normal' else 1)
test_df['attack_flag'] = test_df.attack.map(lambda a: 0 if a == 'normal' else 1)

# we will use these for plotting below
attack_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']


def classify_attack(attack):

    # lists to hold our attack classifications
    dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop',
                   'udpstorm',
                   'worm']
    probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
    privilege_attacks = ['buffer_overflow', 'loadmdoule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
    access_attacks = ['ftp_write', 'guess_passwd', 'http_tunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail',
                      'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xclock', 'xsnoop']

    if attack in dos_attacks:
        # dos_attacks map to 1
        attack_type = 1
    elif attack in probe_attacks:
        # probe_attacks mapt to 2
        attack_type = 2
    elif attack in privilege_attacks:
        # privilege escalation attacks map to 3
        attack_type = 3
    elif attack in access_attacks:
        # remote access attacks map to 4
        attack_type = 4
    else:
        # normal maps to 0
        attack_type = 0

    return attack_type


train_df['attack_map'] = train_df.attack.apply(classify_attack)
test_df['attack_map'] = test_df.attack.apply(classify_attack)


# helper function for drawing mulitple charts.
def bake_pies(data_list, labels):
    list_length = len(data_list)

    # setup for mapping colors
    color_list = sns.color_palette()
    color_cycle = itertools.cycle(color_list)
    cdict = {}

    # build the subplots
    fig, axs = plt.subplots(1, list_length, figsize=(18, 10), tight_layout=False)
    plt.subplots_adjust(wspace=1 / list_length)

    # loop through the data sets and build the charts
    for count, data_set in enumerate(data_list):

        # update our color mapt with new values
        for num, value in enumerate(np.unique(data_set.index)):
            if value not in cdict:
                cdict[value] = next(color_cycle)

        # build the wedges
        wedges, texts = axs[count].pie(data_set, colors=[cdict[v] for v in data_set.index])

        # build the legend
        axs[count].legend(wedges, data_set.index, title="Flags", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        # set the title
        axs[count].set_title(labels[count])

    return axs


# get the series for each protocol
icmp_attacks = pd.crosstab(train_df.attack, train_df.protocol_type).icmp
tcp_attacks = pd.crosstab(train_df.attack, train_df.protocol_type).tcp
udp_attacks = pd.crosstab(train_df.attack, train_df.protocol_type).udp

# create the charts
# protocol_axs = bake_pies([icmp_attacks, tcp_attacks, udp_attacks], ['icmp', 'tcp', 'udp'])

# get a series with the count of each flag for attack and normal traffic
normal_flags = train_df.loc[train_df.attack_flag == 0].flag.value_counts()
attack_flags = train_df.loc[train_df.attack_flag == 1].flag.value_counts()

# create the charts
# flag_axs = bake_pies([normal_flags, attack_flags], ['normal', 'attack'])

# get a series with the count of each service for attack and normal traffic
normal_services = train_df.loc[train_df.attack_flag == 0].service.value_counts()
attack_services = train_df.loc[train_df.attack_flag == 1].service.value_counts()

# create the charts
# service_axs = bake_pies([normal_services, attack_services], ['normal', 'attack'])

# get the intial set of encoded features and encode them
features_to_encode = ['protocol_type', 'service', 'flag']
train_encoded_base = pd.get_dummies(train_df[features_to_encode])
test_encoded_base = pd.get_dummies(test_df[features_to_encode])

# not all of the features are in the test set, so we need to account for diffs
test_index = np.arange(len(test_df.index))
column_diffs = list(set(train_encoded_base.columns.values) - set(test_encoded_base.columns.values))
diff_df = pd.DataFrame(0, index=test_index, columns=column_diffs)

# we'll also need to reorder the columns to match, so let's get those
column_order = train_encoded_base.columns.to_list()

# append the new columns
test_encoded_temp = test_encoded_base.join(diff_df)

# reorder the columns
test_final = test_encoded_temp[column_order].fillna(0)

# get numeric features, we won't worry about encoding these at this point
numeric_features = ['duration', 'src_bytes', 'dst_bytes']

# model to train/test
train_set = train_encoded_base.join(train_df[numeric_features])
test_set = test_final.join(test_df[numeric_features])

# create our target classifications
y_train_binary_target = train_df['attack_flag']
y_train_multi_target = train_df['attack_map']

y_test_binary_target = test_df['attack_flag']
y_test_multi_target = test_df['attack_map']

# build the training sets
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(train_set, y_train_binary_target, test_size=0.2)
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(train_set, y_train_multi_target, test_size=0.2)

# model for the binary classification
# RandomForestClassifier
rf_binary = RandomForestClassifier()
rf_binary.fit(X_train_binary, y_train_binary)
rf_binary_predict = rf_binary.predict(X_test_binary)
# LogisticRegression
lr_binary = LogisticRegression(max_iter=250)
lr_binary.fit(X_train_binary, y_train_binary)
lr_binary_predict = lr_binary.predict(X_test_binary)
# KNeighbors
kn_binary = KNeighborsClassifier()
kn_binary.fit(X_train_binary, y_train_binary)
kn_binary_predict = kn_binary.predict(X_test_binary)


# calculate and display our base accuracy
rf_binary_score = accuracy_score(rf_binary_predict, y_test_binary)
lr_binary_score = accuracy_score(lr_binary_predict, y_test_binary)
kn_binary_score = accuracy_score(kn_binary_predict, y_test_binary)

print('Random Forest Binary Accuracy Score: %s' % rf_binary_score)
print('Logistic Regression Binary Accuracy Score: %s' % lr_binary_score)
print('K Neighbors Binary Accuracy Score: %s' % kn_binary_score)

# model for the multi classification
# RandomForestClassifier
rf_multi = RandomForestClassifier()
rf_multi.fit(X_train_multi, y_train_multi)
rf_multi_predict = rf_multi.predict(X_test_multi)
# KNeighborsClassifier
kn_multi = KNeighborsClassifier()
kn_multi.fit(X_train_multi, y_train_multi)
kn_multi_predict = kn_multi.predict(X_test_multi)

# calculate and display our base accuracy
rf_multi_score = accuracy_score(rf_multi_predict, y_test_multi)
kn_multi_score = accuracy_score(kn_multi_predict, y_test_multi)

print('Random Forest Multi Accuracy Score: %s' % rf_multi_score)
print('K Neighbors Multi Accuracy Score: %s' % kn_multi_score)

# XGBoost
# xgb_binary = XGBClassifier()
# xgb_binary.fit(X_train_binary, y_train_binary)
# xgb_binary_predict = xgb_binary.predict(X_test_binary)
#
# xgb_multi = XGBClassifier()
# xgb_multi.fit(X_train_multi, y_train_multi)
# xgb_multi_predict = xgb_multi.predict(X_test_multi)
#
#
# xgb_binary_score = accuracy_score(xgb_binary_predict, y_test_binary)
# xgb_multi_score = accuracy_score(xgb_multi_predict, y_test_multi)
#
# print('XGBoost Binary Accuracy Score: %s' % xgb_binary_score)
# print('XGBoost Multi Accuracy Score: %s' % xgb_multi_score)





















# 程序运行结束时间
end_time = time.perf_counter()

# 程序运行时间
print('Running Time: %s Seconds' % (end_time - start_time))

