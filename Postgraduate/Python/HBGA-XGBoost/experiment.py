import GA
import PSO
import HBGA
import DE
import WOA

import time

import classify_common
import intrusion_classify
import data_set_preprocess


# 获取处理过后的数据集
# NSL-KDD
classify_dataset, attack_types = data_set_preprocess.get_nsl_kdd()


# GA
# ga_xgb = GA.GAForXGBoost(size=100, dim=6, max_iter=50,
#                          select_type='rws', cross_type='spc', cross_rate=0.8, mutation_type='rm', mutation_rate=0.05,
#                          keep_elite=1)
# ga_xgb.initial(classify_dataset, attack_types)
# ga_xgb.optimal(classify_dataset, attack_types)
# classify_common.curve(ga_xgb.max_iter, ga_xgb.fit_record, 'ga_xgb')
# classify_common.to_csv(ga_xgb.fit_record, ga_xgb.pos_record, 'ga_xgb')


# DE
# de_xgb = DE.DEForXGBoost(size=100, dim=6, max_iter=50, F=1, CR=0.5)
# de_xgb.initial(classify_dataset, attack_types)
# de_xgb.optimal(classify_dataset, attack_types)
# classify_common.curve(de_xgb.max_iter, de_xgb.fit_record, 'de_xgb')
# classify_common.to_csv(de_xgb.fit_record, de_xgb.pos_record, 'de_xgb')


# PSO
# for i in range(3):
#     pso_xgb = PSO.PSOForXGBoost(w_max=0.9, w_min=0.4, c1=2, c2=2, max_iter=100, ps_size=100, dim=11)
#     pso_xgb.initial(classify_dataset, attack_types)
#     pso_xgb.optimal(classify_dataset, attack_types)
#     classify_common.curve(pso_xgb.max_iter, pso_xgb.fit_record, 'pso_xgb')
#     classify_common.to_csv(pso_xgb.fit_record, pso_xgb.pos_record, 'pso_xgb')


# HBGA
# hbga_xgb = HBGA.HBGAForXGBoost(size=90, dim=6, max_self=10, max_iter=50)
# hbga_xgb.initial(classify_dataset, attack_types)
# hbga_xgb.optimal(classify_dataset, attack_types)
# classify_common.curve(hbga_xgb.max_iter, hbga_xgb.fit_record, 'hbga_xgb')
# classify_common.to_csv(hbga_xgb.fit_record, hbga_xgb.pos_record, 'hbga_xgb')


# WOA
for i in range(10):
    woa_xgb = WOA.WOAForXGBoost(size=100, dim=11, max_iter=100, a=2, b=1)
    woa_xgb.initial(classify_dataset, attack_types)
    woa_xgb.optimal(classify_dataset, attack_types)
    classify_common.curve(woa_xgb.max_iter, woa_xgb.fit_record, 'woa_xgb')
    classify_common.to_csv(woa_xgb.fit_record, woa_xgb.pos_record, 'woa_xgb')


# 程序运行时间
print(time.perf_counter())













