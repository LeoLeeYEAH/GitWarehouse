import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

raw_train = pd.read_csv('./input/chinese_news_cutted_train_utf8.csv', sep='\t', encoding='utf8')
raw_test = pd.read_csv('./input/chinese_news_cutted_test_utf8.csv', sep='\t', encoding='utf8')

raw_train_binary = raw_train[((raw_train['分类'] == '科技') | (raw_train['分类'] == '文化'))]
raw_test_binary = raw_test[((raw_test['分类'] == '科技') | (raw_test['分类'] == '文化'))]

stop_words = []
file = open('./input/stopwords.txt')
for line in file:
    stop_words.append(line.strip())
file.close()

vectorizer = CountVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(raw_train_binary['分词文章'])
X_test = vectorizer.transform(raw_test_binary['分词文章'])

# 构建模型
# 感知机
percep_clf = SGDClassifier(loss='perceptron', penalty=None, learning_rate='constant', eta0=1.0,
                           max_iter=1000, random_state=111)
# 逻辑回归
lr_clf = SGDClassifier(loss='log', penalty=None, learning_rate='constant', eta0=1.0,
                       max_iter=1000, random_state=111)
# 线性支持向量机
lsvm_clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, learning_rate='constant', eta0=1.0,
                         max_iter=1000, random_state=111)

# 训练模型
percep_clf.fit(X_train, raw_train_binary['分类'])
lr_clf.fit(X_train, raw_train_binary['分类'])
lsvm_clf.fit(X_train, raw_train_binary['分类'])

# 输出准确率
percepScore = round(percep_clf.score(X_test, raw_test_binary['分类']), 4)
print(percepScore)
lrScore = round(lr_clf.score(X_test, raw_test_binary['分类']), 4)
print(lrScore)
lsvmScore = round(lsvm_clf.score(X_test, raw_test_binary['分类']), 4)
print(lsvmScore)

# 模型效果评估
fig, ax = plt.subplots(figsize=(5, 5))

y_lr_pred = lr_clf.predict(X_test)
y_test_ture = raw_test_binary['分类']
confusion_matrix = confusion_matrix(y_lr_pred, y_test_ture)

ax = sns.heatmap(confusion_matrix, linewidths=0.5, cmap='Greens', annot=True, fmt='d',
                 xticklabels=lr_clf.classes_, yticklabels=lr_clf.classes_)
ax.set_ylabel('真实')
ax.set_xlabel('预测')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

plt.show()

















