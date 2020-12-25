import pandas as pd
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns


def news_cut(text):
    return ' '.join(list(jieba.cut(text)))


raw_train = pd.read_csv('./input/train_sample_utf8.csv', encoding='utf8')
raw_test = pd.read_csv('./input/test_sample_utf8.csv', encoding='utf8')

# print(raw_train.head(5))
# print(raw_test.head(5))
# print(raw_train.shape)
# print(raw_test.shape)

# plt.figure(figsize=(15, 8))
# plt.subplot(1, 2, 1)
# raw_train['分类'].value_counts().sort_index().plot(kind='barh', title='训练集')
# plt.subplot(1, 2, 2)
# raw_test['分类'].value_counts().sort_index().plot(kind='barh', title='测试集')
# plt.show()

# testContent = '六月初的一天，来自深圳的中国旅游团游客纷纷拿起相机拍摄新奇刺激的好莱坞环球影城主题公园场景。'
# print(news_cut(testContent))

raw_train['分词文章'] = raw_train['文章'].map(news_cut)
raw_test['分词文章'] = raw_test['文章'].map(news_cut)

# print(raw_train.head(5))
# print(raw_test.head(5))

stop_words = []
file = open('./input/stopwords.txt')
# file_content = file.read().replace('\n', ' ').split()
for line in file:
    stop_words.append(line.strip())
file.close()

vectorizer = CountVectorizer(stop_words=stop_words)
X_train = vectorizer.fit_transform(raw_train['分词文章'])
X_test = vectorizer.transform(raw_test['分词文章'])

knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
knn.fit(X_train, raw_train['分类'])
Y_test = knn.predict(X_test)

fig, ax = plt.subplots(figsize=(9, 7))
ax = sns.heatmap(confusion_matrix(raw_test['分类'].values, Y_test), linewidths=0.5, cmap='Greens',
                 annot=True, fmt='d', xticklabels=knn.classes_, yticklabels=knn.classes_)
ax.set_ylabel('真实')
ax.set_xlabel('预测')
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_title('混淆矩阵热力图')

plt.show()



















