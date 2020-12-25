import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
import time


# start = time.process_time()


class TreeNode:

    def __init__(self, x_pos, y_pos, layer, class_labels=[0, 1, 2]):
        self.f = None
        self.v = None
        self.left = None
        self.right = None
        self.pos = (x_pos, y_pos)
        self.label_dist = None
        self.layer = layer
        self.class_labels = class_labels

    def __str__(self):
        if self.f is not None:
            return self.f + '\n<=' + str(round(self.v, 2))
        else:
            return str(self.label_dist) + '\n(' + str(np.sum(self.label_dist)) + ')'

    def predict(self, x):
        if self.f is None:
            return self.class_labels[np.argmax(self.label_dist)]
        elif x[self.f] <= self.v:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


class TreeNodeRegression:
    def __init__(self, y_mean, num_samples, x_pos, y_pos, layer):
        self.f = None
        self.v = None
        self.left = None
        self.right = None
        self.pos = (x_pos, y_pos)
        self.y_mean = y_mean
        self.num_samples = num_samples
        self.layer = layer

    def __str__(self):
        if self.f is not None:
            return self.f + '\n<=' + str(round(self.v, 2))
        else:
            return str(round(self.y_mean, 2)) + '\n(' + str(self.num_samples) + ')'

    def predict(self, x):
        if self.f is None:
            return self.y_mean
        elif x[self.f] <= self.v:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


def gini(y):
    return 1 - np.square(y.value_counts()/len(y)).sum()


def gini_with_weight(y, sample_weights):
    weights = sample_weights[y.index]
    return 1 - np.square(weights.groupby(y).sum()).sum()


def generate(X, y, x_pos, y_pos, nodes, min_leaf_samples, max_depth, layer, class_labels):
    current_node = TreeNode(x_pos, y_pos, layer, class_labels)
    current_node.label_dist = [len(y[y==v]) for v in class_labels]
    nodes.append(current_node)
    if len(X) < min_leaf_samples or gini(y) < 0.1 or layer > max_depth:
        return current_node
    max_gini, best_f, best_v = 0, None, None
    for f in X.columns:
        for v in X[f].unique():
            y1, y2 = y[X[f] <= v], y[X[f] > v]
            if len(y1) >= min_leaf_samples and len(y2) >= min_leaf_samples:
                imp_descent = gini(y) - gini(y1)*len(y1)/len(y) - gini(y2)*len(y2)/len(y)
                if imp_descent > max_gini:
                    max_gini, best_f, best_v = imp_descent, f, v
    current_node.f, current_node.v = best_f, best_v
    if current_node.f is not None:
        current_node.left = generate(X[X[best_f] <= best_v], y[X[best_f] <= best_v], x_pos - (2 ** (max_depth - layer)),
                                     y_pos - 1, nodes, min_leaf_samples, max_depth, layer + 1, class_labels)
        current_node.right = generate(X[X[best_f] > best_v], y[X[best_f] > best_v], x_pos + (2 ** (max_depth - layer)),
                                      y_pos - 1, nodes, min_leaf_samples, max_depth, layer + 1, class_labels)
    return current_node


def generate_with_weights(X, y, x_pos, y_pos, nodes, min_leaf_samples, max_depth, layer, class_labels, sample_weights):
    current_node = TreeNode(x_pos, y_pos, layer, class_labels)
    current_node.label_dist = [len(y[y == v]) for v in class_labels]
    nodes.append(current_node)
    if len(X) < min_leaf_samples or gini_with_weight(y, sample_weights) < 0.1 or layer > max_depth:
        return current_node
    max_gini, best_f, best_v = 0, None, None
    for f in X.columns:
        for v in X[f].unique():
            y1, y2 = y[X[f] <= v], y[X[f] > v]
            if len(y1) >= min_leaf_samples and len(y2) >= min_leaf_samples:
                imp_descent = gini_with_weight(y, sample_weights)\
                              - gini_with_weight(y1, sample_weights) * sample_weights[y1.index].sum() / sample_weights[y.index].sum()\
                              - gini_with_weight(y2,sample_weights) * sample_weights[y2.index].sum() / sample_weights[y.index].sum()
                if imp_descent > max_gini:
                    max_gini, best_f, best_v = imp_descent, f, v
    current_node.f, current_node.v = best_f, best_v
    if current_node.f is not None:
        current_node.left = generate_with_weights(X[X[best_f] <= best_v], y[X[best_f] <= best_v],
                                                  x_pos - (2 ** (max_depth - layer)), y_pos - 1, nodes,
                                                  min_leaf_samples, max_depth, layer + 1, class_labels, sample_weights)
        current_node.right = generate_with_weights(X[X[best_f] > best_v], y[X[best_f] > best_v],
                                                   x_pos + (2 ** (max_depth - layer)), y_pos - 1, nodes,
                                                   min_leaf_samples, max_depth, layer + 1, class_labels, sample_weights)
    return current_node


def generate_regression(X, y, x_pos, y_pos, nodes, min_leaf_samples, max_depth, layer):
    current_node = TreeNodeRegression(np.mean(y), len(y), x_pos, y_pos, layer)
    nodes.append(current_node)
    if len(X) < min_leaf_samples:
        return current_node
    min_square, best_f, best_v = 10e10, None, None
    for f in X.columns:
        for v in X[f].unique():
            y1, y2 = y[X[f] <= v], y[X[f] > v]
            split_error = np.square(y1 - np.mean(y1)).sum()*len(y1)/len(y) + np.square(y2 - np.mean(y2)).sum()*len(y2)/len(y)
            if (split_error < min_square and len(y1) >= min_leaf_samples
                    and len(y2) > min_leaf_samples and layer <= max_depth):
                min_square, best_f, best_v = split_error, f, v
    current_node.f, current_node.v = best_f, best_v
    if current_node.f is not None:
        current_node.left = generate_regression(X[X[best_f] <= best_v], y[X[best_f] <= best_v],
                                                x_pos - (2 ** (max_depth - layer)), y_pos - 1, nodes,
                                                min_leaf_samples, max_depth, layer + 1)
        current_node.right = generate_regression(X[X[best_f] > best_v], y[X[best_f] > best_v],
                                                 x_pos + (2 ** (max_depth - layer)), y_pos - 1, nodes,
                                                 min_leaf_samples, max_depth, layer + 1)
    return current_node


def decision_tree_classifier(X, y, min_leaf_samples, max_depth):
    nodes = []
    root = generate(X, y, 0, 0, nodes, min_leaf_samples=min_leaf_samples, max_depth=max_depth,
                    layer=1, class_labels=y.unique())
    return root, nodes


def decision_tree_regression(X, y, min_leaf_samples, max_depth):
    nodes = []
    root = generate_regression(X, y, 0, 0, nodes, min_leaf_samples=min_leaf_samples, max_depth=max_depth, layer=1)
    return root, nodes


def random_forest(X, y, num_trees, num_features, min_leaf_samples, max_depth):
    trees = []
    nodes_list = []
    for t in range(num_trees):
        features_sample = np.random.choice(X.columns, num_features, replace=False)
        sample_index = np.random.choice(X.index, len(X), replace=True)
        X_sample = X[features_sample].loc[sample_index, :]
        y_sample = y[sample_index]
        tree, nodes = decision_tree_classifier(X_sample, y_sample, min_leaf_samples, max_depth)
        trees.append(tree)
        nodes_list.append(nodes)
    return trees, nodes_list


def adaboost(X, y, num_trees, min_leaf_samples, max_depth):
    trees = []
    tree_weights = []
    sample_weights = pd.Series(data=np.ones_like(y) / len(y), index=y.index)
    for t in range(num_trees):
        nodes = []
        tree = generate_with_weights(X, y, 0, 0, nodes, min_leaf_samples, max_depth, 1, [-1, 1], sample_weights)
        y_pred = []
        for _, sample in X.iterrows():
            y_pred.append(tree.predict(sample))
        y_pred_ts = pd.Series(data=y_pred, index=y.index)
        error = sample_weights[y != y_pred_ts].sum() / sample_weights.sum()
        alpha_t = 0.5 * np.log((1 - error) / error)
        sample_weights = sample_weights * np.power(np.e, -alpha_t * y_pred_ts * y)
        sample_weights = sample_weights / sample_weights.sum()
        trees.append(tree)
        tree_weights.append(alpha_t)
    return trees, tree_weights


def rf_predict(trees, X_test):
    results = []
    for _, sample in X_test.iterrows():
        pred_y = []
        for tree in trees:
            pred_y.append(tree.predict(sample))
        results.append(pd.Series(pred_y).value_counts().idxmax())
    return results


def adaboost_predict(trees, weights, X_test):
    results = []
    for _, sample in X_test.iterrows():
        pred_y_sum = 0
        for t in range(len(trees)):
            pred_y_sum += trees[t].predict(sample) * weights[t]
        results.append(1 if pred_y_sum >= 0 else -1)
    return results


def get_networkx_graph(G, root):
    if root.left is not None:
        G.add_edge(root, root.left)
        get_networkx_graph(G, root.left)
    if root.right is not None:
        G.add_edge(root, root.right)
        get_networkx_graph(G, root.right)


def get_tree_pos(G):
    pos = {}
    for node in G.nodes:
        pos[node] = node.pos
    return pos


def get_node_color(G):
    color_dict = []
    for node in G.nodes:
        if node.f is None:
            label = np.argmax(node.label_dist)
            if label%3 == 0:
                color_dict.append('green')
            elif label%3 == 1:
                color_dict.append('red')
            else:
                color_dict.append('blue')
        else:
            color_dict.append('gray')
    return color_dict


def get_node_color2(G):
    color_dict = []
    for node in G.nodes:
        if node.f is None:
            color_dict.append('green')
        else:
            color_dict.append('gray')
    return color_dict


# iris = load_iris()
# iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# iris_df['target'] = iris.target
# iris_df.columns = ['sepal_len', 'sepal_width', 'petal_len', 'petal_width', 'target']
# X_iris = iris_df.iloc[:, :-1]
# y_iris = iris_df['target']

# root, nodes = decision_tree_classifier(X_iris, y_iris, min_leaf_samples=10, max_depth=4)
# fig, ax = plt.subplots(figsize=(9, 9))
# graph = nx.DiGraph()
# get_networkx_graph(graph, root)
# pos = get_tree_pos(graph)
# nx.draw_networkx(graph, pos=pos, ax=ax, node_shape='o', font_color='w', node_size=5000, node_color=get_node_color(graph))
# plt.box(False)
# plt.axis('off')
# plt.show()

# tree, tree_nodes = decision_tree_classifier(X_iris,y_iris, min_leaf_samples=10, max_depth=4)
# y_pred = []
# for _,sample in iris_df.iterrows():
#     y_pred.append(tree.predict(sample))
# fig, ax = plt.subplots(figsize=(9, 9))
# confusion_matrix = confusion_matrix(y_pred, iris_df['target'])
# ax = sns.heatmap(confusion_matrix, linewidths=0.5, cmap='Greens', annot=True, fmt='d',
#                  xticklabels=iris.target_names, yticklabels=iris.target_names)
# ax.set_ylabel('真实')
# ax.set_xlabel('预测')
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()
# plt.show()

# abalone = pd.read_csv('./input/abalone_dataset.csv')
# abalone['sex'] = abalone['sex'].map({'M': 1, 'F': 2, 'I': 3})
# X_abalone = abalone.iloc[:, :-1]
# y_abalone = abalone['rings']
#
# abalone_tree, nodes = decision_tree_regression(X_abalone, y_abalone, min_leaf_samples=50, max_depth=4)
# y_abalone_pred = []
# for _, sample in abalone.iterrows():
#     y_abalone_pred.append(abalone_tree.predict(sample))
# print("均方误差:", round(mean_squared_error(abalone["rings"], y_abalone_pred), 4))
# print("平均绝对误差:", round(mean_absolute_error(abalone["rings"], y_abalone_pred), 4))
# print("决定系数：", round(r2_score(abalone["rings"], y_abalone_pred), 4))

# fig, ax = plt.subplots(figsize=(20, 20))
# graph = nx.DiGraph()
# get_networkx_graph(graph,  abalone_tree)
# pos = get_tree_pos(graph)
# nx.draw_networkx(graph, pos=pos, ax=ax, node_shape='o', font_color='w', node_size=5000,
#                  node_color=get_node_color2(graph), font_size=10)
# plt.box(False)
# plt.axis('off')
# plt.show()

# X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size=0.5, stratify=y_iris)
# trees, nodes_list = random_forest(X_train, y_train, num_trees=10, num_features=3, min_leaf_samples=10, max_depth=3)

# fig, ax = plt.subplots(figsize=(30, 30))
# for i in range(len(trees)):
#     plt.subplot(3, 4, i+1)
#     graph = nx.DiGraph()
#     get_networkx_graph(graph, trees[i])
#     pos = get_tree_pos(graph)
#     nx.draw_networkx(graph, pos=pos, node_shape='o', font_color='w', node_size=5000, node_color=get_node_color(graph))
#     plt.box(False)
#     plt.axis('off')
# plt.show()

# y_test_pred = rf_predict(trees, X_test)
#
# fig, ax = plt.subplots(figsize=(9,9))
# confusion_matrix = confusion_matrix(y_test_pred, y_test)
# ax = sns.heatmap(confusion_matrix, linewidths=0.5, cmap='Greens', annot=True, fmt='d',
#                  xticklabels=iris.target_names, yticklabels=iris.target_names)
# ax.set_ylabel('真实')
# ax.set_xlabel('预测')
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()
# plt.show()

# sample, target = datasets.make_classification(n_samples=20, n_features=2, n_informative=2, n_redundant=0, n_classes=2,
#                                               n_clusters_per_class=2, scale=7.0, random_state=110)
#
# data = pd.DataFrame(data=sample, columns=['x1', 'x2'])
# data['label'] = target
# data['label'] = data['label'].map({0: -1, 1: 1})
# X, y = data.iloc[:, :-1], data['label']
#
# adatrees_two_dimension, _ = adaboost(X, y, num_trees=10, min_leaf_samples=2, max_depth=1)
#
# fig, ax = plt.subplots(figsize=(9, 9))
# sns.scatterplot(x=X.iloc[:, 0], y=X.iloc[:, 1], ax=ax, hue=y, s=100)
#
# for tree in adatrees_two_dimension:
#     if tree.f == 'x1':
#         plt.vlines(tree.v, data['x2'].min(), data['x2'].max(), color='gray')
#     if tree.f == 'x2':
#         plt.hlines(tree.v, data['x1'].min(), data['x1'].max(), color='gray')
#
# plt.show()

credit = pd.read_csv('./input/credit-data.csv')
credit.columns = ['sd', 'ruoul', 'age', 'due30-50', 'debt_ratio', 'income', 'loan_num', 'num_90due', 'num_rlines',
                  'due60-89', 'num_depents']
credit['sd'] = credit['sd'].map({1: 1, 0: -1})

X_neg_samples = credit[credit['sd'] == -1].sample(300)
credit_new = pd.concat([X_neg_samples, credit[credit['sd'] == 1].sample(300)])

X_credit = credit_new.iloc[:, 1:]
y_credit = credit_new['sd']
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2)

# 决策树
credit_tree, tree_nodes = decision_tree_classifier(X_credit_train, y_credit_train, min_leaf_samples=50, max_depth=8)

# fig, ax = plt.subplots(figsize=(10, 10))
# graph = nx.DiGraph()
# get_networkx_graph(graph, credit_tree)
# pos = get_tree_pos(graph)
# nx.draw_networkx(graph, pos=pos, ax=ax, node_shape="o", font_color="w", node_size=5000,
#                  node_color=get_node_color(graph), font_size=10)
# plt.box(False)
# plt.axis("off")
# plt.show()

# 随机森林
credit_rf_trees, _ = random_forest(X_credit_train, y_credit_train, num_trees=10, num_features=4, min_leaf_samples=30,
                                   max_depth=3)

# AdaBoost
credit_ada_trees, credit_ada_weight = adaboost(X_credit_train, y_credit_train, num_trees=10, min_leaf_samples=30,
                                               max_depth=3)

credit_tree_pred = []
for _, sample in X_credit_test.iterrows():
    credit_tree_pred.append(credit_tree.predict(sample))

credit_rf_pred = rf_predict(credit_rf_trees, X_credit_test)

credit_ada_pred = adaboost_predict(credit_ada_trees, credit_ada_weight, X_credit_test)

print(round(accuracy_score(y_credit_test, credit_tree_pred), 4))
print(round(accuracy_score(y_credit_test, credit_rf_pred), 4))
print(round(accuracy_score(y_credit_test, credit_ada_pred), 4))


# end = time.process_time()
#
# print(round(end - start, 4))














