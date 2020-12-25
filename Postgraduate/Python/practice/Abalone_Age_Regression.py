import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection


warnings.filterwarnings('ignore')
data = pd.read_csv('./input/abalone_dataset.csv')

# print(data.head())
# print(data.shape)

# sns.countplot(x='sex', data=data)

# i = 1
# plt.figure(figsize=(16, 8))
# for col in data.columns[1:]:
#     plt.subplot(4, 2, i)
#     i = i + 1
#     sns.distplot(data[col])
# plt.tight_layout()

# sns.pairplot(data, hue='sex')

# fig, ax = plt.subplots(figsize=(12, 12))
# ax = sns.heatmap(data.corr(), linewidths=0.5, cmap='Greens', annot=True,
#                  xticklabels=data.corr().columns, yticklabels=data.corr().index)
# ax.xaxis.set_label_position('top')
# ax.xaxis.tick_top()

sex_onehot = pd.get_dummies(data['sex'], prefix='sex')
data[sex_onehot.columns] = sex_onehot
data['ones'] = 1
data['age'] = data['rings'] + 1.5

y = data['age']
features_with_ones = ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight",
                      "shell_weight", "sex_F", "sex_M", "ones"]
features_without_ones = ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight",
                         "shell_weight", "sex_F", "sex_M"]
X = data[features_without_ones]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=111)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

lr = LinearRegression()
lr.fit(X_train[features_without_ones], y_train)
rigde = Ridge(alpha=1.0)
rigde.fit(X_train[features_without_ones], y_train)
lasso = Lasso(alpha=0.01)
lasso.fit(X_train[features_without_ones], y_train)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

y_test_pred_lr = lr.predict(X_test[features_without_ones])
y_test_pred_rigde = rigde.predict(X_test[features_without_ones])
y_test_pred_lasso = lasso.predict(X_test[features_without_ones])
print('MSE:')
print(round(mean_squared_error(y_test, y_test_pred_lr), 4))
print(round(mean_squared_error(y_test, y_test_pred_rigde), 4))
print(round(mean_squared_error(y_test, y_test_pred_lasso), 4))
print('MAE:')
print(round(mean_absolute_error(y_test, y_test_pred_lr), 4))
print(round(mean_absolute_error(y_test, y_test_pred_rigde), 4))
print(round(mean_absolute_error(y_test, y_test_pred_lasso), 4))
print('R2:')
print(round(r2_score(y_test, y_test_pred_lr), 4))
print(round(r2_score(y_test, y_test_pred_rigde), 4))
print(round(r2_score(y_test, y_test_pred_lasso), 4))

# plt.figure(figsize=(9, 6))
# y_train_pred_lr = lr.predict(X_train[features_without_ones])
# plt.scatter(y_train_pred_lr, y_train_pred_lr - y_train, c='g', alpha=0.6)
# plt.scatter(y_test_pred_lr, y_test_pred_lr - y_test, c='r', alpha=0.6)
# plt.hlines(y=0, xmin=0, xmax=30, color='b', alpha=0.6)
# plt.ylabel("Residuals")
# plt.xlabel("Predict")
# plt.show()























