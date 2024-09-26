# 导入必要的库
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

def RF_cal_result(data_csv, n_estimators):
    data = np.array(data_csv.values.tolist())
    T2s_all, T2c_all, time_opt = data[:, 0:128], data[:, 128:256], data[:, 256]
    n_samples, n_features = T2s_all.shape
    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(T2s_all, time_opt, test_size=0.9)
    # 创建随机森林回归模型
    rf_regressor = RandomForestRegressor(n_estimators=n_estimators)
    rf_regressor.fit(X_train, y_train)
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"n_estimators: {n_estimators}, 均方误差 (MSE):", mse)
    print(f"n_estimators: {n_estimators}, 决定系数 (R^2):", r2)
    return n_samples, mse, y_test, y_pred, rf_regressor

# 数据的读取
data_csv = pd.read_csv('data_type_all_new_500.csv')

# 设置不同的 n_estimators 值
n_estimators_list = [1,2,4,7,12,21,35,59,100]

# 遍历不同的 n_estimators
for n_estimators in n_estimators_list:
    n_samples_all, mse_all, y_test_all, y_pred_all, rf_regressor_all = RF_cal_result(data_csv, n_estimators)
    print("均方误差 (MSE ALL)     :", mse_all)

    # 保存预测结果到 CSV 文件
    data = np.array(data_csv.values.tolist())
    T2s_all, T2c_all, time_opt = data[:, 0:128], data[:, 128:256], data[:, 256]
    y = rf_regressor_all.predict(T2s_all)
    y0 = np.random.rand(1)
    ysave = np.concatenate((y0, y))
    results_df = pd.DataFrame({'time_opt': time_opt, 'predicted': y})
    results_df.to_csv(f'predictions_n_estimators_{n_estimators}.csv', index=False)
    plt.scatter(time_opt, y, color='blue')
    plt.plot([min(time_opt), max(time_opt)], [min(time_opt), max(time_opt)], color='red', linestyle='--')
    plt.show()

print("代码执行完成")
