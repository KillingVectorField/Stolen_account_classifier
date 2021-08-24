import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
from model_settings import *

settings = open("predict_settings.txt", "rb")
def my_readline():
    filename = settings.readline().decode().strip()
    filename = filename[filename.find(':')+2:]
    return filename

use_weeks = int(my_readline())

model_file = my_readline()
if model_file.endswith('h5'):
    import tensorflow as tf
    from tensorflow import keras
    def create_model():
        # create model
        model = keras.Sequential([keras.layers.Flatten(input_shape=(10,)), keras.layers.Dense(64, activation='relu'), keras.layers.Dense(32, activation='relu'), keras.layers.Dense(1, activation='sigmoid')])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    clf = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=10)
    clf.model = keras.models.load_model(model_file)
elif model_file.endswith('pickle'):
    with open(model_file, 'rb') as f:
        clf = pickle.load(f) # load model from file

#new_account_file_list = [r'data/' + 'stealing_real_accounts_20210810_20210817_feature_2021' + date +'.txt' for date in ['0719', '0726', '0802']]
new_account_file_list = [[s.strip() for s in my_readline()[1:-1].split(',')] for _ in range(use_weeks)]
[my_readline() for _ in range(3-use_weeks)]
new_account_by_dates = [read_longitudinal_date([file_list[i] for file_list in new_account_file_list]) for i in range(len(new_account_file_list[0]))]
new_account_by_dates = [pd.concat([new_account_by_dates[j][i] for j in range(len(new_account_by_dates))]) for i in range(use_weeks)]
new_account_by_dates = [df[np.logical_not(df.index.duplicated())] for df in new_account_by_dates] # 待预测玩家的数据文件

# add `ip_counts` to data
for i in range(use_weeks):
    new_account_by_dates[i]["ip_counts"] = new_account_by_dates[i]["vclientip"].value_counts()[new_account_by_dates[i]["vclientip"]].tolist() # 加入ip重复次数

if use_weeks==1:
    new_accounts = new_account_by_dates[0]
else:
    new_accounts = new_account_by_dates[1]

if use_weeks>1: # 有超过一周的数据，可以定义变化量
    for feature in diff_list:
        if use_weeks==2:
            new_accounts[feature + "_diff"] = new_account_by_dates[1][feature] - new_account_by_dates[0][feature]
        elif use_weeks==3:
            new_accounts[feature + "_diff"] = new_account_by_dates[2][feature] - new_account_by_dates[0][feature]

if use_weeks == 1:
    new_X = new_accounts[personal+relations].drop(["segment", "friend_num_plat"], axis=1) 
else: # 有超过一周的数据，可以定义变化量
    new_X = new_accounts[personal+relations+[feature +"_diff" for feature in diff_list]].drop(["segment", "friend_num_plat"], axis=1)

na_uid = new_X.index[np.where(new_X.isnull().sum(axis=1)!=0)[0]] # 删除带有nan, inf等数据的玩家
new_X.drop(na_uid, inplace=True) 

predicted_proba = clf.predict_proba(new_X)[:,1]
print("positive proportion (threshold=0.5): ", (predicted_proba > 0.5).mean())

output = new_accounts[["vopenid"]].drop(na_uid)
output["predicted_proba"] = predicted_proba

if model_file.endswith('pickle'):
    importance_rank = clf.feature_importances_.argsort()
    show_features_num = 20
    importance_list = []
    for feature, score in zip(new_X.columns[importance_rank][::-1][:show_features_num].tolist(), np.sort(clf.feature_importances_)[::-1][:show_features_num] * 100):
        importance_list.append(feature+':{score:.2f}'.format(score=score))
    tmp = [''] * output.shape[0]
    tmp[:show_features_num] = importance_list
    output["feature_importances"] = tmp

# 保存预测概率至本地文件
output_file = my_readline()
output.to_csv(output_file, sep='\t')
print("Output file:", output_file)

if my_readline().lower() in ["true", "t", "yes", "y"]:
    # 预测概率的分布
    sns.histplot(x = predicted_proba)
    plt.savefig(my_readline())
    plt.show()