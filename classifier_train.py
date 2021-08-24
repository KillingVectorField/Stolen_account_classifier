# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
import xgboost
from model_settings import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, use
import pickle
import seaborn as sns
from sklearn import metrics, model_selection

# sns.set_theme(style="ticks")

settings = open("training_settings.txt", "rb")
def my_readline():
    filename = settings.readline().decode().strip()
    filename = filename[filename.find(':')+2:]
    return filename
# %%
use_weeks = int(my_readline())
if use_weeks not in [1,2,3]: raise(ValueError("Can only take value 1, 2, 3"))

steal_account_file_list = [[s.strip() for s in my_readline()[1:-1].split(',')] for _ in range(use_weeks)]
[my_readline() for _ in range(3-use_weeks)]
normal_account_file_list = [[s.strip() for s in my_readline()[1:-1].split(',')] for _ in range(use_weeks)]
[my_readline() for _ in range(3-use_weeks)]

steal_account_by_dates = [read_longitudinal_date([file_list[i] for file_list in steal_account_file_list]) for i in range(len(steal_account_file_list[0]))]
steal_account_by_dates = [pd.concat([steal_account_by_dates[j][i] for j in range(len(steal_account_by_dates))]) for i in range(use_weeks)]
steal_account_by_dates = [df[np.logical_not(df.index.duplicated())] for df in steal_account_by_dates]
print("Number of stealed accounts:", steal_account_by_dates[0].shape[0])

normal_account_by_dates = [read_longitudinal_date([file_list[i] for file_list in normal_account_file_list]) for i in range(len(normal_account_file_list[0]))]
normal_account_by_dates = [pd.concat([normal_account_by_dates[j][i] for j in range(len(normal_account_by_dates))]) for i in range(use_weeks)]
normal_account_by_dates = [df[np.logical_not(df.index.duplicated())] for df in normal_account_by_dates]

normal_account_by_dates = [normal_account.drop(np.intersect1d(normal_account.index, steal_account_by_dates[0].index)) for normal_account in normal_account_by_dates] #正常玩家数据里需要剔除被盗号玩家
print("Number of normal accounts:", normal_account_by_dates[0].shape[0])

all_account_by_dates = []
for i in range(use_weeks):
    normal_account_by_dates[i]["is_steal"] = False # 添加负样本标签
    steal_account_by_dates[i]["is_steal"] = True # 添加正样本标签
    all_account_by_dates.append(pd.concat([normal_account_by_dates[i], steal_account_by_dates[i]]))
    all_account_by_dates[i]["ip_counts"] = all_account_by_dates[i]["vclientip"].value_counts()[all_account_by_dates[i]["vclientip"]].tolist() # ip地址重复次数
    # all_account_by_dates[i]["avg_skin_num"] = all_account_by_dates[i]["skin_num"] * 1.0 / (all_account_by_dates[i]["classical_num"] + all_account_by_dates[i]["funny_num"]) # 场均皮肤数
    # all_account_by_dates[i]["avg_skin_num"][all_account_by_dates[i]["classical_num"] + all_account_by_dates[i]["funny_num"] <= 1] = all_account_by_dates[i]["skin_num"][all_account_by_dates[i]["classical_num"] + all_account_by_dates[i]["funny_num"] <= 1]

if use_weeks==1:
    all_accounts = all_account_by_dates[0]
else:
    all_accounts = all_account_by_dates[1] # 采样中间一周的特征

if use_weeks>1:
    for feature in diff_list:
        if use_weeks==2:
            all_accounts[feature + "_diff"] = all_account_by_dates[1][feature] - all_account_by_dates[0][feature] # 在数据中加入两周的变化量
        elif use_weeks==3:
            all_accounts[feature + "_diff"] = all_account_by_dates[2][feature] - all_account_by_dates[0][feature] # 在数据中加入两周的变化量

for feature in ["ip_counts"]:
    if feature not in personal:
        personal.append(feature) # 将ip地址重复次数也包括金玩家基本信息

# friend_list = all_accounts["friend_list_game"].apply(lambda s: np.intersect1d(np.array(str(s).split(sep="+")), all_accounts.index, assume_unique=True))

# with open(r'models/friend_list_0614.pickle', 'wb') as f:
#     pickle.dump(friend_list, f)

# %% [markdown]
# ### ---------------特征对比可视化 --------------------

# for feature in personal[:-1]:
#     print(feature)
#     sns.histplot(all_accounts, x = feature, hue = "is_steal", bins = 40, stat = "density" , common_norm=False)
#         #,stat="density", common_norm=False)
#     plt.savefig(r'figures/comparison_0614_' + feature +'.png')
#     plt.show()

# # %%
# for feature in performance:
#     print(feature)
#     sns.histplot(all_accounts, x = feature, hue = "is_steal", bins = 40, stat="density", common_norm=False)
#     plt.show()

# # %%
# for feature in relations:
#     print(feature)
#     sns.histplot(all_accounts, x = feature, hue = "is_steal", bins = 40, stat="density", common_norm=False)
#     plt.savefig(r'figures/comparison_0614_' + feature +'.png')
#     plt.show()


# # %%
# plt.scatter(all_accounts["friend_num_game"], all_accounts["del_friend_num"], s=1, 
#             color = ["pink" if s else "blue" for s in all_accounts["is_steal"]])
# plt.xlabel("好友人数")
# plt.ylabel("已删除好友人数")
# plt.savefig(r'figures/del_friend.png')

###### -------------- Classifier -------------- ##############
if use_weeks == 1:  # 只使用某一周的特征
    X = all_accounts[personal+relations].drop(["segment", "friend_num_plat"], axis=1)
else: # 某一周特征以及特征变化量
    X = all_accounts[personal+relations+[feature +"_diff" for feature in diff_list]].drop(["segment", "friend_num_plat"], axis=1)
na_uid = X.index[np.where(X.isnull().sum(axis=1)!=0)[0]] # 删除带有nan, inf等数据的玩家
X.drop(na_uid, inplace=True)
y = all_accounts["is_steal"]
y.drop(na_uid, inplace=True)

# 神经网络分类器
# from sklearn import neural_network as nn
# clf_mlp = nn.MLPClassifier((64, 64), verbose=True, max_iter=500)
# clf_mlp.fit(X, y)
# pr_mlp = metrics.precision_recall_curve(y, clf_mlp.predict_proba(X)[:,1])
# metrics.plot_confusion_matrix(clf_mlp, X, y)
# precision_recall_fscore_support_clf_mlp = metrics.precision_recall_fscore_support(y, clf_mlp.predict(X))
model_type = my_readline()
print("Modelling method:", model_type)

if model_type.lower()=="dnn":
    import tensorflow as tf
    from tensorflow import keras
    def create_model():
        # create model
        model = keras.Sequential([keras.layers.Flatten(input_shape=(X.shape[1],)), keras.layers.Dense(64, activation='relu'), keras.layers.Dense(32, activation='relu'), keras.layers.Dense(1, activation='sigmoid')])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    clf = keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, epochs=10)

    clf.fit(X, y)

# Gradient boosting 分类器
# clf_gb = ensemble.GradientBoostingClassifier(n_estimators=1000, learning_rate=1.0, max_depth=2, random_state=0, verbose=True)
# clf_gb.fit(X,y)

elif model_type.lower() in ['xgb', 'xgboost']:
    # XGBoost 分类器
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=1000, use_label_encoder = False, learning_rate=1.0, max_depth=2, random_state=0, verbosity=0)
    clf.fit(X,y)


# 保存模型到本地文件
model_file = my_readline()
if model_type.lower()=="dnn":
    clf.model.save(model_file)
elif model_type.lower() in ['xgb', 'xgboost']:
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)
print("model saved:", model_file)

# with open(r'models/clf_xgb_0614_diff2_max_depth_2.pickle', 'wb') as f:
#     pickle.dump(clf, f)

# with open(r'models/clf_xgb_0614_diff2_max_depth_2.pickle', 'rb') as f:
#     clf = pickle.load(f) # 从本地文件中导入模型

# 模型评估
print("Training accuracy:", clf.score(X,y)) # 正确率

# model_selection.cross_val_score(clf, X, y, cv=5).mean() # CV 正确率

# 准确率、召回率、F score，以及准召曲线
# pr_xgb_0614 = metrics.precision_recall_curve(y, clf.predict_proba(X)[:,1]) # 仅用06/14数据的准召曲线
pr_xgb_diff = metrics.precision_recall_curve(y, clf.predict_proba(X)[:,1]) # 用06/14数据，以及纵向变化量的准召曲线
# precision_recall_fscore_support_clf_0614= metrics.precision_recall_fscore_support(y, clf.predict(X)) # 仅用06/14数据
precision_recall_fscore_support_clf_diff= metrics.precision_recall_fscore_support(y, clf.predict(X)) # 用06/14数据，以及纵向变化量
print("precision_recall_fscore_support:", precision_recall_fscore_support_clf_diff)
print("Confusion matrix:")
print(metrics.confusion_matrix(y, clf.predict(X)))

if my_readline().lower() in ["true", "t", "yes", "y"]:
    # 绘制准召曲线
    # plt.plot(pr_xgb_0614[0], pr_xgb_0614[1], label = "XGB 0614")
    # plt.plot([precision_recall_fscore_support_clf_0614[0][1]],[precision_recall_fscore_support_clf_0614[1][1]], "x", label="XGB 0614")
    plt.plot(pr_xgb_diff[0], pr_xgb_diff[1], label = "XGB including diff")
    plt.plot([precision_recall_fscore_support_clf_diff[0][1]],[precision_recall_fscore_support_clf_diff[1][1]], "x", label="XGB including diff")
    # plt.plot(pr_mlp[0], pr_mlp[1], label = "DNN")
    # plt.plot([precision_recall_fscore_support_clf_mlp[0][1]],[precision_recall_fscore_support_clf_mlp[1][1]], "x", label="current DNN diff")
    plt.xlabel("precision")
    plt.ylabel("recall")
    plt.legend()
    #plt.savefig(r'figures/pr_curve_0614_diff_xgb_max_depth_2.png')
    plt.savefig(my_readline())
    plt.show()

    # 混淆矩阵
    if model_type.lower() in ['xgboost','xgb']:
        #绘制混淆矩阵
        metrics.plot_confusion_matrix(clf, X, y)
        #plt.savefig(r'figures/confusion_matrix_xgb_diff_max_depth_2.png')
        plt.savefig(my_readline())
        plt.show()

        # 特征重要性排序
        importance_rank = clf.feature_importances_.argsort()
        show_features_num = 15
        plt.figure(figsize=[10.5,6])
        plt.barh(range(show_features_num), np.sort(clf.feature_importances_)[-show_features_num:], height=0.7, color='steelblue', alpha=0.8)      # 从下往上画
        plt.yticks(range(show_features_num), X.columns[importance_rank][-show_features_num:])
        #plt.title("XGB 20210614 data")
        plt.title("XGB Including Difference")
        #plt.savefig(r'figures/feature_importance_xgb_diff_max_depth_2.png')
        plt.savefig(my_readline())
        plt.show()
    else:
        [my_readline() for _ in range(2)]

    # 训练样本的打分分布
    sns.histplot(x = clf.predict_proba(X)[:,1], hue = y)
    #plt.savefig(r'figures/predicted_proba_hist_xgb_diff_max_depth_2.png')
    plt.savefig(my_readline())
    plt.show()

    sns.boxplot(x=y, y=clf.predict_proba(X)[:,1])
    plt.ylabel("predicted prob")
    #plt.savefig(r'figures/predicted_proba_boxplot_xgb_diff_max_depth_2.png')
    plt.savefig(my_readline())
    plt.show()

# 测试集表现
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=42) # 8:2 分割训练集测试集
# clf_train = XGBClassifier(n_estimators=1000, use_label_encoder = False, learning_rate=1.0, max_depth=2, random_state=0, verbosity=0)
# clf_train.fit(X_train, y_train)

# clf_train.score(X_train,y_train)
# clf_train.score(X_test,y_test)
# metrics.plot_confusion_matrix(clf_train, X_test, y_test)
# metrics.precision_recall_fscore_support(y_test, clf_train.predict(X_test))