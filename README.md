# 一. 训练模型：
`classifier_train.py`
可以修改 "training_settings.txt" 中的设定，包括
+ 使用几周的数据训练，可以是 1, 2, 或3
    - 若输入1，则使用 `stealing_accounts file 0` 和 `normal_account_file_0`的数据
    - 若输入2，则使用 `stealing_accounts file 0` `stealing_accounts file 1` 和 `normal_accounts file 0` `normal_accounts file 1` 的数据，分别为第0周和第1周，模型包含第0周到第1周的特征变化量
    - 若输入3，则使用 `stealing_accounts file 0` `stealing_accounts file 1` `stealing_accounts file 2` 和 `normal_accounts file 0` `normal_accounts file 1` `normal_accounts file 2` 的数据，分别为第0周，第1周，和第2周，模型包含第0周到第2周的特征变化量
+ 正样本（被盗账号）数据路径
可以放入多组样本，格式为[file_0, file_1, ...]
如我们使用两组数据的stealed accounts：
stealing_accounts file 0: [data/20210607_steal_fea_data.txt, data/stealing_real_accounts_20210810_20210817_feature_20210719.txt]
stealing_accounts file 1: [data/20210614_steal_fea_data.txt, data/stealing_real_accounts_20210810_20210817_feature_20210726.txt]
stealing_accounts file 2: [data/20210621_steal_fea_data.txt, data/stealing_real_accounts_20210810_20210817_feature_20210802.txt]
**如果只用2周的数据，则只使用第0周和第1周，第2周路径可以随意填写，将不被使用；如果只用1周的数据，则只使用第0周，第1周和第2周路径可以随意填写，将不被使用**
+ 负样本（正常账号）数据路径
+ 训练好的模型的保存路径
+ 是否保存图片: True/False
+ 图片保存路径（仅当True时有效），描述模型效果的图片，如混淆矩阵，准确率召回率曲线，特征重要性排序等

# 二. 设定模型包含哪些特征：
可修改`model_settings.py`
`usecols`: 读取数据中哪些变量
`personal`: 玩家基本信息（模型默认使用）
`relations`: 玩家互动信息（模型默认使用）
`performance`: 玩家战斗表现（模型默认不使用）
`diff`: 模型加入哪些变量的变化量

# 三. 已训练好的模型
已训练好的模型保存在`models`文件夹下。均为`XGBoost(n_estimators=1000, max_depth=2)`，特征包括玩家基本信息和玩家互动信息，和特征的变化量

现有的已训练好的模型如下：
- clf_xgb_june_data_1week_max_depth_2.pickle 用06/14这一周的数据（不考虑变化量）
- clf_xgb_june_data_2weeks_max_depth_2.pickle 用06/07和06/14这两周的数据（考虑一周的变化量）
- clf_xgb_june_data_3weeks_max_depth_2.pickle 用06/07, 06/14 和 06/21 这三周的数据（考虑两周的变化量）

以下三个模型是类似的，但是在训练时补充了八月份的实锤盗号的连续三周的数据
- clf_xgb_supplement_data_1week_max_depth_2.pickle
- clf_xgb_supplement_data_2weeks_max_depth_2.pickle
- clf_xgb_supplement_data_3weeks_max_depth_2.pickle

# 四. 用已训练好的模型预测新的样本：
可以修改 `predict_settings.txt` 中的设定，包括
+ 将使用几周的数据，可以是1,2,3
+ 训练好的模型的调用路径。已训练好的模型见上
+ 待预测数据路径，与 `training_settings.txt`中的格式类似
+ 是否保存图片
+ 图片保存路径

注意：
+ 预测模型和model_settings要匹配，即需要有相同的X.shape。如果在模型中增减变量，需要重新训练模型（运行 classifier_train.py）。
+ 如果预测时只用两周的数据，则应使用相对应的用两周数据训练出的模型，如 `clf_xgb_june_data_2weeks_max_depth_2.pickle`。
