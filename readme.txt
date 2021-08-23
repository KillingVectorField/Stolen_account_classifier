训练模型：
可以修改 "training_settings.txt" 中的路径，包括
正样本数据路径
负样本数据路径
训练好的模型的保存路径
图片保存路径

可以放入多组样本，格式为[file_0.txt, file_1.txt, ...]
如我们使用两组数据的stealed accounts：
stealing_accounts file 0: [data/20210607_steal_fea_data.txt, data/stealing_real_accounts_20210810_20210817_feature_20210719.txt]
stealing_accounts file 1: [data/20210614_steal_fea_data.txt, data/stealing_real_accounts_20210810_20210817_feature_20210726.txt]
stealing_accounts file 2: [data/20210621_steal_fea_data.txt, data/stealing_real_accounts_20210810_20210817_feature_20210802.txt]

修改模型参数：
"model_settings.py" 文件中
usecols: 读取数据中哪些变量
personal: 玩家基本信息（模型默认使用）
relations: 玩家互动信息（模型默认使用）
performance: 玩家战斗表现（模型默认不使用）
diff: 模型加入哪些变量的变化量

用于预测：
可以修改 "predict_settings.txt" 中的路径，包括
训练好的模型的调用路径
待预测数据路径
图片保存路径

注意：预测模型和model_settings要匹配，即需要又相同的X.shape。如果再模型中增减变量，需要重新训练模型（运行 classifier_train.py）
