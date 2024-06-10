core目录：  自定义的爬虫类代码, 数据分析类代码
resultdata:   爬取的信息放置目录比如csv, txt
setting:   存放全局设置的变量如cookie , headers ....
utils:    其他功能函数自定义的工具代码,以备使用时调用
main.py  执行程序


整体开发逻辑和步骤：

# 1. 数据清洗
    Data_Clean()
# 2. 获取高相关性列表
    columnList=Get_Coefficient_HighCorrColumns()
# 3. 生成有效数据源
    Set_FinalData(columnList)
# 4. 不同算法进行建模和预测
    Try_Different_Modules()   #尝试不同模型的结果

    ###===========  MAE越小，说明模型预测得越准确。
# 5. 决策树建模
    BaggingReg_Predict()          #Bagging回归        0.96    MAE值（平均绝对误差）： 679.0657421485693

    # RandomForest_Predict()  #随机森林回归预测     0.96    MAE值（平均绝对误差）：693.1051189629313
    # GradientBoostReg_Predict()   #梯度提升回归        0.94    MAE值（平均绝对误差）： 1029.9595700032057
    # Tree_Predict()                      #决策树       0.921     MAE值（平均绝对误差）： 967.9951798067017
    # ExtraTreeReg_Predict()          #极端随机树回归  0.92  MAE值（平均绝对误差）： 994.6986336653093
    # XGBReg_Predict()                   #              0.963  MAE             716.5915114270899

# 6. 调用模型文件后来对TestB文件内容预测
    Get_TestBPrice_Predict()

二、数据预处理
Data_Clean()函数进行数据预处理,列出TrainData中的缺失项目统计信息，并以视图的显示直观看到有缺失的列分布，删除所有缺失项的行信息，
并将notRepairedDamage列中的'-',替换成哑变量2 ，以方便后续处理

三、模型建模
Get_Coefficient_HighCorrColumns(),用来获取高相关性列表，找出与price列相关性从高到低的15列功能项作为自变量因子项目，生成最终有效数据
Try_Different_Modules(), 根据事先加入的五个模型预测算法进行六次各自交叉的验证，得到score矩阵和对应的回归检验矩阵，直观的看到MAE的分布状况
从而选取MAE最小值得BaggingReg回归预测方法，并进行TrainData的验证，最后生成saveclf.pkl文件模型。

四、模型性能分析
分别使用如下子函数单独求得Score值，并用sklearn.metrics评价自身模型，得出 MVE,MAE,MSE,R2等指标值
回归的决定系数R2
回归方差， 反应自变量与因变量之间的相关程度
MAE值（平均绝对误差， 反应自变量与因变量之间的相关程度 ，
MSE值（均方差） 计算均方误差
中值绝对误差  计算均方误差
