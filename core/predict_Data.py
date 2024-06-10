# -*- coding:utf-8 -*-

import os
import warnings

import pandas as pd
import numpy as np
from sklearn import tree, svm, naive_bayes  # 导入sklean学习的各种模型
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
from imblearn.combine import SMOTEENN
import sklearn.metrics as sm  # 各种评价指标  无论利用机器学习算法进行回归、分类或者聚类时，评价指标，即检验机器学习模型效果的定量指标  https://blog.csdn.net/Yqq19950707/article/details/90169913
from sklearn import preprocessing  # 预处理
from sklearn.linear_model import LinearRegression  # 线性回归
from sklearn.linear_model import BayesianRidge  # 贝叶斯岭回归
from sklearn.linear_model import Lasso  # 岭回归    https://zhuanlan.zhihu.com/p/165493873
from sklearn.naive_bayes import GaussianNB  # 朴素贝叶斯   常用于分类
from sklearn.neighbors import KNeighborsClassifier  # KNN分类 常用于分类
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score  # 用于性能评估
from sklearn.tree import DecisionTreeRegressor  # 决策树回归预测
from sklearn.tree import ExtraTreeRegressor  # 极端随机树回归
from sklearn.neighbors import KNeighborsRegressor  # KNN回归预测
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor  # 梯度提升决策树
from sklearn.ensemble import BaggingRegressor  # 贝叶斯岭回归
from xgboost.sklearn import XGBRegressor  # 极端梯度提升树  pip install xgboost==1.3.1   https://blog.csdn.net/weixin_39791152/article/details/110512231

# from lightgbm.sklearn import LGBMRegressor
import joblib  # 用来存储模型

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder  # 量化特殊类别
from sklearn.model_selection import train_test_split  # 数据集划分
import seaborn as sns
smote_enn = SMOTEENN(random_state=0)


# 加入如下代码，否则中文不能正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['font.size'] = '12'
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

rawDataFile = '../resultdata/data.xlsx'
finalDataCsv = '../resultdata/datafinal.csv'
cleanDataCsv = '../resultdata/dataclean.csv'
df_final = pd.DataFrame()
regPredictList = list()  #放入十个模型预测的结果
data_Predict = [];data_True = []


# sklearn 回归预测 训练算法
clfs_Regressor = {
    # 'SVM_SVR': svm.SVR(),    #建立支持向量机回归模型对象,  # 支持向量机  support vector machine
    'DecisionTreeReg': DecisionTreeRegressor(),  # 决策树
    # 'RandomForestReg': RandomForestRegressor(n_estimators=20,criterion='mse',oob_score=True, random_state = 1),   #n_estimators决策树选了20个,
    # 'KNNReg': KNeighborsRegressor(weights='distance'),   #初始化距离加权回归的KNN回归器
    # 'LinearReg': LinearRegression(),
    # 'GradientBoostingReg': GradientBoostingRegressor(n_estimators=50),
    # 'AdaBoostReg': AdaBoostRegressor(n_estimators=500),#这里使用50个决策树
    # 'BaggingReg': BaggingRegressor(n_estimators=50),
    'ExtraTreeReg': ExtraTreeRegressor(),
    # 'ARDReg': ARDRegression(),
    # 'RANSACReg': RANSACRegressor(),
    # 'GNBReg':GaussianNB()
}




def Data_Clean():
    if not os.path.exists(finalDataCsv):
        print('数据清洗。。。。。。。。。。。。。。。。。')
        df = pd.read_excel(rawDataFile)
        print('清洗去重前数据数量：', df.shape[0])
        df_clean = df.drop_duplicates()
        print('清洗去重后数据数量：', df_clean.shape[0])
        df_final = df_clean[['品牌', '新车价格', '里程(万公里)', '价格']]
        df_final.to_csv(finalDataCsv, index=False)

    df_final = pd.read_csv(finalDataCsv, low_memory=False)
    le = LabelEncoder()  # 把品牌进行标签编码（Label Encoding）： 将类别标签映射为整数值，通常是0到n-1，其中n是类别的数量。
    df_final['品牌'] = le.fit_transform(df_final['品牌'])
    df_final = df_final.drop_duplicates()
    print(df_final)  # 显示df的数据类型
    df_final = df_final.drop(df_final[df_final['新车价格'] == '暂无报价'].index)  # 删除'新车价格'列中是'暂无报价'的内容
    df_final.loc[df_final['里程(万公里)'] == '百公里内', '里程(万公里)'] = '0.01'  # 更改'里程(万公里)'列中'百公里内'为0.01
    df_final = df_final.astype(float)
    print(df_final.dtypes)
    print(df_final.shape)  # 显示df的数据类型
    df_final.to_csv(cleanDataCsv, index=False)
    return df_final


def reduce_mem_usage(df):  # 通过调整数据类型，减少数据在内存中占用的空间
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def GNB_Predict():  #
    print('=================朴素贝叶斯算法训练模型中。。。==============================')
    p_DF = pd.read_csv(cleanDataCsv, index_col=None)
    p_DF = reduce_mem_usage(p_DF)  # 降低内存占用
    # zscore= preprocessing.MinMaxScaler()
    # pp_DF = zscore.fit_transform(p_DF)
    # print(pp_DF)
    # X=np.array(pp_DF[:,np.r_[0:3]])   #.所有自变量数据
    # y=np.array(pp_DF[:,3]).astype(float)     #因变量

    X = np.array(p_DF.iloc[:, np.r_[0:3]])  # .所有自变量数据
    y = np.array(p_DF.iloc[:, 3]).astype(float)  # 因变量

    # print(y.head(10))
    # X1,y1=SMOTE().fit_resample(X,y)
    # X1, y1 = smote_enn.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    # # 预测
    print('=================朴素贝叶斯算法预测值==============================')
    y_pred = (clf.predict(X_test))  # 预测特征值
    # return
    print('=================真实值==============================')
    print(y_test)  # 真实值
    print('预测得分score: ', clf.score(X_test, y_test))
    print(sm.accuracy_score(y_test, y_pred))
    print('auc值：', sm.roc_auc_score(y_test, y_pred))
    print('查准率：', sm.precision_score(y_test, y_pred))


def Tree_Predict(ndarrayTuple:tuple):  #
    X_train, X_test, y_train, y_test = ndarrayTuple
    print('=================决策树回归算法训练模型中。。。==============================')
    # from sklearn import tree
    # clf=tree.DecisionTreeClassifier(splitter='random',max_depth=100)
    # clf=tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
    clf = DecisionTreeRegressor()  # default的标准用的mse,"mse"比"mae"更加精确
    clf.fit(X_train, y_train)
    # print(X.shape)
    # print(X_train.shape)
    y_pred = clf.predict(X_test)
    # # 预测
    print('=================决策树算法预测值==============================')
    print(clf.predict(X_test))  # 预测特征值
    # return
    print('=================真实值==============================')
    print(y_test)  # 真实值
    p_score =clf.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("决策树回归",round(p_score,3)))

    # 回归指标用一下检验：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度 ，
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error


def Get_TestBPrice_Predict():  # 使用模型对测试数据进行预测，生成csv
    if not os.path.exists('../model/saveclf.pkl'):
        print('回归预测模型还未建立。。')
        return

    myCLF = joblib.load('../model/saveclf.pkl')  # 注意文件格式：pkl（二进制文件）
    print('=================回归模型文件预测验证。。。。。==============================')
    m_DF = pd.read_csv(cleanDataCsv, index_col=None)
    print(m_DF.shape)
    print((m_DF.columns))
    colist = m_DF.columns.tolist()
    colist.remove('price')
    colist.insert(0, 'SaleID')
    # print(colist)
    p_DF = pd.read_csv(cleanDataCsv,index_col=None, low_memory=False)
    p_DF = p_DF[colist]
    # print(p_DF.head(10))
    p_DF.dropna(axis=0, how='any', inplace=True)
    p_DF.notRepairedDamage = p_DF.notRepairedDamage.replace('-', 2)
    # msno.matrix(p_DF.sample(1000))
    # plt.show()

    # print(p_DF.shape)
    # print(p_DF.columns)

    # p_DF = reduce_mem_usage(p_DF)  #降低内存占用
    X_TestB = np.array(p_DF.iloc[:, 1:])  # .astype(float)    #取index=1---->最后一列的所有自变量数据

    print(X_TestB.shape)

    pred_Y_TestB = myCLF.predict(X_TestB)
    print('=================算法预测值==============================')
    print(pred_Y_TestB)  # 预测特征值
    p_DF['Pred_Price'] = np.around(pred_Y_TestB)
    p_DF.to_csv(clntestBPath, index=None)
    p_DF = p_DF[['SaleID', 'Pred_Price']]
    p_DF.to_csv('../data/used_car_sample_submit.csv', index=None)
    print('生成最终预测价格csv文件')


def SVM_Predict(ndarrayTuple:tuple):  # SVC是分类   SVR是回归   regression（SVR）和classification（SVC）两个部分
    X_train, X_test, y_train, y_test =  ndarrayTuple
    print('=================支持向量机算法训练模型中。。。==============================')
    # clf = svm.SVC(kernel="linear", decision_function_shape="ovo")
    # clf = svm.SVC(kernel='rbf',C=10,gamma=0.1,probability=True)   #用于二分类
    clf = svm.SVR()  # 建立支持向量机回归模型对象
    clf.fit(X_train, y_train)
    # # 预测
    print('=================支持向量机算法预测值==============================')
    y_pred = (clf.predict(X_test))  # 预测特征值
    print(y_pred)
    print('=================真实值==============================')
    print(y_test)  # 真实值
    p_score =clf.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("SVM回归",round(p_score,3)))
    # 回归指标：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error


def KNN_Predict(ndarrayTuple:tuple):
    X_train, X_test, y_train, y_test = ndarrayTuple
    print('=================KNN近邻回归算法训练模型中。。。==============================')
    # knn = KNeighborsClassifier(n_neighbors=5,weights='uniform')  # 选择邻近的5个点   #n_neighbors=12
    # knn = KNeighborsRegressor(weights='uniform')   #初始化平均回归的KNN回归器
    knn = KNeighborsRegressor(weights='distance')  # 初始化距离加权回归的KNN回归器
    knn.fit(X_train, y_train)  # 进行填充测试训练
    print('=================KNN近邻回归算法预测值==============================')  # 从评测结果可见，采用K近邻加权平均的回归策略可以获得较高的模型性能。
    y_pred = (knn.predict(X_test))  # 预测特征值
    print(y_pred)
    print('=================真实值==============================')
    print(y_test)  # 真实值
    p_score =knn.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("KNN近邻回归",round(p_score,3)))
    # 回归指标：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error


def RandomForest_Predict(ndarrayTuple:tuple):
    X_train, X_test, y_train, y_test = ndarrayTuple
    print('=================随机森林回归算法训练模型中。。。==============================')
    # knn = KNeighborsClassifier(n_neighbors=5,weights='uniform')  # 选择邻近的5个点   #n_neighbors=12
    clf = RandomForestRegressor(n_estimators=20, criterion='mse', oob_score=True,
                                random_state=1)  # n_estimators决策树选了20个
    clf.fit(X_train, y_train)  # 进行填充测试训练
    print('=================随机森林回归算法预测值==============================')  # 从评测结果可见，采用K近邻加权平均的回归策略可以获得较高的模型性能。
    y_pred = (clf.predict(X_test))  # 预测特征值
    print(y_pred)
    print('=================真实值==============================')
    print(y_test)  # 真实值
    p_score =clf.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("随机森林回归",round(p_score,3)))
    # 回归指标：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error


def LinearReg_Predict(ndarrayTuple:tuple):
    X_train, X_test, y_train, y_test = ndarrayTuple
    print('=================线性回归算法训练模型中。。。==============================')
    clf = LinearRegression()  #
    clf.fit(X_train, y_train)  # 进行填充测试训练
    print('=================线性回归算法预测值==============================')  # 从评测结果可见，采用K近邻加权平均的回归策略可以获得较高的模型性能。
    y_pred = (clf.predict(X_test))  # 预测特征值
    print(y_pred)
    print('=================真实值==============================')
    print(y_test)  # 真实值
    p_score =clf.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("线性回归",round(p_score,3)))
    # 回归指标：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error


def AdaBoostReg_Predict(ndarrayTuple:tuple):
    X_train, X_test, y_train, y_test = ndarrayTuple
    print('=================AdaBoost回归算法训练模型中。。。==============================')
    clf = AdaBoostRegressor(n_estimators=500)  # 这里使用50个决策树   #
    clf.fit(X_train, y_train)  # 进行填充测试训练
    print('=================AdaBoost回归算法预测值==============================')  #
    y_pred = (clf.predict(X_test))  # 预测特征值
    print(y_pred)
    print('=================真实值==============================')
    print(y_test)  # 真实值
    p_score =clf.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("AdaBoost回归",round(p_score,3)))
    # 回归指标：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error


def GradientBoostReg_Predict(ndarrayTuple:tuple):
    X_train, X_test, y_train, y_test = ndarrayTuple
    print('=================GradientBoost回归算法训练模型中。。。==============================')
    clf = GradientBoostingRegressor(n_estimators=50)
    clf.fit(X_train, y_train)  # 进行填充测试训练
    print('=================GradientBoost回归算法预测值==============================')  #
    y_pred = (clf.predict(X_test))  # 预测特征值
    print(y_pred)
    print('=================真实值==============================')
    print(y_test)  # 真实值
    p_score =clf.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("GradientBoost回归",round(p_score,3)))
    # 回归指标：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error


def BaggingReg_Predict(ndarrayTuple:tuple):
    X_train, X_test, y_train, y_test = ndarrayTuple
    print('=================Bagging回归算法训练模型中。。。==============================')
    clf = BaggingRegressor(n_estimators=50)
    clf.fit(X_train, y_train)  # 进行填充测试训练
    print('=================Bagging回归算法预测值==============================')  #
    y_pred = (clf.predict(X_test))  # 预测特征值
    print(y_pred)
    print('=================真实值==============================')
    print(y_test)  # 真实值
    p_score =clf.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("Bagging回归",round(p_score,3)))
    # 回归指标：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error


# 模型存储：
# beginT=time.time()
# joblib.dump(clf,'../model/saveclf.pkl')
# print('构建模型完成，../model/saveclf.pkl 耗时 {:.2f} s'.format(time.time()-beginT))

def ExtraTreeReg_Predict(ndarrayTuple:tuple):
    X_train, X_test, y_train, y_test = ndarrayTuple
    print('=================ExtraTree回归算法训练模型中。。。==============================')
    clf = ExtraTreeRegressor()
    clf.fit(X_train, y_train)  # 进行填充测试训练
    print('=================ExtraTree回归算法预测值==============================')  #
    y_pred = (clf.predict(X_test))  # 预测特征值
    print(y_pred)
    print('=================真实值==============================')
    print(y_test)  # 真实值
    p_score =clf.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("ExtraTree回归",round(p_score,3)))
    # 回归指标：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error


def Get_Train_Test_Split_Data(rs = 69): #返回获取最终数据集的分割结果，(元组类型)
    p_DF = pd.read_csv(cleanDataCsv, index_col=None)  # nrows=68000,
    p_DF = reduce_mem_usage(p_DF)  # 降低内存占用

    # zscore= preprocessing.MinMaxScaler()
    # pp_DF = zscore.fit_transform(p_DF)
    # print(pp_DF)
    # X=np.array(pp_DF[:,np.r_[0:3]])   #.所有自变量数据
    # y=np.array(pp_DF[:,3]).astype(float)     #因变量

    X = np.array(p_DF.iloc[:, np.r_[0:3]])  # .所有自变量数据
    y = np.array(p_DF.iloc[:, 3]).astype(float)  # 因变量
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=rs)  # 调整要调整random_state参数，您可以尝试不同的整数值，或者使用None来使其随机化
    return X_train, X_test, y_train, y_test


def XGBReg_Predict(ndarrayTuple: tuple):
    global data_Predict,data_True
    X_train, X_test, y_train, y_test = ndarrayTuple
    print('=================XGB回归算法训练模型中。。。==============================')
    # clf = ARDRegression()           # 0.675    2677.4653666656764
    # clf = BayesianRidge()             #  0.667 MAE值（平均绝对误差）： 2749.101653672683
    # clf = TheilSenRegressor()    #MAE值（平均绝对误差）： 6.817617622528577
    # clf = RANSACRegressor()    #  0.42  MAE值（平均绝对误差）： 2760.992745245779
    # clf = MLPRegressor()       #差  MAE值（平均绝对误差）：
    clf = XGBRegressor()  # 0.96 MAE值（平均绝对误差）： 716.5915114270899
    clf.fit(X_train, y_train)  # 进行填充测试训练
    print('=================XGB回归算法预测值==============================')  #
    y_pred = (clf.predict(X_test))  # 预测特征值
    print(y_pred);data_Predict = y_pred[:200]
    print('=================真实值==============================')
    print(y_test);data_True = y_test[:200]  # 真实值
    p_score =clf.score(X_test, y_test)
    print('预测得分score: ', p_score)
    regPredictList.append(("XGB回归",round(p_score,3)))

    import time
    beginT=time.time()
    joblib.dump(clf,'../model/saveclf.pkl')
    print('构建模型完成，../model/saveclf.pkl 耗时 {:.2f} s'.format(time.time()-beginT))

    # 回归指标：
    print('R平方值：', sm.r2_score(y_test, y_pred))  # 回归的决定系数R2
    print('回归方差：', sm.explained_variance_score(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MAE值（平均绝对误差）：', sm.mean_absolute_error(y_test, y_pred))  # 反应自变量与因变量之间的相关程度
    print('MSE值（均方差）：', sm.mean_squared_error(y_test, y_pred))  # 计算均方误差mean squared error
    print('中值绝对误差：', sm.median_absolute_error(y_test, y_pred))  # 计算均方误差mean squared error




def Modules_Debug():
    p_DF = pd.read_csv(cleanDataCsv, index_col=None)
    print(p_DF)
    p_DF = reduce_mem_usage(p_DF)  # 降低内存占用
    X = np.array(p_DF.iloc[:, np.r_[0:3]])  # .astype(float)    #取index=1---->最后一列的所有自变量数据
    y = np.array(p_DF.iloc[:, 3]).astype(float)  # 取index=0一列为因变量
    print(X[:100])
    print(y[:100])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=69)  # random_state的值相当于一种规则，通过设定为相同的数，每次分割的结果都是相同的

    # 模型评估性能！！
    # cross_val_score 函数的结果范围通常取决于所用的评估指标。对于分类问题，常用的指标包括准确率（accuracy）、精确率（precision）、召回率（recall）、F1 分数等；而对于回归问题，常用的指标包括均方误差（Mean Squared Error，MSE）、平均绝对误差（Mean Absolute Error，MAE）、R² 分数等。
    # 这些指标的范围是不同的：
    # 准确率（Accuracy）的范围是 [0, 1]，越接近1表示模型预测的结果越准确。
    # 精确率（Precision）的范围是 [0, 1]，越接近1表示模型在预测为正例的样本中真正为正例的比例越高。
    # 召回率（Recall）的范围是 [0, 1]，越接近1表示模型成功地找出了正例中的比例。
    # F1 分数的范围也是 [0, 1]，是精确率和召回率的调和平均值。

    # 对于回归问题：
    # 均方误差（Mean Squared Error，MSE）和平均绝对误差（Mean Absolute Error，MAE）没有严格的范围限制，但通常情况下越低越好，越接近0表示模型的性能越好。
    # R² 分数的范围是 [-∞, 1]，越接近1表示模型对目标变量的解释能力越强，而负值表示模型拟合得比简单平均要差。
    from sklearn.metrics import mean_absolute_error, make_scorer, accuracy_score
    models = [LinearRegression(),
              BayesianRidge(),
              Lasso()]

    result = dict()
    for model in models:
        model_name = str(model).split('(')[0]
        # cross_val_score 计算模型在每次交叉验证中的性能指标，并返回每次验证的性能指标，通常是准确率、精确度、召回率、F1 值
        # cross_val_score 会对数据进行交叉验证，对于每次验证，它会自动进行模型训练和测试，并返回每次验证的性能指标。最终返回一个包含所有验证性能指标的数组。通常，可以通过计算这些性能指标的平均值来得到模型的平均性能。
        scores = cross_val_score(model, X=X_train, y=y_train, verbose=0, cv=5,
                                 scoring=make_scorer(mean_absolute_error))  # 均方误差
        result[model_name] = scores  # cv1          5.145650       5.144085  5.072050  [不接近零，模型的性能不好]
        print(model_name + ' is finished')

    result = pd.DataFrame(result)  # 得到模型的平均性能
    result.index = ['cv' + str(x) for x in range(1, 6)]
    print(result)



    # 非线性回归预测，定义模型
    bayes = GaussianNB()  # 朴素贝斯叶
    # Decision = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)  #决策树   criterion = gini或者entropy,前者是基尼系数，后者是信息熵。
    Decision_Tree = DecisionTreeRegressor()  # 决策树回归
    # RandomForest = RandomForestRegressor()
    # GradBoost = GradientBoostingRegressor()
    # MLPReg = MLPRegressor(solver='lbfgs', max_iter=100)
    # XGBReg = XGBRegressor(n_estimators = 100, objective='reg:squarederror')
    # LGBMRegressor(n_estimators = 100)
    # from sklearn.ensemble import VotingClassifier
    # voting_clf = VotingClassifier(estimators=[('Bayes',bayes),
    #                                           ('Decision ',Decision)],voting='hard') #投票方式，hard,soft

    # 输出每个模型法预测准确率
    print(X_train.shape, y_train.shape)
    bayes.fit(X_train, y_train)
    y_pre1 = bayes.predict(X_test)
    print(bayes.__class__, accuracy_score(y_pre1, y_test))
    return

    for clf, label in zip([bayes, Decision_Tree],
                          ['Bayes', 'Decision', 'DecisionTreeRegressor']):
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
        # print(scores)
        print("accuracy Score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))  #

    for clf in (bayes, Decision_Tree):
        clf.fit(X_train, y_train)
        y_pre = clf.predict(X_test)
        print(clf.__class__, accuracy_score(y_pre, y_test))


def Try_Different_Modules():
    # 1. 读取有效数据到df
    p_DF = pd.read_csv(cleanDataCsv, index_col=None)  # nrows=88000,
    p_DF = reduce_mem_usage(p_DF)  # 降低内存占用

    # zscore= preprocessing.StandardScaler()
    # zscore = preprocessing.MinMaxScaler()
    # p_DF = zscore.fit_transform(p_DF)
    # print(p_DF)

    # 2. 取出自变量、因变量的数组
    X = np.array(p_DF.iloc[:, np.r_[0:3]])  # .所有自变量数据
    y = np.array(p_DF.iloc[:, 3]).astype(float)  # 因变量

    # 3. 按照7:3 分割数据集给Train Data 和 Validation Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=69)  # random_state的值相当于一种规则，通过设定为相同的数，每次分割的结果都是相同的

    # 4. 加入所有回归算法模型
    cross_numbs = 6  # 设置交叉检验的次数
    cv_score_list = []  # 交叉检验结果列表
    pre_y_list = []  # 各个回归模型预测的y值列表
    modelname_list = []
    for model_name, model_clf in clfs_Regressor.items():  # 读出每个回归模型对象
        scores = cross_val_score(model_clf, X_train, y_train, cv=cross_numbs,scoring='neg_mean_squared_error')  # 将每个回归模型导入交叉检验模型中做训练检验
        print("accuracy Score: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), model_name))
        cv_score_list.append(scores)  # 将交叉检验结果存入结果列表
        modelname_list.append(model_name)
        pre_y_list.append(model_clf.fit(X_train, y_train).predict(X_test))  # 将回归训练中得到的预测y存入列表
    # 模型效果指标评估
    n_samples, n_features = X_train.shape  # 总样本量,总特征数
    # explained_variance_score:解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量
    # 的方差变化，值越小则说明效果越差。
    # mean_absolute_error:平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度
    # ，其其值越小说明拟合效果越好。
    # mean_squared_error:均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的
    # 平方和的均值，其值越小说明拟合效果越好。
    # r2_score:判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因
    # 变量的方差变化，值越小则说明效果越差。
    model_metrics_name = [sm.explained_variance_score, sm.mean_absolute_error, sm.mean_squared_error,sm.r2_score]  # 回归评估指标对象集
    model_metrics_list = []  # 回归评估指标列表

    for i in range(len(clfs_Regressor)):  # 循环每个模型索引
        tmp_list = []  # 每个内循环的临时结果列表
        for m in model_metrics_name:  # 循环每个指标对象
            tmp_score = m(y_test, pre_y_list[i])  # 计算每个回归指标结果
            tmp_list.append(tmp_score)  # 将结果存入每个内循环的临时结果列表
        model_metrics_list.append(tmp_list)  # 将结果存入回归评估指标列表
    df1 = pd.DataFrame(cv_score_list, index=modelname_list)  # 建立交叉检验的数据框
    df2 = pd.DataFrame(model_metrics_list, index=modelname_list, columns=['EVS', 'MAE', 'MSE', 'R2'])  # 建立回归指标的数据框
    print('samples: %d \t features: %d' % (n_samples, n_features))  # 打印输出样本量和特征数量
    print(70 * '-')  # 打印分隔线
    print('交叉验证的结果:')  # 打印输出标题
    print(df1)  # 打印输出交叉检验的数据框
    print(70 * '-')  # 打印分隔线
    print('回归检验:')  # 打印输出标题
    print(df2)  # 打印输出回归指标的数据框
    print(70 * '-')  # 打印分隔线
    print('简称 \t 解释')  # 打印输出缩写和全名标题
    print('EVS \t 解释回归模型的方差得分')
    print('MAE \t 平均绝对误差')
    print('MSE \t 均方差')
    print('r2 \t r2判定系数')
    print(70 * '-')  # 打印分隔线
    # 模型效果可视化
    plt.figure()  # 创建画布
    plt.plot(np.arange(X_train.shape[0]), y_train, color='k', label='true y')  # 画出原始值的曲线
    color_list = ['r', 'b', 'g', 'y', 'c']  # 颜色列表
    linestyle_list = ['-', '.', 'o', 'v', '*']  # 样式列表
    for i, pre_y in enumerate(pre_y_list):  # 读出通过回归模型预测得到的索引及结果
        plt.plot(np.arange(X_test.shape[0]), pre_y_list[i], color_list[i], label=modelname_list[i])  # 画出每条预测结果线
    plt.title('regression result comparison')  # 标题
    plt.legend(loc='upper right')  # 图例位置
    plt.ylabel('real and predicted value')  # y轴标题
    plt.show()  # 展示图像

import random
def Draw_BarChart(infoList, str_Title, x_Label, y_Label):
    Value_list = []  # 准备数据
    Label_list = []  # 准备标签

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2., 1.03 * height,
                     "{:d}".format(int(height)))  # "{:.2f}".format(float(height))

    for il in infoList:
        Label_list.append(il[0])
        Value_list.append(il[1])
    p1 = plt.figure(figsize=(10, 6), dpi=80)  # 将
    mcolor = ['red', 'gold', 'turquoise', 'plum', 'g', 'c', 'm', 'y', 'orange', 'lightgreen', 'tan', 'khaki', 'pink',
              'skyblue', 'lawngreen', 'salmon']
    pcolor = random.sample(mcolor, len(infoList))  # 随机获取mcolor的元素 n个
    rects = plt.bar(Label_list, Value_list, align='center', color=pcolor)
    # plt.bar(x2, y2, color='g', align='center')
    plt.title(str_Title)
    plt.ylabel(y_Label)
    plt.xlabel(x_Label)
    plt.xticks(Label_list, Label_list, rotation=30)
    autolabel(rects)

    # p1.tight_layout()
    plt.savefig('./{}.jpg'.format(str_Title))
    plt.legend(rects, Label_list, bbox_to_anchor=(0.9, 0.9), loc="best", fontsize=10,
               bbox_transform=plt.gcf().transFigure)
    plt.show()
    pass


def Draw_BarHChart(infoList, str_Title, x_Label, y_Label):
    Value_list = []  # 准备数据
    Label_list = []  # 准备标签

    for il in infoList:
        Label_list.append(il[0])
        Value_list.append(il[1])

    Label_list.reverse()  # 反向一下，显示从大到小
    Value_list.reverse()
    # """
    # 绘制水平条形图方法barh
    # 参数一：y轴
    # 参数二：x轴
    # """
    p1 = plt.figure(figsize=(15, 6), dpi=80)  # 将
    mcolor = ['red', 'gold', 'turquoise', 'plum', 'g', 'c', 'm', 'y', 'orange', 'lightgreen', 'tan', 'khaki', 'pink',
              'skyblue', 'lawngreen', 'salmon']
    pcolor = random.sample(mcolor, len(infoList))  # 随机获取mcolor的元素 n个
    rects = plt.barh(range(len(infoList)), Value_list, height=0.7, color=pcolor, alpha=0.8)  # 从下往上画
    plt.yticks(range(len(infoList)), Label_list)
    plt.xlim(min(Value_list) * 0.9, max(Value_list) * 1.1)
    plt.xlabel(y_Label)
    plt.title(str_Title)
    for x, y in enumerate(Value_list):
        plt.text(y + 0.0, x - 0.1, '%s' % y)

    # p1.tight_layout()
    plt.savefig('./{}.jpg'.format(str_Title))
    # plt.legend(rects, Label_list, bbox_to_anchor=(0.7, 0.2), loc="lower center", fontsize=10,
    #            bbox_transform=plt.gcf().transFigure)
    plt.show()
    pass


def main():

    #
    # myCLF = joblib.load('../model/saveclf.pkl')  # 注意文件格式：pkl（二进制文件）
    #
    # df_final = pd.read_csv(finalDataCsv, low_memory=False)
    # le = LabelEncoder()  # 把品牌进行标签编码（Label Encoding）： 将类别标签映射为整数值，通常是0到n-1，其中n是类别的数量。
    # df_final['品牌'] = le.fit_transform(df_final['品牌'])
    # print(le.classes_)
    #
    # print('=================回归模型文件预测验证。。。。。==============================')
    # carCorp = input('===========> 请输入二手车的品牌:')
    # carNumb = list(le.classes_).index(carCorp)
    # # print(carNumb)
    # carNewP = input('===========> 请输入发售购买价格(万元):')
    # carLiCheng = input('===========> 请输入里程数(万公里):')
    #
    #
    # # flat_array = np.ravel([carNumb,carNewP,carLiCheng])  #list转一维数组 ，方法同下
    #
    # x_TestB = np.array(pd.DataFrame([carNumb,carNewP,carLiCheng]))
    # x_TestB = x_TestB.astype(float).reshape(1,3)
    # # print(x_TestB,x_TestB.shape)
    # # print(type(x_TestB))
    # pred_Y_TestB = myCLF.predict(x_TestB)
    # print('=================算法预测值==============================')
    # print('预测本台二手车价格： {}万'.format(pred_Y_TestB[0]))  # 预测特征值
    #
    # return



    # 1.数据清洗和去重干扰项目
    df = Data_Clean();print(df)

    # 2. 不同算法进行建模和预测
    # Modules_Debug()      #选择不同模型做交叉验证     舍弃不用
    # Try_Different_Modules()   #尝试不同模型的结果
    # return

    #3 .  Top 10  & 热力图
    df_final = pd.read_csv(finalDataCsv, low_memory=False)
    TopList = (df_final.品牌.value_counts())
    print(TopList)
    print(TopList.index)
    print(TopList.values)

    plt.figure(figsize=(15, 8), dpi=80)
    TopList[:10].plot(kind='bar')
    plt.xlabel('品牌')
    plt.ylabel('数量')
    plt.title('二手车品牌前十大量视图')
    plt.xticks(rotation=30)
    plt.show()

    df_clean = pd.read_csv(cleanDataCsv, low_memory=False)
    df_corr = df_clean[['品牌','新车价格','里程(万公里)', '价格']].corr(method='pearson')
    print(df_corr)
    plt.figure(figsize=(10,8),dpi=80)
    #当相关系数接近1时，表示两个变量呈正相关关系；当相关系数接近-1时，表示两个变量呈负相关关系；当相关系数接近0时，表示两个变量之间没有线性相关关系
    #当相关系数的绝对值大于0.7时，可以认为存在较强的相关性
    sns.heatmap(df_corr, annot=True, vmax=1, square=True, cmap='Blues')
    plt.title('Corr Missing of Columns & label Result: ')
    plt.show()
    #
    return
    ###===========  MAE越小，说明模型预测得越准确。
    # # 5. 建模
    #
    # ndarrayData = Get_Train_Test_Split_Data()
    # XGBReg_Predict(ndarrayData)  # XGB回归算法      0.89  MAE             2.75
    # BaggingReg_Predict(ndarrayData)  # Bagging回归        0.8826    MAE值（平均绝对误差）： 2.913
    # RandomForest_Predict(ndarrayData)  # 随机森林回归预测     0.87    MAE值（平均绝对误差）：2.919
    # GradientBoostReg_Predict(ndarrayData)  # 梯度提升回归       0.870   MAE值（平均绝对误差）： 3.39
    # Tree_Predict(ndarrayData)  # 决策树      0.83     MAE值（平均绝对误差）： 3.63
    # ExtraTreeReg_Predict(ndarrayData)  # 极端随机树回归  0.84  MAE值（平均绝对误差）： 3.65
    # #
    # LinearReg_Predict(ndarrayData)  # 线性回归              0.684    MAE值（平均绝对误差）： 5.02
    # KNN_Predict(ndarrayData)            #KNN回归                0.64   MAE值（平均绝对误差）： 2.9505
    # AdaBoostReg_Predict(ndarrayData)    # Adaboost回归  结果很差 0.70 MAE值（平均绝对误差）： 6.466
    # SVM_Predict(ndarrayData)  # SVR回归  结果很差   0.40   MAE值（平均绝对误差）： 4.4236

    #6. 十个模型的得分比对
    # print(regPredictList)
    regPredictList=[('XGB回归', 0.947),
                    ('Bagging回归', 0.936),
                    ('随机森林回归', 0.941),
                    ('GradientBoost回归', 0.899),
                    ('决策树回归', 0.823),
                    ('ExtraTree回归', 0.911),
                    ('线性回归', 0.841),
                    ('KNN近邻回归', 0.912),
                    ('SVM回归', 0.265),
                    ('AdaBoost回归', 0.897)]

    new_regPredictList = sorted(regPredictList,key = lambda x:x[1],reverse=True)
    Draw_BarHChart(new_regPredictList,'十种回归模型的得分比对','模型名称','accuracy')

    #7. True & Predict Value Chart
    XGBReg_Predict(ndarrayData)
    # 生成x轴的数据（这里假设数据点是等间距的）
    x_values = range(1, len(data_Predict) + 1)

    print(max(data_Predict))
    # 绘制预测值和真实值的曲线
    p1 = plt.figure(figsize=(15, 6), dpi=80)  # 将
    plt.plot(x_values, data_Predict, label='Predicted Values', marker='o')
    plt.plot(x_values, data_True, label='True Values', marker='x')

    # 添加图例和标签
    plt.legend()
    plt.xlabel('Data Points')
    plt.ylabel('Values')

    # 显示图形
    plt.title('Predicted vs True Values')
    plt.show()


    #8。 根据用户输入信息进行模型预测值计算
    if not os.path.exists('../model/saveclf.pkl'):
        print('回归预测模型还未建立。。')
        return

    myCLF = joblib.load('../model/saveclf.pkl')  # 注意文件格式：pkl（二进制文件）

    df_final = pd.read_csv(finalDataCsv, low_memory=False)
    le = LabelEncoder()  # 把品牌进行标签编码（Label Encoding）： 将类别标签映射为整数值，通常是0到n-1，其中n是类别的数量。
    df_final['品牌'] = le.fit_transform(df_final['品牌'])
    print(le.classes_)

    print('=================回归模型文件预测验证。。。。。==============================')
    carCorp = input('===========> 请输入二手车的品牌:')
    carNumb = list(le.classes_).index(carCorp)
    # print(carNumb)
    carNewP = input('===========> 请输入发售购买价格(万元):')
    carLiCheng = input('===========> 请输入里程数(万公里):')


    # flat_array = np.ravel([carNumb,carNewP,carLiCheng])  #list转一维数组 ，方法同下

    x_TestB = np.array(pd.DataFrame([carNumb,carNewP,carLiCheng]))
    x_TestB = x_TestB.astype(float).reshape(1,3)
    # print(x_TestB,x_TestB.shape)
    # print(type(x_TestB))
    pred_Y_TestB = myCLF.predict(x_TestB)
    print('=================算法预测值==============================')
    print('预测本台二手车价格： {}万'.format(pred_Y_TestB[0]))  # 预测特征值






    return


    # 6. 调用模型文件后来对TestB文件内容预测
    # Get_TestBPrice_Predict()

    pass


if __name__ == '__main__':
    main()
