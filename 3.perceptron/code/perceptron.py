import numpy as np
import pandas as pd
from itertools import combinations


class model_perceptron(object):
    """
    功能：实现感知机算法（原始形式）
    参数 w：权重，默认都为None
    参数 b：偏置项，默认为0
    参数 alpha：学习率，默认为0.001
    参数 iter_epoch：迭代轮数，默认最大为1000
    """
    def __init__(self, w = None, b = 0, alpha = 0.001, max_iter_epoch = 1000):
        self.w = w
        self.b = b
        self.alpha = alpha
        self.max_iter_epoch = max_iter_epoch
        
    def linear_model(self, X):
        """功能：实现线性模型"""
        return np.dot(X, self.w) + self.b
         
    def fit(self, X, y):
        """
        功能：拟合感知机模型
        参数 X：训练集的输入数据
        参数 y：训练集的输出数据
        """
        # 按训练集的输入维度初始化w
        self.w = np.zeros(X.shape[1])
        # 误分类的样本就为True
        state = np.sign(self.linear_model(X)) != y
        # 迭代轮数
        total_iter_epoch = 1
        
        while state.any() and (total_iter_epoch <= self.max_iter_epoch):
            # 使用误分类点进行权重更新
            self.w += self.alpha * y[state][0] * X[state][0]
            self.b += self.alpha * y[state][0]
            
            # 状态更新
            total_iter_epoch += 1
            state = np.sign(self.linear_model(X)) != y
            
        print(f"fit model_perceptron(alpha = {self.alpha}, max_iter_epoch = {self.max_iter_epoch}, total_iter_epoch = {min(self.max_iter_epoch, total_iter_epoch)})")
        
    def predict(self, X):
        """
        功能：模型预测
        参数 X：测试集的输入数据
        """
        return np.sign(self.linear_model(X)) 
    
    def score(self, X, y):
        """
        功能：模型评价（准确率）
        参数 X：测试集的输入数据
        参数 y：测试集的输出数据
        """
        y_predict = self.predict(X)
        y_score = (y_predict == y).sum() / len(y)
        
        return y_score
    
 

class perceptron_dual(object):
    """
    功能：实现感知机的对偶形式
    参数 beta：每个实例点更新的次数组成的向量
    参数 w：权重，默认都为None
    参数 b：偏置项，默认为0
    参数 alpha：学习率，默认0.001
    参数 max_iter_epoch：最大迭代次数，默认为1000
    """
    def __init__(self, alpha = 0.001, max_iter_epoch = 1000):
        self.beta = None
        self.w = None
        self.b = 0
        self.alpha = alpha
        self.max_iter_epoch = max_iter_epoch
        
    def fit(self, X, y):
        # 实例点的数量
        xnum = X.shape[0]
        # 初始化
        self.beta = np.zeros(xnum)
        
        # gram矩阵
        gram = np.dot(X, X.T)
        
        # 迭代条件
        state = y*((self.beta * y * gram).sum(axis = 1) + self.b) <= 0
        iter_epoch = 1
        
        while state.any() and (iter_epoch <= self.max_iter_epoch):
            nx = X[state][0]
            ny = y[state][0]
            index = (X == nx).argmax()
            self.beta[index] += self.alpha
            self.b += ny
            
            # 更新条件
            iter_epoch += 1
            state = y*((self.beta * y * gram).sum(axis = 1) + self.b) <= 0
        
        # 通过beta计算出w
        self.w = ((self.beta * y).reshape(-1, 1) * X).sum(axis = 0) 
        print(f"fit perceptron_dual(alpha = {self.alpha}, total_iter_epoch = {min(self.max_iter_epoch, iter_epoch)})")
              
    def predict(self, X):
        """
        功能：模型预测
        参数 X：测试集的输入数据
        """
        y_predict = np.sign(X @ self.w  + self.b)
        return y_predict
    
    def score(self, X, y):
        """
        功能：模型评价（准确率）
        参数 X：测试集的输入数据
        参数 y：测试集的输出数据
        """
        y_score = (self.predict(X) == y).sum() / len(y)
        
        return y_score
    
    

class perceptron_ovr(object):
    """
    功能：实现感知机的多分类情形（采用one vs. rest策略）
    参数 w：权重，默认都为None
    参数 b：偏置项，默认为0
    参数 alpha：学习率，默认0.001
    参数 max_iter_epoch：最大迭代次数，默认为1000
    """
    def __init__(self, alpha = 0.001, max_iter_epoch = 1000):
        self.w = None
        self.b = None
        self.alpha = alpha
        self.max_iter_epoch = max_iter_epoch
        
    def linear_model(self, X):
        """功能：实现线性模型"""
        return np.dot(self.w, X.T) + self.b
        
    def fit(self, X, y):
        """
        功能：拟合感知机模型
        参数 X：训练集的输入数据
        参数 y：训练集的输出数据
        """
        # 生成各分类器对应的标记
        self.y_class = np.unique(y)
        y_ovr = np.vstack([np.where(y == i, 1, -1) for i in self.y_class])
        
        # 初始化w, b
        self.w = np.zeros([self.y_class.shape[0], X.shape[1]])
        self.b = np.zeros([self.y_class.shape[0], 1])
        
        # 拟合各分类器，并更新相应维度的w和b
        for index in range(self.y_class.shape[0]):
            ppn = model_perceptron(alpha = self.alpha, max_iter_epoch = self.max_iter_epoch)
            ppn.fit(X, y_ovr[index])
            self.w[index] = ppn.w
            self.b[index] = ppn.b
            
    def predict(self, X):
        """
        功能：模型预测
        参数 X：测试集的输入数据
        """
        # 值越大，说明第i维的分类器将该点分得越开，即属于该分类器的概率值越大
        y_predict = self.linear_model(X).argmax(axis = 0)
        
        # 还原原数据集的标签
        for index in range(self.y_class.shape[0]):
            y_predict = np.where(y_predict == index, self.y_class[index], y_predict)
            
        return y_predict
    
    def score(self, X, y):
        """
        功能：模型评价（准确率）
        参数 X：测试集的输入数据
        参数 y：测试集的输出数据
        """
        y_score = (self.predict(X) == y).sum()/len(y)
        return y_score
    

class perceptron_ovo(object):
    """
    功能：实现感知机的多分类情形（采用one vs. one策略）
    参数 w：权重，默认都为None
    参数 b：偏置项，默认为0
    参数 alpha：学习率，默认0.001
    参数 max_iter_epoch：最大迭代次数，默认为1000
    """
    def __init__(self, alpha = 0.001, max_iter_epoch = 1000):
        self.w = None
        self.b = None
        self.alpha = alpha
        self.max_iter_epoch = max_iter_epoch
        
    def linear_model(self, X):
        """功能：实现线性模型"""
        return np.dot(self.w, X.T) + self.b
        
    def fit(self, X, y):
        """
        功能：拟合感知机模型
        参数 X：训练集的输入数据
        参数 y：训练集的输出数据
        """
        # 生成各分类器对应的标记（使用排列组合）
        self.y_class = np.unique(y)
        self.y_combine = [i for i in combinations(self.y_class, 2)]
        
        # 初始化w和b
        clf_num = len(self.y_combine)
        self.w = np.zeros([clf_num, X.shape[1]])
        self.b = np.zeros([clf_num, 1])
        
        for index, label in enumerate(self.y_combine):
            # 根据各分类器的标签选取数据集
            cond = pd.Series(y).isin(pd.Series(label))
            xdata, ydata = X[cond], y[cond]
            ydata = np.where(ydata == label[0], 1, -1)
            
            # 拟合各分类器，并更新相应维度的w和b
            ppn = model_perceptron(alpha = self.alpha, max_iter_epoch = self.max_iter_epoch)
            ppn.fit(xdata, ydata)
            self.w[index] = ppn.w
            self.b[index] = ppn.b
    
    def voting(self, y):
        """
        功能：投票
        参数 y：各分类器的预测结果，接受的是元组如(1, 1, 2)
        """
        # 统计分类器预测结果的出现次数
        y_count = np.unique(np.array(y), return_counts = True)
        # 返回出现次数最大的结果位置索引
        max_index = y_count[1].argmax()
        # 返回某个实例投票后的结果
        y_predict = y_count[0][max_index]
        
        return y_predict  
                      
    def predict(self, X):
        """
        功能：模型预测
        参数 X：测试集的输入数据
        """
        # 预测结果
        y_predict = np.sign(self.linear_model(X))
        # 还原标签（根据排列组合的标签）
        for index, label in enumerate(self.y_combine):
            y_predict[index] = np.where(y_predict[index] == 1, label[0], label[1])
        
        # 列为某一个实例的预测结果，打包用于之后的投票
        predict_zip = zip(*(i.reshape(-1) for i in np.vsplit(y_predict, self.y_class.shape[0])))
        # 投票得到预测结果
        y_predict = list(map(lambda x: self.voting(x), list(predict_zip)))
        
        return np.array(y_predict)
    
    def score(self, X, y):
        """
        功能：模型评价（准确率）
        参数 X：测试集的输入数据
        参数 y：测试集的输出数据
        """
        y_predict = self.predict(X)
        y_score = (y_predict == y).sum() / len(y)
        
        return y_score
