import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def decision_plot(X, Y, clf, test_idx = None, resolution = 0.02):
    """
    功能：画分类器的决策图
    参数 X：输入实例
    参数 Y：实例标记
    参数 clf：分类器
    参数 test_idx：测试集的index
    参数 resolution：np.arange的间隔大小
    """
    # 标记和颜色设置
    markers = ['o', 's', 'x', '^', '>']
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(Y))])
    
    # 图形范围
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    x = np.arange(xmin, xmax, resolution)
    y = np.arange(ymin, ymax, resolution)
    
    # 网格
    nx, ny = np.meshgrid(x, y)
    
    # 数据合并
    xdata = np.c_[nx.reshape(-1), ny.reshape(-1)]
    
    # 分类器预测
    z = clf.predict(xdata)
    z = z.reshape(nx.shape)
    
    # 作区域图
    plt.contourf(nx, ny, z, alpha = 0.4, cmap = cmap)
    plt.xlim(nx.min(), nx.max())
    plt.ylim(ny.min(), ny.max())
    
    # 画点
    for index, cl in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y == cl, 0], y=X[Y == cl, 1],
                    alpha=0.8, c = cmap(index), 
                    marker=markers[index], label=cl)
    
    # 突出测试集的点
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    alpha=0.15,
                    linewidths=2,
                    marker='^',
                    edgecolors='black',
                    facecolors='none',
                    s=55, label='test set')
