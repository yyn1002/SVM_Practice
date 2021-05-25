import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt


# 鸢尾花(iris)数据集
# 数据集内包含 3 类共 150 条记录，每类各 50 个数据，
# 每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，
# 可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
# 这里只取前100条记录，两项特征，两个类别。
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    # print(data)
    return data[:, :2], data[:, -1]


def showData(line=None):
    dataArr, yArr = create_data()
    data_class_1_index = np.where(yArr == -1)
    data_class_1 = dataArr[data_class_1_index, :].reshape(-1, 2)

    data_class_2_index = np.where(yArr == 1)
    data_class_2 = dataArr[data_class_2_index, :].reshape(-1, 2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(data_class_1[:, 0], data_class_1[:, 1], c='r', label="$-1$")
    ax.scatter(data_class_2[:, 0], data_class_2[:, 1], c='g', label="$+1$")
    plt.legend()
    if line is not None:
        b, alphas = line
        x = np.linspace(1, 8, 50)
        w = np.sum(alphas * yArr[:, np.newaxis] * dataArr, axis=0)
        y = np.array([(-b - w[0] * x[i]) / w[1] for i in range(50)])
        y1 = np.array([(1 - b - w[0] * x[i]) / w[1] for i in range(50)])
        y2 = np.array([(-1 - b - w[0] * x[i]) / w[1] for i in range(50)])
        ax.plot(x, y, 'b-')
        ax.plot(x, y1, 'b--')
        ax.plot(x, y2, 'b--')
    plt.show()


def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


# 对 alpha2 的修正函数
def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


# 建立算法
def smo(dataArr, yArr, C, toler, maxIter):
    """smo
    Args:
        dataArr    特征集合
        yArr       类别标签
        C   松弛变量(常量值)，允许有些数据点可以处于分隔面的错误一侧。
            控制最大化间隔和保证大部分的函数间隔小于 1.0。
            可以通过调节该参数达到不同的结果。
        toler   容错率（是指在某个体系中能减小一些因素或选择对某个系统产生不稳定的概率。）
        maxIter 退出前最大的循环次数（alpha 不发生变化时迭代的次数）
    Returns:
        b       模型的常量值
        alphas  拉格朗日乘子
    """
    numSample, numDim = dataArr.shape  # 100 和 2

    # 初始化参数值 b 和系数 alphas
    b = 0
    alphas = np.zeros((numSample, 1))

    iterations = 0  # 记录迭代次数
    # 只有在所有数据集上遍历 maxIter 次，且不再发生任何 alpha 修改之后，才退出 while 循环
    while iterations < maxIter:
        """
        设置一个参数 alphaPairsChanged 记录 alpha 是否已经进行优化，每次循环开始
        记为 0，然后对整个集合顺序遍历, 如果没变化，则记为迭代一次 
        """
        alphaPairsChanged = 0
        for i in range(numSample):
            #  alpha1选取，按顺序遍历，选不满足KKT条件的
            # 首先对第 i 个样本预测类别
            fXi = np.sum(alphas * yArr[:, np.newaxis] * dataArr * dataArr[i, :]) + b
            # 计算 Ei，也相当于误差
            Ei = fXi - yArr[i]
            """
            #约束条件：KKT 条件
                yi*ui >= 1 and alpha = 0   正常分类
                yi*ui == 1 and 0 < alpha < C 边界上面
                yi*ui < 1 and alpha = C   边界之间

            # yArr[i]*Ei = yArr[i]*（fXi - 1），toler为容错率。
              需要优化的情况为：如果 (yArr[i]*Ei < -toler)，此时 alpha 应该为 C ,但是其值小于C，那就需要优化，
              同理如果 (yArr[i]*Ei > toler)，此时 alpha 应该为 0 ,但是其值却大于 0，也需要优化。
            """
            if (((yArr[i] * Ei < -toler) and (alphas[i] < C)) or
                    ((yArr[i] * Ei > toler) and (alphas[i] > 0))):  # 选取alpha1不满足KKT条件

                # 选取alpha2，随机选取非 i 的一个点，判断是否满足KKT条件
                j = selectJrand(i, numSample)

                # 预测样本 j 的结果
                fXj = np.sum(alphas * yArr[:, np.newaxis] * dataArr * dataArr[j, :]) + b
                Ej = fXj - yArr[j]

                # 如果满足KKT条件，则不需优化，continue
                if (((yArr[j] * Ej < -toler) and (alphas[j] >= C)) or
                        ((yArr[j] * Ej > toler) and (alphas[j] <= 0))):
                    continue

                # 更新 alpha 前先复制，作为 old
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                # 计算 L 和 H
                if yArr[i] != yArr[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                # 计算 eta，eta 是 alphas[j] 的最优修改量，如果 eta == 0，需要退出
                eta = np.sum(dataArr[i, :] * dataArr[i, :]) + \
                      np.sum(dataArr[j, :] * dataArr[j, :]) - \
                      2. * np.sum(dataArr[i, :] * dataArr[j, :])
                if eta <= 0:
                    print("eta <= 0")
                    continue
                # 计算出新的 alpha2 值
                alphas[j] = alphaJold + yArr[j] * (Ei - Ej) / eta
                # 对 alpha2 进行修正
                alphas[j] = clipAlpha(alphas[j], H, L)

                # 检查alpha2是否只是轻微的改变，如果是的话，就继续选取alpha2
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j is not moving enough")
                    continue
                # 更新alpha1
                alphas[i] = alphaIold + yArr[i] * yArr[j] * (alphaJold - alphas[j])

                # 计算参数 b
                bi = b - Ei - yArr[i] * (alphas[i] - alphaIold) * np.sum(dataArr[i, :] * dataArr[i, :]) - \
                     yArr[j] * (alphas[j] - alphaJold) * np.sum(dataArr[i, :] * dataArr[j, :])
                bj = b - Ej - yArr[i] * (alphas[i] - alphaIold) * np.sum(dataArr[i, :] * dataArr[j, :]) - \
                     yArr[j] * (alphas[j] - alphaJold) * np.sum(dataArr[j, :] * dataArr[j, :])
                # b 的更新条件
                if 0 < alphas[i] < C:
                    b = bi
                elif 0 < alphas[j] < C:
                    b = bj
                else:
                    b = (bi + bj) / 2.
                # 到了这一步，说明 alpha , b 被更新了
                alphaPairsChanged += 1
                # 输出迭代信息
                print("iter: %d, i: %d, pairs changed %d" % (iterations, i, alphaPairsChanged))
        # 在 for 循环之外，检查 alpha 值是否做了更新，如果在更新将 iterations 设为
        # 0 后继续运行程序
        # 知道更新完毕后，iterations 次循环无变化，则退出循环
        if alphaPairsChanged == 0:
            iterations += 1
        else:
            iterations = 0
        # print("iteration number: %d" % iterations)

    return b, alphas


def SVM():
    dataArr, yArr = create_data()
    C = 0.6
    toler = 0.001
    maxIter = 40
    b, alphas = smo(dataArr, yArr, C, toler, maxIter)
    return b, alphas


if __name__ == "__main__":
    b, alphas = SVM()
    print(alphas)

    showData(line=(b, alphas))
