import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # 将label==0的实例赋值为-1
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    return data[:, :-1], data[:, -1]


def SVM():
    x, y = create_data()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    model = SVC(kernel="linear")
    model.fit(x_train, y_train)

    precise = model.score(x_test, y_test)
    print(precise)
    # 画出散点图
    plt.scatter(x[:50, 0], x[:50, 1], label='0')
    plt.scatter(x[50:100, 0], x[50:100, 1], label='1')
    # 查看权重矩阵
    weight = model.coef_[0]
    #print(model.coef_)
    # 取出截距
    bias = model.intercept_[0]
    k = -weight[0] / weight[1]
    b = -bias / weight[1]
    # 获取支持向量
    support_vector = model.support_vectors_
    # 画出支持向量
    for i in support_vector:
        plt.scatter(i[0], i[1], marker=',', c='b')

    # 画出超平面
    xx = np.linspace(4, 7, 10)
    yy = k * xx + b
    plt.plot(xx, yy)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    SVM()