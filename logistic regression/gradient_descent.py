import numpy as np
import matplotlib.pyplot as plt

'''
获取二分类问题数据集，两类数据均服从正态分布
- N 数据集大小，默认为100
- pi 正例的比例，默认为0.3
- mean 两类样本的均值，为2 * d 维矩阵，d为正态分布维数
- 例: [[1, 1], [3, 3]]表示正例样本均值为(1, 1), 反例样本为(3, 3)
- cov 协方差矩阵，为d * d维矩阵
- 例: [[1, 0], [0, 1]]
- cov2 用于测试不满足Logistic回归假设(不是朴素贝叶斯假设)时结果。若不传入该参数，两类样本协方差相同；否则cov2代表第二类样本的协方差。
'''


def get_data(mean, cov, n=100, pi=0.3, cov2=None):
    mean = np.array(mean, dtype='float')
    cov = np.array(cov, dtype='float')
    assert mean.shape[0] == 2 and mean.shape[1] == cov.shape[0] and cov.shape[0] == cov.shape[1], '参数不合法!'
    positive = int(n *pi)
    negative = n - positive
    pdata = np.random.multivariate_normal(mean[0], cov, positive)
    ndata = np.random.multivariate_normal(mean[1], cov if cov2 is None else cov2, negative)
    return np.concatenate([pdata, ndata]), np.concatenate([np.ones(positive), np.zeros(negative)])

'''
梯度下降法优化损失函数。
- data 数据集
- target 数据的标签，与数据集一一对应
- max_iteration 最大迭代次数，默认为10000
- lr 学习率，默认为0.05
'''
def gradient_descent(data, target, l, max_iteration = 10000, lr = 0.05):
    epsilon = 1e-8
    N, d = data.shape
    beta = np.random.randn(d + 1)
    X = np.concatenate([np.ones((N, 1)), data], axis = 1)

    for i in range(max_iteration):
        term1 = -np.sum(X * target.reshape(N, 1), axis = 0)
        term2 = np.sum((1 / (np.e ** -np.dot(beta, X.T) + 1)).reshape(N, 1) * X, axis = 0)
        regularize = l * beta
        grad = term1 + term2 + regularize
        if np.linalg.norm(grad) < epsilon:
            break
        beta -= lr * grad
    return beta

# 根据模型beta预测x所属的类, 返回1(正例), 或0(反例)。
def predict(beta, x):
    return 1 if np.dot(beta[1:], x) + beta[0] > 0 else 0
# 评估整个数据集上的效果
def evaluate(beta, data, target):
    cnt = 0
    for x, y in zip(data, target):
        if predict(beta, x) == int(y):
            cnt += 1
    print(f'模型参数: {beta}')
    print(f'模型复杂度(模长): {np.linalg.norm(beta)}')
    print(f'预测准确率: {cnt / len(target)}')



if __name__ == '__main__':
    mean = [[1, 1], [4, 4]]
    cov = [[1.3, 1], [1, 0.8]]
    #cov = np.diag([1, 1])

    data, target = get_data(mean, cov)
    l = 0
    beta = gradient_descent(data, target, l, max_iteration=100000,lr = 0.05)
    l = 0.02
    beta2 = gradient_descent(data, target, l, max_iteration=100000, lr=0.05)
    for (x, y), label in zip(data, target):
        plt.scatter(x, y, c='red' if label else 'black')

    x = np.linspace(-1, 4,1200)
    '''
    由于beta(d=2)形如(beta[0], beta[1], beta[2]),
    在二维空间中表示分界面为w^T*x+b=0,当x=(1, x, y)时(这里x, y指横纵坐标)
    分界面为beta[0]+beta[1]x+beta[2]y=0,或写成标准一次函数形式
    y=(-beta[1]x -beta[0]) / beta[2].
    '''
    y = (-beta[1] * x - beta[0]) / beta[2]

    plt.plot(x, y)
    plt.show()
    evaluate(beta, data, target)
    evaluate(beta2, data, target)
