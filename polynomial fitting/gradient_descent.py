import numpy as np  # 导入numpy库，此库用于进行矩阵运算等操作
import matplotlib.pyplot as plt  # 导入pyplot绘图库
import time

def sin(x):
    return np.sin(2 * np.pi * x)


def get_data(X_range, X_num, func, noise_variance):
    X = np.linspace(X_range[0], X_range[1], num=X_num)  # 生成对应范围和数量的等距向量
    Y = func(X) + np.random.normal(loc=0, scale=noise_variance, size=X_num)
    # 生成X对应函数的值向量,增加了高斯噪声
    return X, Y


def get_matrix(x, M):  # 获得参数矩阵X
    X = pow(x, 0)
    for i in range(1, M + 1):
        X = np.column_stack((X, pow(x, i)))
    return X


def get_ERMS(W, X, T):  # 计算损失函数
    diff = X @ W - T  # 残差 diff
    ERMS = (diff.T @ diff) / (2 * sample_num)
    return ERMS


def get_gradient(W, X, T):          # 定义梯度函数
    diff = X @ W - T			# 残差 diff
    return (1/sample_num) * (X.T @ diff)


def gradient_descent(X, T):
    start = time.time()
    W = np.zeros(rank+1)  # 初始化W
    iter = 0
    while True:
        iter += 1
        ERMS = get_ERMS(W, X, T)
        gradient = get_gradient(W, X, T)
        W = W - lr * gradient + lam * W
        if np.abs(ERMS) <= thresh or iter == iteration_thresh:
            break
    end = time.time()
    print("Run time : {0}, with iter : {1} and RMS : {2}".format(round((end-start), 2), iter,round(ERMS, 2)))
    errorlist.append(ERMS)
    return W


# 常量赋值
X_range = (0.0, 1.0)
sample_num = 10
funcName = sin
rank = 15
noise_variance = 0.1  # 噪声的方差
lr = 0.1  # 学习率
errorlist = []  # 保存损失函数值
lam = 1e-6  # 正则项系数
iteration_thresh = 4000000  # 最大迭代次数
thresh = 0.003  # 可接受误差

_X, _Y = get_data(X_range, sample_num, funcName, noise_variance)
X = get_matrix(_X, rank)
# 计算解向量W
W = gradient_descent(X, _Y)
# 测试不同学习率
# for lr1 in [0.1,0.01,0.001,0.0001,0.00001]:
#     theta,W=gradient_descent(X,lr1,lam,level,err)
# print('theta:', theta)
# err


# 结果可视化
x = np.linspace(X_range[0], X_range[1], num=1200)  # 真值弧线和拟合弧线
y = sin(x)  # 画出弧线
X2 = get_matrix(x, rank)
value = X2 @ W
plt.plot(x, value, "r-", x, y, _X, _Y, "co")
plt.legend(["$Result$", "$Y=sin(2𝝅X)$","$DataSet$"])
plt.title("learning_rate=" + str(lr))
plt.show()