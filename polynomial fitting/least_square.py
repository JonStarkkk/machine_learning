import numpy as np  # 导入numpy库，此库用于进行矩阵运算等操作
import matplotlib.pyplot as plt  # 导入pyplot绘图库


def sin(x):  # 定义初始函数
    return np.sin(2 * np.pi * x)


def get_data(x_range, x_num, func, noise_variance):   # 生成数据集
    x = np.linspace(x_range[0], x_range[1], num=x_num)  # 生成对应范围和数量的等距向量
    y = func(x) + np.random.normal(loc=0, scale=noise_variance, size=x_num)
    # 生成X对应函数的值向量,增加了高斯噪声
    return x, y


def get_matrix(x, M):   #生成(sample_num,m)的参数矩阵
        X = pow(x, 0)
        for i in range(1, M + 1):
            X = np.column_stack((X, pow(x, i)))
        return X


def get_ERMS(W, X, T):  #  定义对应损失函数
    diff = X @ W - T  # 残差 diff
    ERMS = (diff.T @ diff) / (2 * sample_num)
    return ERMS


def least_square(X, T, M):  # 最小二乘法求解向量
    W = np.linalg.inv(X.T @ X + np.identity(M + 1) * lam) @ X.T @ T  # lam=0时无惩罚项
    return W


# 常量区
X_range = (0.0, 1.0)
sample_num = 10
funcName = sin
noise_variance = 0.1  # 噪声的方差
rank = 15  # 多项式拟合阶数
lam = 0.00002  # 正则化系数

# x, y为两个向量，对应的x[k],y[k]则是图像上的一个点
x = np.linspace(X_range[0], X_range[1], num=1200)  # 真值弧线和拟合弧线
y = sin(x)
errorList = list()  # 记录不同阶数的错误率List
_X, _Y = get_data(X_range, sample_num, funcName, noise_variance)
for i in range(1, rank):
# for i in range(m-1, m):
    X = get_matrix(_X, i)
    W = least_square(X, _Y,i)
    # W的计算使用样本的X
    errorList.append(get_ERMS(W, X, _Y))  # 计算这次循环的错误率，并加入到 errorList向量中
    # 可视化
    X2 = get_matrix(x, i)  # 画出拟合曲线所需的x坐标参数矩阵
    value = X2 @ W  # 得到拟合的y坐标
    plt.plot(x, value, "r-", x, y, _X, _Y, "co", markersize=2.)
    plt.legend(["$Result$", "$Y=sin(2𝝅X)$", "$DataSet$"])
    plt.title("sample_num=" + str(sample_num)+"\nrank=" + str(i))
    plt.savefig(r"D:\\code\\python\\machine learning\\polynomial fitting\\result\\" + str(sample_num) + "_" + str(i) + ".png")
    plt.grid()
    plt.close()

# 绘制ERMS图像
x_errorList = np.linspace(1, rank, rank-1)
plt.plot(x_errorList, errorList)
plt.title("lam=" + str(lam))
plt.grid()
plt.show()
