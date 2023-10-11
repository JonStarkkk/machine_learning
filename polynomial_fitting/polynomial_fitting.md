# 《模式识别与机器学习A》实验报告
#### 实验题目：多项式拟合正弦函数实验
#### 学号：2021110737       
#### 姓名：朱建宇           
## 一．	实验目的：
掌握机器学习训练拟合原理（无惩罚项的损失函数）、掌握加惩罚项（L2范数）的损失函数优化、梯度下降法、理解过拟合、克服过拟合的方法(如加惩罚项、增加样本)
## 二．	实验要求：
1. 生成数据，加入噪声；
2. 用高阶多项式函数拟合曲线（建议正弦函数曲线）；
3. 优化方法求解最优解（梯度下降）；
4. 用你得到的实验数据，解释过拟合；
5. 用不同数据量，不同超参数，不同的多项式阶数，比较实验效果。
## 三．	实验环境
- pycharm 
- anaconda python 3.11
- win10
- X86-64

## 四．理论支撑
由高数中的泰勒级数可知，足够高阶的多项式可以拟合任意函数。因此，我们可以用多项式来拟合正弦函数 $sin(2πx)$ 。在m阶多项式中，有m+1个待定系数。m+1个系数（由低阶到高阶）记为
$w_i$
,组成的（列）向量记作
$w$
。拟合函数可表示为：

$$y({x,w})=\sum_{i=0}^mw_ix^i$$

### 4.1 最小二乘法 
#### 4.1.1理论
最小二乘法的代价函数为：

$E（w） = {\frac{1}{2}}\sum _{n=1}^N{(y(X,w)-t_n)}^2$
求偏导得 
$${\frac{\partial E}{\partial w}}=X'Xw-X'T$$

令 
${\frac{\partial E}{\partial w}}=0$
可得 
$w= (X^TX)^{-1}X^TT$
加入惩罚项（L2范数）时，代价函数为：

$$E（w） = {\frac{1}{2}}\sum _{n=1}^N{(y(X,w)-t_n)}^2+{\frac{\lambda}{2}||w||^2}$$
同理，令偏导 
${\frac{\partial E}{\partial w}}=0$
得 
$w= (X^TX+\lambda)^{-1}X^TT$
其中 
$X=
\left[
\begin{matrix}
 1      & x_1      & \cdots & x_1^m      \\
 1      & x_2      & \cdots & x_2^m      \\
 \vdots & \vdots & \ddots & \vdots \\
 1      & x_N      & \cdots & x_N^m      \\
\end{matrix}
\right],
w=\left[
\begin{matrix}
w_1     \\
w_2      \\
\vdots \\
w_m    \\
\end{matrix}
\right],
T=\left[
\begin{matrix}
t_1     \\
t_2      \\
\vdots \\
t_m    \\
\end{matrix}
\right]$
#### 4.1.2核心算法
首先编写函数get_matrix函数得到
$X$
矩阵
```
def get_matrix(x, M):   #生成(sample_num,m)的参数矩阵
        X = pow(x, 0)
        for i in range(1, M + 1):
            X = np.column_stack((X, pow(x, i)))
        return X
```
根据理论基础中
$w$
的计算公式编写least_square函数求出$w$
```
def least_square(X, T, M):  # 最小二乘法求解向量
    W = np.linalg.inv(X.T @ X + np.identity(M + 1) * lam) @ X.T @ T  # lam=0时无惩罚项
    return W
```


### 4.2 梯度下降法
#### 4.2.1理论
梯度下降法的基本思想可以类比为一个下山的过程。假设这样一个场景：一个人被困在山上，需要从山上下来(找到山的最低点，也就是山谷)。但此时山上的浓雾很大，导致可视度很低；因此，下山的路径就无法确定，必须利用自己周围的信息一步一步地找到下山的路。这个时候，便可利用梯度下降算法来帮助自己下山
在最小二乘法中，我们已经得到偏导数
${\frac{\partial E}{\partial w}}=X'Xw-X'T=X'(Xw-T)$
定义学习率即步长
$\alpha$
,可以得到经过一次迭代后的
$w'=w-\alpha X'(Xw-T)$
,通过设置合适的
$\alpha$
，经过足够多的迭代次数后，我们能够得到一个在可接受误差内的优化解
$w^*$
，当然为了限制时间，我们也需要设置最大迭代次数

#### 4.2.2核心算法
首先通过get_gradient函数计算当前
$w$
对应梯度。其中，可以通过计算残差
$D=Xw-T$
简化计算
```python
def get_gradient(W, X, T):          # 定义梯度函数
    diff = X @ W - T			# 残差 diff
    return (1/sample_num) * (X.T @ diff)
```
编写gradient_descent函数迭代获得新的$w$直到到达最大迭代次数或代价$ERMS$进入最大可接受误差内
```python
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
```
其中$ERMS$通过get_ERMS函数计算
```python
def get_ERMS(W, X, T):  # 计算损失函数
    diff = X @ W - T  # 残差 diff
    ERMS = (diff.T @ diff) / (2 * sample_num)
    return ERMS
```
## 五．	实验结果与分析
### 5.1 最小二乘法
#### 5.1.1 求解最优解

#### 5.2 过拟合
#### 5.3 不同超参数的影响
## 六．	完整实验代码
```python
import numpy as np  # 导入numpy库，此库用于进行矩阵运算等操作
import matplotlib.pyplot as plt  # 导入pyplot绘图库
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False#显示负号

def funcy(X):  #定义初始函数
        return np.sin(2 * np.pi * X)
def getData(X_range, X_num, func, noise_variance):   #生成数据集
        X = np.linspace(X_range[0], X_range[1], num=sample_num)  # 生成对应范围和数量的等距向量
        Y = funcy(X) + np.random.normal(loc=0, scale=noise_variance, size=sample_num)
        # 生成X对应函数的值向量,增加了高斯噪声
        return X, Y
def get_matrix(x, M):   #生成(sample_num,order)的参数矩阵
        X = pow(x, 0)
        for i in range(1, M + 1):
                X = np.column_stack((X, pow(x, i)))
        return X
def LeastSquareM(X, T):  # 最小二乘法求解向量
    s=X.size/sample_num
    regular = np.eye(int(s))
    regular[0][0] = 0
    lam=0.00001
    W = np.linalg.inv((X.T) @ X+lam*regular) @ (X.T) @ (T)  # lam=0时认无惩罚项
    return W
def loss(T, X, W):  #  定义对应损失函数
    ERMS = ((X.dot(W) - T).T.dot((X.dot(W)) - T)) / (2 * sample_num)
    return ERMS
# 常量区
X_range = (0.0, 1.0)
sample_num = 8
funcName = funcy
noise_variance = 0.1  # 噪声的方差
level=20#多项式拟合阶数
# x, y为两个向量，对应的x[k],y[k]则是图像上的一个点\
x = np.linspace(0.0, 1.0, num=sample_num)  # 生成从0.0开始到2.0结束的sample_num个等距向量
y = funcy(x)  # 同样是一个数列，x的sin函数，注意numpy中默认使用弧度制
x2 = np.linspace(0.0, 1.0, num=1200)  #以此画出真值弧线和拟合弧线
y2 = funcy(x2)
errorList = list()  # 记录不同阶数的错误率List
_X, _Y = getData(X_range, sample_num, funcName, noise_variance)
# x 原始数据集的x坐标集合
# X 由x的1到n次方构成的矩阵
# W 求解的目标解向量
for i in range(1, level):
    X = get_matrix(x, i)
    W = LeastSquareM(X, _Y)  # W的计算使用样本的X
    errorList.append(loss(_Y, X, W))  # 计算这次循环的错误率，并加入到 errorList向量中
    X2 = get_matrix(x2, i)  # 画出拟合曲线所需的x坐标参数矩阵
    value = X2 @ W  # 得到拟合的y坐标
    print("Value.shape:", value.shape)
    plt.plot(x2, value.reshape(-1, 1), "r-", x2, y2)
    x1 = np.linspace(0.0, 1.0, num=sample_num)
    plt.plot(x1, _Y, "co", markersize=2.)
    plt.legend(["$Result$", "$Y=sin(2𝝅X)$", "$DataSet$"])
    plt.title("sample_num=" + str(sample_num)+"\norder=" + str(i))
    plt.savefig(r"D:\\code\\python\\machine learning\\data\\" + str(sample_num) + "_" + str(i) + "level.png")
    plt.grid()
    plt.show()

# 绘制不同M下的学习率
x_errorList = np.linspace(1, level, level-1)
plt.plot(x_errorList, errorList)
plt.grid()
plt.show()
```
## 七．	参考文献
    机器学习【周志华】
    统计学习方法【李航】
    数值分析【李庆扬，王能超，易大义】

