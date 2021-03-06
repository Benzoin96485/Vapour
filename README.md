# Vapour
北京大学化学与分子工程学院《物理化学实验》课程《液体饱和蒸气压的测定》实验现场数据处理小助手

![Version](https://img.shields.io/badge/Version-1.3-blue.svg)

版本 1.3 更新内容

bug 修复

1. 图中的直线方程位置位于图片之外
2. 命令行中输出的回归方程有多个连续正负号

新增功能

1. 输出回归的必要统计量，以便计算各类不确定度
2. 提供了对数回归选项，用于更精密的分析

<details>
<summary>展开查看更久远的更新日志</summary>

 ---	
	
版本 1.2 更新内容
	
bug 修复

1. 大量琐碎的画图问题
2. 输出的latex表格温度标签和压力标签写反

新增功能
1. 在图中显示直线方程和r<sup>2</sup>
2. 保存图片到指定位置
	
 ---
	
版本 1.1 更新内容
	
新增功能
1. 使得所绘制的直线拟合图符合物化实验报告的要求	
	
---
	
版本 1.0.5 更新内容
	
bug 修复
	
1. 修复了文件末尾空行导致的崩溃
2. 提高了解析数据文件的健壮性

新增功能
1. 支持了用户文档位于D盘的情况

---

版本 1.0 更新内容
	
bug 修复
1. 命令行参数 `--draw=False` 时仍然会弹窗画图
2. 在删去多行时命令行参数 `--startRow=0` 无效果
3. 不添加 `--ignoreRows` 时程序崩溃

新增功能
1. 可以输出 latex 表格代码，并插入到`.tex`文件的指定位置
	
---
</details>


## 当前功能

1. 将温度-压力计采集的 \*Hold.txt 数据转换为容易编辑的csv格式，方便excel、origin等软件打开
2. 给出拟合直线方程和复相关系数平方
3. 展示并保存数据散点和拟合直线图，符合物化实验报告规范
4. 输出 latex 表格代码，可以插入到`.tex`文件中


## 使用方法

### 准备工作

安装 [python](https://www.python.org)（建议3.8以上）和版本适配的 [pandas](https://pandas.pydata.org) （必备），[numpy](https://numpy.org)（拟合需要），[matplotlib](https://matplotlib.org)（画图需要），[scikit-learn](https://scikit-learn.org/stable/index.html)（对数拟合需要）

建议将 python 添加到环境变量，使得在命令行（cmd）中可以通过

```
python *.py 或 python3 *.py
```

来运行一个文件路径为 `./*.py` 的 python 脚本

### 开始使用

**注意事项**：为了保证程序正常解析原始数据，请不要对原始数据作任何改动，包括加回车、空行、空格等。

首先将小助手 `transformer.py` 放在一个你喜欢的位置，最好与采集得到的原始数据位于同一文件夹。然后打开命令行，使得提示符前的文件夹和存放小助手的文件夹一致。例如存放小助手的文件夹为 `E:\coding\py\vapour_pressure`，则可以输入以下命令

```
E:
cd E:\coding\py\vapour_pressure
```

命令行应该有如下字样

```
E:\coding\py\vapour_pressure>
```

运行小助手时通过命令行参数来选择不同功能，最繁冗的命令是

```
python transformer.py --dataPath='' --ignoreRows=[] --csvPath='./data.csv' --csvEncoding='gb2312' --regress=True --draw==True --log=ln --startRow=1 --latex="output" --latexfile='' --columns=1 --figPath=./1.png --logfit=False
```

所有以双横杠 `--` 开头，空格分割的字段都可以选择不写（上面写的都是默认值，不写的话将按照以上参数执行），而如果写了，它们的作用如下：

#### --datapath

指定数据文件的位置，请用引号包住数据文件的绝对路径或相对路径，这一点是为了防止路径中可能出现的空格。

如果你将小助手和原始数据放在同一个文件夹，小助手将寻找最新的数据进行处理。如果没有放在同一文件夹，小助手将尝试从你的用户文件夹的“文档”中寻找原始数据，如果仍未找到，则会停止工作。
所谓文档文件夹，是指
```
C:\Users\$Username$\文档\
C:\Users\$Username$\Documents\
C:\Users\$Username$\Onedrive\文档\
C:\Users\$Username$\Onedrive\Documents\
```
以及在D盘的类似文件夹

#### --ignoreRows

实验过程中可能在开始时为了调试仪器而多记录了几组数据，它们将原封不动地保留在原始数据中。如果你知道有些行并不是实验要测的数据，你可以将它们删除。例如，要删除第一行可以写 `--ignoreRows=1`，要删除前三行可以写 `--ignoreRows=[1,2,3]`，要删除前十行可以写 `--ignoreRows=list(range(1,11))`。总之，等号后面应该是 python 能读懂的数字或列表/元组。

这些删除的行在拟合直线及输出数据表时都不会再出现。

#### --csvPath

小助手会贴心地帮你制造一个适合用来进行数据分析的 `.csv` 表格，你可以在这里指定它的保存路径，如果没有指定，则默认保存在与小助手同一文件夹下的 `data.csv` 中。

#### --csvEncoding

由于Excel在中文环境下默认解读 csv 表格的中文编码是 gb2312，如果保存时用了别的编码可能会看到“锟斤拷”等乱码；如果你有别的需求，可以在这里改为 `--csvEncoding=utf8` 等其他编码

#### --regress

`--regress=True` 开启拟合功能（结果显示在命令行），False 关闭。开启拟合是开启画图和对数拟合功能的前提条件。

#### --draw

`--draw=True` 开启画图功能（结果弹窗显示，请自行保存，未设置规范的图表格式），False 关闭。

#### --log

选择拟合采用的压强对数的形式，`--log=ln` 是自然对数，`--log=log` 是以10为底的对数。无论如何设置，对数拟合部分的温度对数都会选择自然对数。

#### --startRow

与 `--ignoreRows` 配合，如果你认为最上方的一行数据是第 0 行，请在这里添加 `--startRow=0`。

#### --latex

如果传入 `--latex=output`，则将在命令行界面输出一个表格 `tabular` 环境的三线表代码，类似于
```latex
\begin{tabular}{ccc|ccc}
\toprule
序号    &      压强 $p$/\si{kPa}   &       沸点 $T$/\si{^\circ C}       &       序号    &       压强 $p$/\si{kPa}   &       沸点 $T$/\si{^\circ C}      \\
\midrule
1       &       51.28   &       81.97   &       8       &       85.43   &       95.59   \\
2       &       55.44   &       83.99   &       9       &       90.59   &       97.20   \\
3       &       60.43   &       86.41   &       10      &       95.57   &       98.68   \\
4       &       65.41   &       88.42   &       11      &       100.38  &       100.00  \\
5       &       70.35   &       90.42   &       12      &       100.37  &       100.00  \\
6       &       75.33   &       92.20   &       13      &       100.37  &       100.00  \\
7       &       80.60   &       94.01   \\
\bottomrule
\end{tabular}
```
如果传入 `--latex=insert`，则将在指定或搜索到的 `.tex` 文件中插入以上表格代码。你应该准备好一个表格 `table` 环境，加好表头，设置好对其方式，然后在里面添加插入标记，类似于
```latex
\begin{table}[!htbp]
	\centering
	\small
	\caption{\ce{CCl4} 的饱和蒸气压-温度表}
	%insert my table
  	%insert done
\end{table}
```
然后执行代码的预期效果为
```latex
\begin{table}[!htbp]
	\centering
	\small
	\caption{\ce{CCl4} 的饱和蒸气压-温度表}
  %insert my table
  
	\begin{tabular}{ccc|ccc}
		\toprule
		序号    &       压强 $p$/\si{kPa}   &       沸点 $T$/\si{^\circ C}       &       序号    &      压强 $p$/\si{kPa}   &       沸点 $T$/\si{^\circ C}      \\
		\midrule
		1       &       100.51  &       75.22   &       12      &       100.27  &       76.00   \\
		2       &       100.56  &       75.12   &       13      &       95.77   &       74.46   \\
		3       &       100.55  &       75.00   &       14      &       90.72   &       72.79   \\
		4       &       100.57  &       75.18   &       15      &       85.42   &       70.86   \\
		5       &       100.58  &       75.23   &       16      &       80.01   &       68.71   \\
		6       &       100.59  &       75.18   &       17      &       75.56   &       66.90   \\
		7       &       100.24  &       76.01   &       18      &       70.38   &       64.88   \\
		8       &       100.25  &       76.00   &       19      &       65.62   &       62.81   \\
		9       &       100.29  &       76.17   &       20      &       60.48   &       60.45   \\
		10      &       100.28  &       76.07   &       21      &       55.21   &       57.94   \\
		11      &       100.28  &       76.02   &       22      &       50.39   &       55.32   \\
		\bottomrule
	\end{tabular}
  %insert done
\end{table}
```
在开始标记 `%insert my table` 和结束标记 `%insert done` 之间的任何内容都不会被保留。为了安全起见，请在使用小助手自动插入之前备份你的文档。

#### --latexfile
用于指定要插入的 `.tex` 文件的路径。

#### --columns
如果数据过多，单栏表格会过长而不美观。这里可以设置分栏的栏数，如 `--columns=2` 就会输出一个能渲染成两栏三线表的代码，以上示例中出现的就是两栏代码。

#### --figPath
用于指定保存拟合图片的路径

#### --logfit
`--logfit=True` 时补充以下公式的拟合结果：

![](http://latex.codecogs.com/gif.latex?\\ln\frac{p}{p^\ominus}=-\frac{A}{T}+B+C\ln{T/\mathrm{K}})

---

小助手及小助手的作者不为输出结果的任何计算错误、以及程序造成的文件改动负责。若有程序运行问题和计算结果问题，请及时联系作者。

