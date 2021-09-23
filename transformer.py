# -*- coding: utf-8 -*-
import pandas as pd
import argparse
import numpy as np
import os
import re

target_encoding = 'gb2312'
version = "1.2"

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataPath',
        type=str,
        default='',
        help="原始数据的路径"
    )
    parser.add_argument(
        '--ignoreRows',
        type=str,
        default='[]',
        help="忽略原始数据中的某些行"
    )
    parser.add_argument(
        '--csvPath',
        type=str,
        default='./data.csv',
        help="保存csv文件的路径"
    )
    parser.add_argument(
        '--csvEncoding',
        type=str,
        default='gb2312',
        help="保存csv时的编码格式"
    )
    parser.add_argument(
        '--regress',
        type=str,
        default="True",
        help="是否进行直线拟合"
    )
    parser.add_argument(
        '--latex',
        type=str,
        default="output",
        help="输出latex表格代码"
    )
    parser.add_argument(
        '--latexfile',
        type=str,
        default="",
        help="待插入的latex文件路径"
    )
    parser.add_argument(
        '--columns',
        type=int,
        default=1,
        help="latex表格分栏数"
    )
    parser.add_argument(
        '--draw',
        type=str,
        default="True",
        help="是否画图"
    )
    parser.add_argument(
        '--log',
        type=str,
        default='ln',
        help="使用的对数形式"
    )
    parser.add_argument(
        "--startRow",
        type=int,
        default=1,
        help="行数的起始计数"
    )
    parser.add_argument(
        "--figPath",
        type=str,
        default="./1.png",
        help="保存拟合图片的路径"
    )
    flags, unparsed = parser.parse_known_args()
    return flags


def pop_not_num(s: str):
    numstr = []
    for ch in s:
        if ch in '1234567890.':
            numstr.append(ch)
        else:
            break
    return float(''.join(numstr))


def t_trans(t):
    return -1.0 / (t + 273.15)
    

def p_trans(p, logstr):
    if logstr == 'ln':
        return np.log(p / 100)
    elif logstr == 'log':
        return np.log10(p / 100)

            
def linear_regression(X, Y):
    coeff = np.polyfit(X, Y, 1)
    Xbar = np.average(X)
    Ybar = np.average(Y)
    r2 = (np.sum((X - Xbar) * (Y - Ybar))) ** 2 / (np.sum((X - Xbar) ** 2) * np.sum((Y - Ybar) ** 2))
    return coeff[0], coeff[1], r2


def eval_rows(rows, start):
    evaled = eval(rows)
    if type(evaled) == int:
        return evaled - start
    elif type(evaled) == list:
        return list(map(lambda x:x-start, evaled))


def find_data():
    import getpass
    username = getpass.getuser()
    data_path = ''
    
    prefix_list = ["./",
        "./WXI-05_DATA/", 
        "C:/Users/{}/Documents/WXI-05_DATA/".format(username), 
        "C:/Users/{}/文档/WXI-05_DATA/".format(username), 
        "C:/Users/{}/OneDrive/文档/WXI-05_DATA/".format(username), 
        "C:/Users/{}/OneDrive/Documents/WXI-05_DATA/".format(username),
        "D:/Users/{}/Documents/WXI-05_DATA/".format(username), 
        "D:/Users/{}/文档/WXI-05_DATA/".format(username), 
        "D:/Users/{}/OneDrive/文档/WXI-05_DATA/".format(username), 
        "D:/Users/{}/OneDrive/Documents/WXI-05_DATA/".format(username),
    ]

    for prefix in prefix_list:
        try:
            print("正在尝试从 {} 中寻找数据文件夹".format(prefix))
            os.listdir(prefix)
            data_path = recent_data(prefix)
        except FileNotFoundError:
            continue
        else:
            break

    if data_path:
        return prefix + data_path
    else:
        print("未找到数据文件，程序退出")
        exit()


def recent_data(prefix):
    oslist = os.listdir(prefix)
    pattern = r'\d+\-\d+\-\d+ \& (\d+)\-(\d+)\-(\d+)Hold\.txt'
    recent_time = 0
    recent_file = ''
    for filename in oslist:
        match = re.match(pattern, filename)
        if match:
            print("已找到 {}".format(prefix + filename))
            time = match.groups()
            second = int(time[2]) + int(time[1]) * 60 + int(time[0]) * 3600
            if second > recent_time:
                recent_time = second
                recent_file = filename
    if recent_file:
        print("最新数据文件为 {}".format(prefix + recent_file))
        return recent_file 
    else:
        raise FileNotFoundError


def latexize(df: pd.DataFrame, column_num=1):
    df_reset_index = df.reset_index(drop=True)
    row_num = (df_reset_index.shape[0] + column_num - 1) // column_num
    tabular = [""] * row_num
    decimal2 = lambda x: "{:.2f}".format(x)
    for index, row in df_reset_index.iterrows():
        if tabular[index % row_num]:
            tabular[index % row_num] += "\t&\t" + str(index+1) + "\t&\t" + "\t&\t".join(list(row.apply(decimal2)))
        else:
            tabular[index % row_num] += str(index+1) + "\t&\t" + "\t&\t".join(list(row.apply(decimal2)))
    out = []
    out.append(r"\begin{tabular}{" + "|".join(["ccc"] * column_num) + "}")
    out.append(r"\toprule")
    out.append("\t&\t".join(["序号\t&\t压强 $p$/\si{kPa}\t&\t沸点 $T$/\si{^\circ C}"] * column_num) + "\t\\\\")
    out.append(r"\midrule")
    for line in tabular:
        if line:
            out.append(line + "\t\\\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    return out


def latex_insert(tablelist, path=''):
    if path:
        try:
            file = open(path, 'r', encoding='utf8')
        except FileNotFoundError:
            print("未找到指定的latex文件")
            return
    else:
        filelist = os.listdir(".")
        for filename in filelist:
            if ".tex" in filename:
                print("已找到 {}".format(filename))
                path = "./" + filename
                file = open(path, 'r', encoding='utf8')
                break
        if not file:
            print("在当前目录下未找到任何latex文件")
            return
    content = file.read()
    key_start = r"%insert my table"
    key_stop = r"%insert done"
    
    index_start = content.find(key_start)
    index_stop = content.find(key_stop)
    if index_start >= 0:
        if index_stop >= 0:
            content = content[:index_start + len(key_start)] + "\n\n" + "\n".join(tablelist) + "\n\n" + content[index_stop:]
        else:
            content = content[:index_start + len(key_start)] + "\n" + "\n".join(tablelist) + "\n\n" + content[index_start + len(key_start):]
    else:
        print("未找到插入标记，默认在文件尾插入")
        content = content + "\n" + "\n".join(tablelist)
    file = open(path, 'w', encoding='utf8')
    file.write(content)
    file.close()


def main():
    print("\n当前版本号为 {}，请到 https://github.com/Benzoin96485/Vapour 检查是否为最新版。".format(version))
    
    print("\n作者不对输出结果的正确性和有效性，以及其连带的后果负责。\n\n")
    print("-" * 100)
    FLAGS = parseArgs()
    if FLAGS.dataPath:
        data_path = FLAGS.dataPath
    else:
        data_path = find_data()
    
    with open(data_path, 'r', encoding=target_encoding) as f:
        lines = f.readlines()

    entries = []
    data_pattern = r"\D*(\d+\:\d+\:\d+\.\d+)\s+(\d+\.\d+)KPa\s+(\d+\.\d+).*"
    for line in lines:
        match = re.match(data_pattern, line)
        if match:
            entries.append(match.groups())
    
    columns = list(zip(*entries))
    
    df = pd.DataFrame()
    df["time"] = list(columns[0])
    df["pressure(kPa)"] = list(map(lambda x: pop_not_num(x), columns[1]))
    df["temperature(°C)"] = list(map(lambda x: pop_not_num(x), columns[2]))

    df.drop(index=df.index[eval_rows(FLAGS.ignoreRows, FLAGS.startRow)], inplace=True)
    table = latexize(df[["pressure(kPa)", "temperature(°C)"]], FLAGS.columns)
    if FLAGS.latex == "output":
        print("以下是latex表格部分\n" + "-" * 100 + "\n\n")
        for line in table:
            print(line)
        print("\n\n" + "-" * 100 + "\nlatex表格部分结束")
    elif FLAGS.latex == "insert":
        texfile = FLAGS.latexfile
        latex_insert(table, texfile)
    df.to_csv(FLAGS.csvPath, encoding=FLAGS.csvEncoding)
    
    if eval(FLAGS.regress):
        p = p_trans(df["pressure(kPa)"], logstr=FLAGS.log)
        t = t_trans(df["temperature(°C)"])
        k, b, r2 = linear_regression(t, p)
        print("回归直线方程为 {} p/p0=-{}/(T/K)+{}，复相关系数平方为 {}".format(FLAGS.log, k, b, r2))

        if eval(FLAGS.draw):
            import matplotlib
            import matplotlib.pyplot as plt
            x = np.linspace(min(t), max(t), 1000)
            y = k * x + b

            font = {'family' : 'Arial',
                    'weight' : 'medium',
                    'size' : 10,
                    'style' : 'normal'}
            plt.rcParams['mathtext.fontset'] = 'custom'
            plt.rc('font', **font)
            #plt.rc('font2', **font2)
            plt.rcParams['mathtext.rm'] = 'Arial:normal'
            plt.rcParams['mathtext.it'] = 'Arial:italic'
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            #print(plt.rcParams)

            plt.figure(1)
            ax=plt.gca()
            
            ax.text(-0.0028,-0.1,r"$\ln\dfrac{p}{p^\ominus} = -\dfrac{" + "{:.6}".format(k) + r"}{T/\mathrm{K}}+" + "{:.6}".format(b) + r",\quad r^2=" + "{:.5}".format(r2) + r"$" ,fontsize=12)

            ax.spines['bottom'].set_linewidth(1)
            ax.spines['left'].set_linewidth(1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.xlabel(r'$-(1/T)\,/\, (\mathrm{K}^{-1})$', size=12)
            plt.ylabel(r'$\ln(p/p^\ominus)$', size=12)
            plt.scatter(t, p)
            plt.plot(x, y, linewidth=2)
            plt.savefig(FLAGS.figPath)
            plt.show()
            

if __name__ == "__main__":
    main()