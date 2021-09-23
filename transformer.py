# -*- coding: utf-8 -*-
import pandas as pd
import argparse
import numpy as np
import os
import re

target_encoding = 'gb2312'


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


def recent_data(prefix):
    oslist = os.listdir(prefix)
    pattern = r'\d+\-\d+\-\d+ \& (\d+)\-(\d+)\-(\d+)Hold\.txt'
    recent_time = 0
    recent_file = ''
    for filename in oslist:
        match = re.match(pattern, filename)
        if match:
            print("已找到 {}".format(prefix + "/" + filename))
            time = match.groups()
            second = int(time[2]) + int(time[1]) * 60 + int(time[0]) * 3600
            if second > recent_time:
                recent_time = second
                recent_file = filename
    if recent_file:
        print("最新数据文件为 {}".format(prefix + "/" + recent_file))
    return recent_file        


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
    out.append("\t&\t".join(["序号\t&\t沸点 $T$/\si{^\circ C}\t&\t压强 $p$/\si{kPa}"] * column_num) + "\t\\\\")
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
    FLAGS = parseArgs()
    data_path = FLAGS.dataPath
    if FLAGS.dataPath:
        with open(data_path, 'r', encoding=target_encoding) as f:
            lines = f.readlines()
    else:
        prefix = '.'
        print("正在尝试从当前文件夹中寻找最新数据文件")
        data_path = recent_data(prefix)
        if not data_path:
            print("正在尝试从 C:/Users/$Username$/Document/WXI-05_DATA 中寻找数据文件夹")
            import getpass
            prefix = "C:/Users/" + getpass.getuser() + "/文档/WXI-05_DATA"
            try:
                try:
                    os.listdir(prefix)
                except:
                    prefix = "C:/Users/" + getpass.getuser() + "/Documents/WXI-05_DATA"
                data_path = recent_data(prefix)
                if not data_path:
                    raise FileNotFoundError
            except FileNotFoundError:
                print("正在尝试从 C:/Users/$Username$/OneDrive/Document/WXI-05_DATA 中寻找数据文件夹")
                prefix = "C:/Users/" + getpass.getuser() + "/OneDrive/文档/WXI-05_DATA"
                try:
                    try:
                        os.listdir(prefix)
                    except:
                        prefix = "C:/Users/" + getpass.getuser() + "/Documents/WXI-05_DATA"
                    data_path = recent_data(prefix)
                    if not data_path:
                        raise FileNotFoundError
                except FileNotFoundError:
                    print("未找到数据文件，程序退出")
                    exit()
        data_path = prefix + "/" + data_path
    
    with open(data_path, encoding=target_encoding) as f:
        lines = f.readlines()

    index1 = lines[0].find("temperature")
    index2 = lines[0][index1:].find(')')
    start_index = index1 + index2 + 1
    lines[0] = lines[0][start_index:]
    entries = map(lambda x: x.split(), lines)
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
        print("回归直线方程为 log p/p0={}/(T/K)+{}，复相关系数平方为 {}".format(k, b, r2))

        if eval(FLAGS.draw):
            import matplotlib.pyplot as plt
            x = np.linspace(min(t), max(t), 1000)
            y = k * x + b
            plt.scatter(t, p)
            plt.plot(x, y)
            plt.show()

if __name__ == "__main__":
    main()