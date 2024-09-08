import path_setup
import pickle

import pandas as pd
import numpy as np

from src.configs.common_configs import PathConfig


'''
查看数据
'''

data_folder = PathConfig().checkpoints / 'BA_20vertices' / 'data'
file_name = 'results_BA_20spin_m4_100graphs.pkl'

# 读取.pkl文件
with open(data_folder / file_name, 'rb') as f:
    data = pickle.load(f)
    
# print(data.keys())

# 前五个cds值
# print(data['cds'][:5])

# 对应的解集
# print(data['sol'][:5])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# data._stat_axis.values.tolist()

data = pd.DataFrame(data)

# 显示列名
print(data.columns.values.tolist())

# 显示前5行
# print(data.head())

# dataFrame一行过长显示不完整，转为numpy
# data = data.to_numpy()

# print(data.shape)

# print(data[:5,:2])

print(data.iloc[:5, :2])