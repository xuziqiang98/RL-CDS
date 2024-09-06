import path_setup
import numpy as np
from src.utils import draw_graph

import networkx as nx
import matplotlib.pyplot as plt

import random

graph = [[0, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 1, 1, 0, 0, 0, 0],
         [0, 1, 0, 1, 1, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 1, 0, 0, 0],
         [0, 1, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 0]]

matrix = np.array(graph)

n = matrix.shape[0]
# print(n)

# 用networkx库画出graph
# draw_graph(adj_matrix)

# G = nx.from_numpy_array(matrix)
# nx.draw(G, with_labels=True)
# plt.show()

# state = np.zeros(n)

###############
# 初始化一个CDS
###############

# 选择一个顶点设为2
mask = np.zeros(n)
random_vertex = np.random.randint(n)
mask[random_vertex] = 2
                
# 将邻居都设为2，保证CDS中的顶点都是连通的
for idx in range(n):
    if matrix[random_vertex, idx] == 1:
        mask[idx] = 2
                        
state = mask
                
for i in range(n):
    if mask[i] == 2:
        # state[0, i] = 2
        for j in range(n):
            # state[j] == 0这个条件保证了state中已经赋值的顶点不会被再次赋值
            if matrix[i, j] == 1 and state[j] == 0:
                state[j] = 1

# state = np.array([2, 2, 1, 1, 1, 0, 0, 0, 0])
# state = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])

print(f'初始化一个解集：{state}')

# draw_graph(adj_matrix, state)

################################
# 选择一个action
################################

def get_subgraph(vertices: list) -> np.ndarray:
    '''
    返回一个子图，子图的顶点是CDS中的顶点
    '''
    n = len(vertices)
    subgraph_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            subgraph_matrix[i, j] = matrix[int(vertices[i]), int(vertices[j])]
    return subgraph_matrix

cds = [i for i, val in enumerate(state) if val == 2]
print(f'CDS中的顶点：{cds}')

# subgraph_matrix = get_subgraph(cds)

# draw_graph(subgraph_matrix)

# print('以CDS为顶点集的导出子图的邻接矩阵：')

# print(subgraph_matrix)

def is_cut_vertex(vertex):
    '''
    判断一个点是否是割点
    '''
    current_cds = [i for i, val in enumerate(state) if val == 2]
    induced_cds = get_subgraph(current_cds)
    
    # 获取vertex在导出子图中的idx
    new_idx = current_cds.index(vertex)
    
    n = len(induced_cds)
    
    # 创建一个邻接矩阵副本，并删除指定的顶点
    submatrix = np.delete(induced_cds, new_idx, axis=0)
    submatrix = np.delete(submatrix, new_idx, axis=1)
        
    # 创建一个已访问顶点的数组
    visited = [False] * (n - 1)
        
    # 找到第一个未被删除的顶点并进行DFS遍历
    for start in range(n - 1):
        if not visited[start]:
            break
        
    def dfs(v):
        visited[v] = True
        for u in range(n - 1):
            if submatrix[v][u] == 1 and not visited[u]:
                dfs(u)
        
    # 从start开始进行深度优先搜索
    dfs(start)
        
    # 如果有未访问的顶点，说明删除该顶点导致图不连通
    return not all(visited)

# 从cds中随机选择一个顶点判断其是否为割点
random_vertex_from_cds = random.choice(cds)
print(f'顶点{random_vertex_from_cds}是否在CDS导出子图中是割点：{is_cut_vertex(random_vertex_from_cds)}')

# act()函数
mask = [i for i, val in enumerate(state) if val != 0]
x = mask[:]
# x = mask = np.nonzero(state)

print(f'控制自身或者被控制的顶点：{x}')
# print(mask)

for idx in mask:
    if state[idx] == 2 and is_cut_vertex(idx):
        # x = x[x != idx]
        x.remove(idx)
        
print(f'可以被选择的顶点：{x}')

# if x == mask:
#     print('浅拷贝')
# else:
#     print('深拷贝')

# action = x[np.random.randint(0, len(x))].item()
action = random.choice(x)

# action = 1
# action = 7

print(f"选择的action是顶点{action}")

##################################
# 执行action
##################################

# step()函数

# mask = state

if state[action] == 2:
    mask = state.copy()
    mask[action] = 0
    for i in range(n):
        # 邻居i为1，检查其是否与赋为2的顶点相邻
        if matrix[action, i] == 1 and state[i] == 1:
            # 顶点i的所有邻居都没有加入CDS
            if all(state[j] < 2 for j in range(n) if matrix[i,j] == 1):
                mask[i] = 0
    for i in range(n):
        if matrix[action, i] == 1 and mask[i] == 2:
            mask[action] = 1
    new_state = mask
elif state[action] == 1:
    mask = state.copy()
    mask[action] = 2
    for idx in range(n):
        if matrix[action, idx] == 1 and state[idx] == 0:
            mask[idx] = 1
    new_state = mask
else:
    raise ValueError("The selected vertex should be 1 or 2")

print(f'执行action之后的顶点状态：{new_state}')

############################
# 计算CDS
############################

cds_number = np.sum(new_state == 2)

print(f'当前CDS中顶点的个数：{cds_number}')

if state[action] == 1:
    cds_change = 1
else:
    cds_change = -1

print(f'CDS变化了{cds_change}')