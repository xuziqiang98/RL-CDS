import path_setup
import numpy as np

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

print(f'初始化一个解集：{state}')