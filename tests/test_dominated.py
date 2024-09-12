import path_setup

# from test_cds import matrix

graph = [[0, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 1, 1, 1, 0, 0, 0, 0],
         [0, 1, 0, 1, 1, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 1, 0, 0, 0],
         [0, 1, 1, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 0, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 0]]

def is_dominated(graph, subset: list) -> bool:
        n = len(graph[0])
        
        for idx in range(n):
            if idx not in subset:
                if all(graph[idx][i] == 0 for i in subset):
                    return False
        
        return True

# sol = [1, 2, 2, 2, 1, 2, 1, 0, 0]
sol = [1, 2, 1, 1, 2, 1, 2, 1, 1]
cds = [idx for idx, val in enumerate(sol) if val == 2]

print(f'图是否被控制：{is_dominated(graph, cds)}')