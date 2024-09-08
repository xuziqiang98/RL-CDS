from abc import ABC, abstractmethod
from collections import namedtuple
from operator import matmul

import warnings
import numpy as np
import torch.multiprocessing as mp
from numba import jit, float64, int64, NumbaPerformanceWarning

from src.envs.utils import (EdgeType,
                            RewardSignal,
                            ExtraAction,
                            OptimisationTarget,
                            Observable,
                            VertexBasis,
                            DEFAULT_OBSERVABLES,
                            GraphGenerator,
                            RandomGraphGenerator,
                            HistoryBuffer)

# A container for get_result function below. Works just like tuple, but prettier.
ActionResult = namedtuple("action_result", ("snapshot","observation","reward","is_done","info"))

# 忽略 NumbaPerformanceWarning 警告
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)

class VertexSystemFactory(object):
    '''
    Factory class for returning new VertexSystem.
    '''

    @staticmethod
    def get(graph_generator=None,
            max_steps=20, # episode length
            observables = DEFAULT_OBSERVABLES,
            reward_signal = RewardSignal.DENSE,
            extra_action = ExtraAction.PASS,
            optimisation_target = OptimisationTarget.ENERGY,
            vertex_basis = VertexBasis.SIGNED,
            # vertex_basis = VertexBasis.TRINARY,
            norm_rewards=False,
            memory_length=None,  # None means an infinite memory.
            horizon_length=None, # None means an infinite horizon.
            stag_punishment=None, # None means no punishment for re-visiting states.
            basin_reward=None, # None means no reward for reaching a local minima.
            reversible_vertices=True, # Whether the vertices can be flipped more than once (i.e. True-->Georgian MDP).
            init_snap=None,
            seed=None):

        if graph_generator.biased:
            return VertexSystemBiased(graph_generator,max_steps,
                                    observables,reward_signal,extra_action,optimisation_target,vertex_basis,
                                    norm_rewards,memory_length,horizon_length,stag_punishment,basin_reward,
                                    reversible_vertices,
                                    init_snap,seed)
        else:
            return VertexSystemUnbiased(graph_generator,max_steps,
                                      observables,reward_signal,extra_action,optimisation_target,vertex_basis,
                                      norm_rewards,memory_length,horizon_length,stag_punishment,basin_reward,
                                      reversible_vertices,
                                      init_snap,seed)

class VertexSystemBase(ABC):
    '''
    VertexSystemBase implements the functionality of a VertexSystem that is common to both
    biased and unbiased systems.  Methods that require significant enough changes between
    these two case to not readily be served by an 'if' statement are left abstract, to be
    implemented by a specialised subclass.
    '''

    # Note these are defined at the class level of VertexSystem to ensure that VertexSystem
    # can be pickled.
    class action_space(): # 动作空间
        '''
        动作空间
        '''
        def __init__(self, n_actions):
            '''
            n是动作数，这里选一个点就是一个动作
            所以动作数就是顶点数
            actions是从0到n-1的一个向量
            '''
            self.n = n_actions
            self.actions = np.arange(self.n)

        def sample(self, n=1):
            '''
            从actions中随机选择n个动作
            也就是说随机选择n个点
            '''
            return np.random.choice(self.actions, n)

    class observation_space(): # 状态空间
        '''
        状态空间
        '''
        def __init__(self, n_vertices, n_observables):
            self.shape = [n_vertices, n_observables] # 顶点数x状态数

    def __init__(self,
                 graph_generator=None,
                 max_steps=20,
                 observables=DEFAULT_OBSERVABLES,
                 reward_signal = RewardSignal.DENSE,
                 extra_action = ExtraAction.PASS,
                 optimisation_target=OptimisationTarget.ENERGY,
                 vertex_basis=VertexBasis.SIGNED,
                 norm_rewards=False,
                 memory_length=None,  # None means an infinite memory.
                 horizon_length=None,  # None means an infinite horizon.
                 stag_punishment=None,
                 basin_reward=None,
                 reversible_vertices=False,
                 init_snap=None,
                 seed=None):
        '''
        Init method.

        Args:
            graph_generator: A GraphGenerator (or subclass thereof) object.
            max_steps: Maximum number of steps before termination.
            reward_signal: RewardSignal enum determining how and when rewards are returned.
            extra_action: ExtraAction enum determining if and what additional action is allowed,
                          beyond simply flipping vertices.
            init_snap: Optional snapshot to load vertex system into pre-configured state for MCTS.
            seed: Optional random seed.
        '''

        if seed != None:
            np.random.seed(seed)

        # Ensure first observable is the vertex state.
        # This allows us to access the vertices as self.state[0,:self.n_vertices.]
        assert observables[0] == Observable.VERTEX_STATE, "First observable must be Observation.VERTEX_STATE."

        self.observables = list(enumerate(observables))

        self.extra_action = extra_action

        if graph_generator!=None:
            assert isinstance(graph_generator,GraphGenerator), "graph_generator must be a GraphGenerator implementation."
            self.gg = graph_generator
        else:
            # provide a default graph generator if one is not passed
            self.gg = RandomGraphGenerator(n_vertices=20,
                                           edge_type=EdgeType.DISCRETE,
                                           biased=False,
                                           extra_action=(extra_action!=extra_action.NONE))

        self.n_vertices = self.gg.n_vertices  # Total number of vertices in episode
        # max_steps设为了顶点数的两倍
        self.max_steps = max_steps  # Number of actions before reset

        self.reward_signal = reward_signal
        self.norm_rewards = norm_rewards

        self.n_actions = self.n_vertices # 动作数就是顶点数
        if extra_action != ExtraAction.NONE:
            self.n_actions+=1

        self.action_space = self.action_space(self.n_actions) # 这里action_space是一个类
        self.observation_space = self.observation_space(self.n_vertices, len(self.observables))

        self.current_step = 0

        if self.gg.biased:
            self.matrix, self.bias = self.gg.get()
        else:
            self.matrix = self.gg.get()
            self.bias = None

        self.optimisation_target = optimisation_target
        self.vertex_basis = vertex_basis

        self.memory_length = memory_length
        # 这里horizon_length就是max_steps
        self.horizon_length = horizon_length if horizon_length is not None else self.max_steps
        self.stag_punishment = stag_punishment
        self.basin_reward = basin_reward
        self.reversible_vertices = reversible_vertices

        self.reset()

        self.score = self.calculate_score()
        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        self.best_score = self.score
        self.best_vertices = self.state[0,:]

        if init_snap != None:
            self.load_snapshot(init_snap)

    def reset(self, vertices=None):
        '''
        matrix是graph的邻接矩阵，ExtraAtion=None时没有其他操作
        max_local_reward_available是选择一个顶点后能立刻获得的最大的奖励
        state矩阵的行表示状态，列表示顶点
        state[0]和state[5]会被更新
        score是此时的cds值
        _reset_graph_observables调用后matrix_obs就是graph的matrix
        reset时会新建一个HistoryBuffer
        返回值是state矩阵和matrix_obs矩阵垂直叠加后形成的新矩阵
        注意这个新矩阵的前7行中元素的取值从这里开始就跟BINARY或者SIGNED一致了
        '''
        self.current_step = 0
        if self.gg.biased: # gg是graph_generator
            # self.matrix, self.bias = self.gg.get(with_padding=(self.extra_action != ExtraAction.NONE))
            self.matrix, self.bias = self.gg.get()
        else: # 主要看gg.based = False
            # self.matrix = self.gg.get(with_padding=(self.extra_action != ExtraAction.NONE))
            self.matrix = self.gg.get() # 拿到graph的matrix
        self._reset_graph_observables() # 重置矩阵的状态，由于这里ExtraActin是NONE，其实什么操作也没做

        verticesOne = np.array([1] * self.n_vertices) # 长为n个顶点的全1向量
        local_rewards_available = self.get_immeditate_rewards_avaialable(verticesOne) # 返回一个向量
        local_rewards_available = local_rewards_available[np.nonzero(local_rewards_available)] # 筛出非0值
        if local_rewards_available.size == 0:
            # We've generated an empty graph, this is pointless, try again.
            self.reset()
        else:
            self.max_local_reward_available = np.max(local_rewards_available) # 返回局部能获得的最大的值

        self.state = self._reset_state(vertices) # 这里vertices=None，第0行和第5行的值都被更新了
        self.score = self.calculate_score() # vertexe=None

        if self.reward_signal == RewardSignal.SINGLE:
            self.init_score = self.score

        self.best_score = self.score
        self.best_obs_score = self.score
        self.best_vertices = self.state[0, :self.n_vertices].copy()
        self.best_obs_vertices = self.state[0, :self.n_vertices].copy()

        if self.memory_length is not None: # 这里传进来的memory_length=None
            self.score_memory = np.array([self.best_score] * self.memory_length)
            self.vertices_memory = np.array([self.best_vertices] * self.memory_length)
            self.idx_memory = 1

        self._reset_graph_observables()

        if self.stag_punishment is not None or self.basin_reward is not None:
            self.history_buffer = HistoryBuffer()

        return self.get_observation()

    def _reset_graph_observables(self):
        '''
        extra_action=NONE时matrix_obs就是matrix
        '''
        # Reset observed adjacency matrix
        if self.extra_action != self.extra_action.NONE:
            # Pad adjacency matrix for disconnected extra-action vertices of value 0.
            self.matrix_obs = np.zeros((self.matrix.shape[0] + 1, self.matrix.shape[0] + 1))
            self.matrix_obs [:-1, :-1] = self.matrix
        else: # 我们初始化的时候是NONE
            self.matrix_obs = self.matrix # matrix_obs就是graph的matrix

        # Reset observed bias vector,
        if self.gg.biased:
            if self.extra_action != self.extra_action.NONE:
                # Pad bias for disconnected extra-action vertices of value 0.
                self.bias_obs = np.concatenate((self.bias, [0]))
            else:
                self.bias_obs = self.bias

    def _reset_state(self, vertices=None):
        '''
        设S是V的一个子集
        x是顶点
        x在S中，x = 1，否则x = -1
        state[0,:]是Observable.VERTEX_STATE
        state[5,:]是Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE
        这个函数会重置state[0,:]，即初始化一个解
        此时如果reversible_vertices=True顶点x的取值是-1或者1，否则全为1
        immeditate_rewards_avaialable是一个向量，第i个元素表示选择第i个顶点后立即获得的reward
        计算其中总共有多少个正值，然后得到正值个数在占总体的一个比值
        将这个比值值赋给state[5,:]
        返回state是一个7xn的矩阵，state[0,:]的取值是1或者-1
        '''
        # state是个全0的矩阵，形状是状态数x动作数（顶点数）
        # 也就是有7行，每一行表示一个状态
        # 有n列，每一列都是一个顶点，表示该顶点执行的动作
        state = np.zeros((self.observation_space.shape[1], self.n_actions))

        if vertices is None:
            if self.reversible_vertices: # reversible_vertices设为True
                # For reversible vertices, initialise randomly to {+1,-1}.
                # 第一行表示顶点的状态，这里相当于初始化了一个随机的解
                # 第一行的元素随机设为-1或者1
                # state[0, :self.n_vertices] = 2 * np.random.randint(2, size=self.n_vertices) - 1 
                '''
                初始化一个解
                顶点可以被重复选择
                此时顶点的取值是0，1，2
                随机选择一个点，将其以及邻居设为2，表示这些点加入了CDS中
                与这些点相邻的点取值设为1，表示这些点被控制了
                剩下的点取值设为0，表示这些点还没有被控制
                self.matrix是图的邻接矩阵
                '''
                # 选择一个顶点设为2
                mask = np.zeros(self.n_vertices)
                random_vertex = np.random.randint(self.n_vertices)
                mask[random_vertex] = 2
                
                # 将邻居都设为2，保证CDS中的顶点都是连通的
                for idx in range(self.n_vertices):
                    if self.matrix[random_vertex, idx] == 1:
                        mask[idx] = 2
                        
                state[0, :self.n_vertices] = mask
                
                for i in range(self.n_vertices):
                    if mask[i] == 2:
                        # state[0, i] = 2
                        for j in range(self.n_vertices):
                            if self.matrix[i, j] == 1 and state[0, j] == 0:
                                state[0, j] = 1
            else: # 顶点不能重复选择
                # For irreversible vertices, initialise all to +1 (i.e. allowed to be flipped).
                state[0, :self.n_vertices] = 1
                '''
                初始化的时候都为0，表示该点没有被控制，可以被选择
                '''
                # state[0, :self.n_vertices] = np.zeros(self.n_vertices)
        else:
            state[0, :] = self._format_vertices_to_signed(vertices) # 取值1或者-1

        state = state.astype('float')

        # If any observables other than "immediate energy available" require setting to values other than
        # 0 at this stage, we should use a 'for k,v in enumerate(self.observables)' loop.
        for idx, obs in self.observables: # idx是5
            if obs==Observable.IMMEDIATE_REWARD_AVAILABLE:
                state[idx, :self.n_vertices] = self.get_immeditate_rewards_avaialable(vertices=state[0, :self.n_vertices]) / self.max_local_reward_available
            elif obs==Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE: # 初始化是这个
                immeditate_rewards_avaialable = self.get_immeditate_rewards_avaialable(vertices=state[0, :self.n_vertices])
                # 先计算有几个<=0的值，再得到>0的正值个数的比值
                # 将第五行的元素都设为这个比值
                # 表示有多少个可以使得奖励增加的顶点占总顶点数的比值
                state[idx, :self.n_vertices] = 1 - np.sum(immeditate_rewards_avaialable <= 0) / self.n_vertices

        return state

    def _get_vertices(self, basis=VertexBasis.TRINARY):
        '''
        unbiased时这里basis=VertexBasis.SIGNED
        返回的vertices就是顶点的向量
        '''
        vertices = self.state[0, :self.n_vertices]
        '''
        先假设这里只会有VertexBasis.TRINARY
        '''
        # if basis == VertexBasis.SIGNED:
        #     pass
        # elif basis == VertexSystemBiased:
        #     # convert {1,-1} --> {0,1}
        #     vertices[0, :] = (1 - vertices[0, :]) / 2
        # else:
        #     raise NotImplementedError("Unrecognised VertexBasis")

        return vertices

    # def calculate_best_energy(self):
    #     if self.n_vertices <= 10:
    #         # Generally, for small systems the time taken to start multiple processes is not worth it.
    #         res = self.calculate_best_brute()

    #     else:
    #         # Start up processing pool
    #         n_cpu = int(mp.cpu_count()) / 2

    #         pool = mp.Pool(mp.cpu_count())

    #         # Split up state trials across the number of cpus
    #         iMax = 2 ** (self.n_vertices)
    #         args = np.round(np.linspace(0, np.ceil(iMax / n_cpu) * n_cpu, n_cpu + 1))
    #         arg_pairs = [list(args) for args in zip(args, args[1:])]

    #         # Try all the states.
    #         #             res = pool.starmap(self._calc_over_range, arg_pairs)
    #         try:
    #             res = pool.starmap(self._calc_over_range, arg_pairs)
    #             # Return the best solution,
    #             idx_best = np.argmin([e for e, s in res])
    #             res = res[idx_best]
    #         except Exception as e:
    #             # Falling back to single-thread implementation.
    #             # res = self.calculate_best_brute()
    #             res = self._calc_over_range(0, 2 ** (self.n_vertices))
    #         finally:
    #             # No matter what happens, make sure we tidy up after outselves.
    #             pool.close()

    #         if self.vertex_basis == VertexBasis.BINARY:
    #             # convert {1,-1} --> {0,1}
    #             best_score, best_vertices = res
    #             best_vertices = (1 - best_vertices) / 2
    #             res = best_score, best_vertices

    #         if self.optimisation_target == OptimisationTarget.CDS:
    #             best_energy, best_vertices = res
    #             best_cds = self.calculate_cds(best_vertices)
    #             res = best_cds, best_vertices
    #         elif self.optimisation_target == OptimisationTarget.ENERGY:
    #             pass
    #         else:
    #             raise NotImplementedError()

    #     return res

    def seed(self, seed):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed
        np.random.seed(seed)

    def step(self, action):
        '''
        args:
            action: 当前选择的顶点
            
        返回一个四元组，表示执行action后的状态、奖励、完成标志和_
        只有一个episode完成done才为True
        '''
        done = False

        rew = 0 # Default reward to zero.
        randomised_vertices = False
        self.current_step += 1

        if self.current_step > self.max_steps: # 一个episode只选择2n次顶点
            print("The environment has already returned done. Stop it!")
            raise NotImplementedError

        new_state = np.copy(self.state) # 7xn的矩阵

        ############################################################
        # 1. Performs the action and calculates the score change. #
        ############################################################

        # ExtraAction.None, action=n时表示不做任何操作
        if action==self.n_vertices: 
            if self.extra_action == ExtraAction.PASS:
                delta_score = 0
            if self.extra_action == ExtraAction.RANDOMISE:
                # Randomise the vertex configuration.
                randomised_vertices = True
                random_actions = np.random.choice([1, -1], self.n_vertices)
                new_state[0, :] = self.state[0, :] * random_actions
                new_score = self.calculate_score(new_state[0, :])
                delta_score = new_score - self.score
                self.score = new_score
        else: # action应该是0~n-1之间的一个数
            # Perform the action and calculate the score change.
            # 将new_state[0, action]这个位置的值取反
            # 也就是1变成-1，或者-1变成1
            # 但是如果值为0，取反还是0，等于没做
            # new_state[0,action] = -self.state[0,action]
            '''
            我们这里设的是不能重复选择，所以说action一定是个不为2的点
            new_state[0,action]应设为2，表示当前的点被控制了
            与action这个点相邻的所有为0的点也要被改为1
            如果要重复选择的话想想怎么写
            '''
            # new_state[0,action] = 2
            # for idx, val in enumerate(self.matrix[action]):
            #     if val == 1 and self.state[0,idx] == 0:
            #         new_state[0,idx] = 1
            # 用np.where实现
            # new_state[0,:self.n_vertices] = np.where(self.state[0,:self.n_vertices] == 0, self.matrix[action], self.state[0,:self.n_vertices])
            # new_state[0,action] = 2
            '''
            假设顶点可以重复被选择
            1）选择的顶点是2
                将这个顶点赋为0，不能破坏CDS的连通性，这点由act()函数保证
                检查所有的邻居，如果邻居不与另一个赋为2的顶点相邻，那么将其赋为0
            2）选择的顶点是1
                将这个顶点赋为2，由于该顶点被邻居控制，加入CDS不会破坏连通性
                检查所有邻居，如果邻居值为0，那么将其赋为1
            3）选择的顶点是0
                不应该出现这种情况，如果赋为0的点加入CDS，那么新的CDS一定不连通
            '''
            if self.state[0, action] == 2:
                mask = self.state[0, :self.n_vertices].copy()
                # print(f'mask: {mask}')
                mask[action] = 0
                for i in range(self.n_vertices):
                    # 邻居i为1，检查其是否与赋为2的顶点相邻
                    if self.matrix[action, i] == 1 and self.state[0, i] == 1:
                        # 顶点i的所有邻居都没有加入CDS
                        if all(mask[j] < 2 for j in range(self.n_vertices) if self.matrix[i,j] == 1):
                            mask[i] = 0
                # 检查action这个顶点是否与2相连
                for i in range(self.n_vertices):
                    if self.matrix[action, i] == 1 and mask[i] == 2:
                        mask[action] = 1
                new_state[0, :self.n_vertices] = mask
            elif self.state[0, action] == 1:
                mask = self.state[0, :self.n_vertices].copy()
                mask[action] = 2
                for idx in range(self.n_vertices):
                    if self.matrix[action, idx] == 1 and self.state[0, idx] == 0:
                        mask[idx] = 1
                new_state[0, :self.n_vertices] = mask
            else:
                # raise ValueError("The selected vertex should be 1 or 2")
                '''
                如果顶点都为0，随机选择一个点加入CDS中
                '''
                mask = self.state[0, :self.n_vertices].copy()
                mask[action] = 2
                for idx in range(self.n_vertices):
                    if self.matrix[action, idx] == 1 and self.state[0, idx] == 0:
                        mask[idx] = 1
                new_state[0, :self.n_vertices] = mask
            
            if self.gg.biased:
                delta_score = self._calculate_score_change(new_state[0,:self.n_vertices], self.matrix, self.bias, action)
            else:
                # delta_score = self._calculate_score_change(new_state[0,:self.n_vertices], self.matrix, action)
                '''
                传旧的顶点状态和当前选择的顶点
                '''
                delta_score = self._calculate_score_change(self.state[0,:self.n_vertices], self.matrix, action)
            self.score += delta_score # 更新score，就知道了选择action后的分数

        #############################################################################################
        # 2. Calculate reward for action and update anymemory buffers.                              #
        #   a) Calculate reward (always w.r.t best observable score).                              #
        #   b) If new global best has been found: update best ever score and vertex parameters.      #
        #   c) If the memory buffer is finite (i.e. self.memory_length is not None):                #
        #          - Add score/vertices to their respective buffers.                                  #
        #          - Update best observable score and vertices w.r.t. the new buffers.                #
        #      else (if the memory is infinite):                                                    #
        #          - If new best has been found: update best observable score and vertex parameters. #                                                                        #
        #############################################################################################

        self.state = new_state # 更新state，形状还是7xn
        immeditate_rewards_avaialable = self.get_immeditate_rewards_avaialable()

        if self.score > self.best_obs_score: # 只有在出现更好的解的时候才更新外部reward
            if self.reward_signal == RewardSignal.BLS:
                rew = self.score - self.best_obs_score # BLS时rew是现在最好的分数和之前最好的分数的差
            elif self.reward_signal == RewardSignal.CUSTOM_BLS:
                rew = self.score - self.best_obs_score
                rew = rew / (rew + 0.1)

        if self.reward_signal == RewardSignal.DENSE:
            rew = delta_score
        elif self.reward_signal == RewardSignal.SINGLE and done:
            rew = self.score - self.init_score

        if self.norm_rewards: # True
            rew /= self.n_vertices # BLS时rew就是max(C(s_t)-C(s^*), 0)/|V|

        if self.stag_punishment is not None or self.basin_reward is not None: # stag_punishment和basin_reward必须有一个不为None
            visiting_new_state = self.history_buffer.update(action) # visiting_new_state是一个布尔变量

        if self.stag_punishment is not None: # None
            if not visiting_new_state:
                rew -= self.stag_punishment

        if self.basin_reward is not None: # 1/|V|
            if np.all(immeditate_rewards_avaialable <= 0): # 此时选任何一个解都不会立刻增加CDS，陷入了局部最优解
                # All immediate score changes are +ive <--> we are in a local minima.
                if visiting_new_state: # 这个局部最优解是之前没有出现过的
                    # #####TEMP####
                    # if self.reward_signal != RewardSignal.BLS or (self.score > self.best_obs_score):
                    # ####TEMP####
                    rew += self.basin_reward # 陷入局部最优解时更新内部reward

        if self.score > self.best_score:
            self.best_score = self.score
            self.best_vertices = self.state[0, :self.n_vertices].copy()

        if self.memory_length is not None: # 缓冲区长度有限
            # For case of finite memory length.
            self.score_memory[self.idx_memory] = self.score
            self.vertices_memory[self.idx_memory] = self.state[0, :self.n_vertices]
            self.idx_memory = (self.idx_memory + 1) % self.memory_length
            self.best_obs_score = self.score_memory.max()
            self.best_obs_vertices = self.vertices_memory[self.score_memory.argmax()].copy()
        else:
            self.best_obs_score = self.best_score
            self.best_obs_vertices = self.best_vertices.copy()

        #############################################################################################
        # 3. Updates the state of the system (except self.state[0,:] as this is always the vertex     #
        #    configuration and has already been done.                                               #
        #   a) Update self.state local features to reflect the chosen action.                       #                                                                  #
        #   b) Update global features in self.state (always w.r.t. best observable score/vertices)     #
        #############################################################################################

        for idx, observable in self.observables:

            ### Local observables ###
            if observable==Observable.IMMEDIATE_REWARD_AVAILABLE:
                self.state[idx, :self.n_vertices] = immeditate_rewards_avaialable / self.max_local_reward_available

            elif observable==Observable.TIME_SINCE_FLIP:
                # 这里的值最大，说明顶点越久没有被操作过了
                self.state[idx, :] += (1. / self.max_steps) # 这里体现了每一个点被选择后又过去了多少个step
                if randomised_vertices: # False
                    self.state[idx, :] = self.state[idx, :] * (random_actions > 0)
                else:
                    self.state[idx, action] = 0 # 当前被选择的点置为0

            ### Global observables ###
            elif observable==Observable.EPISODE_TIME:
                self.state[idx, :] += (1. / self.max_steps)

            elif observable==Observable.TERMINATION_IMMANENCY:
                # Update 'Immanency of episode termination'
                # 这个值最大表示里这轮episode越久
                self.state[idx, :] = max(0, ((self.current_step - self.max_steps) / self.horizon_length) + 1)

            elif observable==Observable.NUMBER_OF_GREEDY_ACTIONS_AVAILABLE:
                # 表示能使得reward增加的顶点数占总顶点数的比值
                self.state[idx, :] = 1 - np.sum(immeditate_rewards_avaialable <= 0) / self.n_vertices

            elif observable==Observable.DISTANCE_FROM_BEST_SCORE:
                # 当前分数与已经得到的最佳的分数之间的差距
                self.state[idx, :] = np.abs(self.score - self.best_obs_score) / self.max_local_reward_available

            elif observable==Observable.DISTANCE_FROM_BEST_STATE:
                # 当前顶点的选择情况和最佳情况的差别
                self.state[idx, :self.n_vertices] = np.count_nonzero(self.best_obs_vertices[:self.n_vertices] - self.state[0, :self.n_vertices])

        #############################################################################################
        # 4. Check termination criteria.                                                            #
        #############################################################################################
        
        if self.current_step == self.max_steps:
            # Maximum number of steps taken --> done.
            # print("Done : maximum number of steps taken")
            done = True

        if not self.reversible_vertices: # reversible_vertices=False时才会执行，所以这里我们先不用改
            # 判断是否存在大于0的元素，在这里1表示可以被选择的顶点
            if len((self.state[0, :self.n_vertices] > 0).nonzero()[0]) == 0:
                # If no more vertices to flip --> done.
                # print("Done : no more vertices to flip")
                done = True
            '''
            不能重复选择的情况，如果没有为0的点说明没有点可以选择了
            '''
            # if np.all(self.state[0, :self.n_vertices] > 0):
                # done = True

        return (self.get_observation(), rew, done, None)

    def get_observation(self):
        '''
        前面设置state时顶点x的取值是-1或者1
        在这里会处理BINARY的情况，将取值变成0或者1
        将state矩阵和matrix_obs矩阵按照垂直方向进行叠加
        新矩阵的列数不变，行数是7+顶点数，前7行是observables
        返回这个叠加后的矩阵
        '''
        state = self.state.copy()
        '''
        要求取值就是0，1，2，所以这里不需要了
        '''
        if self.vertex_basis == VertexBasis.BINARY:
            # convert {1,-1} --> {0,1}
            state[0,:] = (1-state[0,:])/2
        elif self.vertex_basis == VertexBasis.TRINARY:
            pass

        if self.gg.biased:
            return np.vstack((state, self.matrix_obs, self.bias_obs))
        else:
            return np.vstack((state, self.matrix_obs))

    def get_immeditate_rewards_avaialable(self, vertices=None):
        '''
        返回一个向量，每一个元素的值表示选择第i个顶点后立刻得到的CDS
        '''
        if vertices is None:
            vertices = self._get_vertices()

        if self.optimisation_target==OptimisationTarget.ENERGY:
            immediate_reward_function = lambda *args: -1*self._get_immeditate_energies_avaialable_jit(*args)
        elif self.optimisation_target==OptimisationTarget.CDS:
            # 如果vertices是全1的，matrix @ vertices得到一个列向量，每个元素是matrix的一行之和
            '''
            CDS中的割点和CDS是否控制了整个图在这里计算好
            '''
            immediate_reward_function = self._get_immeditate_cds_avaialable_jit
        else:
            raise NotImplementedError("Optimisation target {} not recognised.".format(self.optimisation_ta))

        vertices = vertices.astype('float64')
        matrix = self.matrix.astype('float64')

        if self.gg.biased:
            bias = self.bias.astype('float64')
            return immediate_reward_function(vertices,matrix,bias)
        else:
            return immediate_reward_function(vertices,matrix)

    def get_allowed_action_states(self):
        '''
        如果顶点可以重复选择，那么返回(0,1)或者(1,-1)
        否则，返回0或者1
        BINARY中0表示可以选择的点
        SIGNED中1表示可以选择的点
        '''
        if self.reversible_vertices:
            # If MDP is reversible, both actions are allowed.
            if self.vertex_basis == VertexBasis.BINARY:
                return (0,1)
            elif self.vertex_basis == VertexBasis.SIGNED:
                return (1,-1)
            elif self.vertex_basis == VertexBasis.TRINARY:
                pass
            '''
            1) 值为2且不为割点的点可以被选择
            2）值为1的点可以被选择
            3）值为0的点不能被选择
            '''
        else:
            # If MDP is irreversible, only return the state of vertices that haven't been flipped.
            if self.vertex_basis==VertexBasis.BINARY:
                return 0
            if self.vertex_basis==VertexBasis.SIGNED:
                return 1
            elif self.vertex_basis == VertexBasis.TRINARY and self.current_step == 0:
                '''
                初始化时0是可以选择的
                '''
                return 0
            elif self.vertex_basis == VertexBasis.TRINARY and self.current_step > 0:
                '''
                但是之后只能选择为1的点
                '''
                return 1

    def calculate_score(self, vertices=None):
        '''
        在reset中调用时vertices=state[0]，score即是初始时的cds值
        '''
        if self.optimisation_target==OptimisationTarget.CDS:
            score = self.calculate_cds(vertices)
        elif self.optimisation_target==OptimisationTarget.ENERGY:
            score = -1.*self.calculate_energy(vertices)
        else:
            raise NotImplementedError
        return score

    def _calculate_score_change(self, new_vertices, matrix, action):
        '''
        计算CDS时new_vertices传进来的是旧的顶点的状态
        '''
        if self.optimisation_target==OptimisationTarget.CDS:
            delta_score = self._calculate_cds_change(new_vertices, matrix, action)
        elif self.optimisation_target == OptimisationTarget.ENERGY:
            delta_score = -1. * self._calculate_energy_change(new_vertices, matrix, action)
        else:
            raise NotImplementedError
        return delta_score

    def _format_vertices_to_signed(self, vertices):
        '''
        BINARY时：
        原本vertices取0或1，然后放大2倍再减1，vertices取1或者-1
        SIGNED时：
        vertices不变，还是取1或者-1
        因为在算cds时x的取值就是1或者-1，这里是为了当BINARY时把vertices的取值变成需要的取值范围
        返回vertices向量，取值1或者-1
        '''
        if self.vertex_basis == VertexBasis.BINARY:
            if not np.isin(vertices, [0, 1]).all():
                raise Exception("VertexSystem is configured for binary vertices ([0,1]).")
            # Convert to signed vertices for calculation.
            vertices = 2 * vertices - 1
        elif self.vertex_basis == VertexBasis.SIGNED:
            if not np.isin(vertices, [-1, 1]).all():
                raise Exception("VertexSystem is configured for signed vertices ([-1,1]).")
        elif self.vertex_basis == VertexBasis.TRINARY:
            '''
            假设取值是规范的0，1，2
            '''
            pass

        return vertices

    @abstractmethod
    def calculate_energy(self, vertices=None):
        raise NotImplementedError

    @abstractmethod
    def calculate_cds(self, vertices=None):
        raise NotImplementedError

    @abstractmethod
    def get_best_cds(self):
        raise NotImplementedError

    @abstractmethod
    def _calc_over_range(self, i0, iMax):
        raise NotImplementedError

    @abstractmethod
    def _calculate_energy_change(self, new_vertices, matrix, action):
        raise NotImplementedError

    @abstractmethod
    def _calculate_cds_change(self, new_vertices, matrix, action):
        raise NotImplementedError
    
    def get_subgraph(self, vertices: list) -> np.ndarray:
        '''
        返回一个子图，子图的顶点是CDS中的顶点
        '''
        n = len(vertices)
        subgraph_matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                subgraph_matrix[i, j] = self.matrix[int(vertices[i]), int(vertices[j])]
        return subgraph_matrix
    
    def is_cut_vertex(self, vertex: int) -> bool:
        '''
        args:
            current_cds: CDS中顶点的索引
            induced_cds: 以这个CDS作为导出子图，
        
        判断一个点是否是割点
        '''
        # current_cds = [self.state[0, :] == 2].nonzero()
        current_cds = [i for i, val in enumerate(self.state[0, :]) if val == 2]
        induced_cds = self.get_subgraph(current_cds)
        
        n = len(induced_cds)
        
        # 获取vertex在导出子图中的idx
        new_idx = current_cds.index(vertex)
    
        # 创建一个邻接矩阵副本，并删除指定的顶点
        submatrix = np.delete(induced_cds, new_idx, axis=0)
        submatrix = np.delete(submatrix, new_idx, axis=1)
        
        # 创建一个已访问顶点的数组
        visited = [False] * (n - 1)
        
        # 找到第一个未被删除的顶点并进行DFS遍历
        # for start in range(n - 1):
        #     if not visited[start]:
        #         break
        for start in range(n - 1):
            if any(submatrix[start, :]):
                break
        else:
            # 如果图为空（所有顶点都孤立），直接返回 False
            return False
        
        def dfs(v):
            visited[v] = True
            for u in range(n - 1):
                if submatrix[v][u] == 1 and not visited[u]:
                    dfs(u)
        
        # 从start开始进行深度优先搜索
        dfs(start)
        
        # 如果有未访问的顶点，说明删除该顶点导致图不连通
        return not all(visited)
        

##########
# Classes for implementing the calculation methods with/without biases.
##########
class VertexSystemUnbiased(VertexSystemBase):

    def calculate_energy(self, vertices=None):
        if vertices is None:
            vertices = self._get_vertices()
        else:
            vertices = self._format_vertices_to_signed(vertices)

        vertices = vertices.astype('float64')
        matrix = self.matrix.astype('float64')

        return self._calculate_energy_jit(vertices, matrix)

    def calculate_cds(self, vertices=None):
        '''
        reset时vertices=state[0]
        在计算CUT之前，如果是BINARY，取值0和1，要把取值变成-1和1再计算
        返回CUT值
        '''
        if vertices is None:
            vertices = self._get_vertices() # basis=None
        else:
            vertices = self._format_vertices_to_signed(vertices)

        # return (1/4) * np.sum( np.multiply( self.matrix, 1 - np.outer(vertices, vertices) ) )
        '''
        标为2的顶点的个数就是连通控制数
        '''
        return np.sum(vertices == 2)

    def get_best_cds(self):
        if self.optimisation_target==OptimisationTarget.CDS:
            return self.best_score
        else:
            raise NotImplementedError("Can't return best cds when optimisation target is set to energy.")

    def _calc_over_range(self, i0, iMax):
        list_vertices = [2 * np.array([int(x) for x in list_string]) - 1
                      for list_string in
                      [list(np.binary_repr(i, width=self.n_vertices))
                       for i in range(int(i0), int(iMax))]]
        matrix = self.matrix.astype('float64')
        return self.__calc_over_range_jit(list_vertices, matrix)
    
    '''
    数组是连续存储的话使用@运算符效率会更高
    1）计算之前检查
    if not new_vertices.flags['C_CONTIGUOUS']:
        new_vertices = np.ascontiguousarray(new_vertices)
    2）创建连续数组
    new_vertices = np.array(your_data, order='C')
    '''

    @staticmethod
    @jit(float64(float64[:],float64[:,:],int64), nopython=True)
    def _calculate_energy_change(new_vertices, matrix, action):
        # return -2 * new_vertices[action] * matmul(new_vertices.T, matrix[:, action])
        return -2 * new_vertices[action] * (new_vertices.T @ matrix[:, action])

    # @staticmethod
    # @jit(float64(float64[:], float64[:,:], int64), nopython=True, nogil=True)
    # def _calculate_energy_change(new_vertices, matrix, action):
    #     # Convert arrays to contiguous layout
    #     new_vertices_contig = np.ascontiguousarray(new_vertices, requirements='C')
    #     matrix_contig = np.ascontiguousarray(matrix, requirements='C')

    #     # Perform the calculation
    #     return -2 * new_vertices_contig[action] * (new_vertices_contig.T @ matrix_contig[:, action])
    
    @staticmethod
    @jit(float64(float64[:],float64[:,:],int64), nopython=True)
    def _calculate_cds_change(new_vertices, matrix, action):
        '''
        args:
            new_vertices: 这个值传进来的是state[0,:]，不是new_state[0,:]
        计算选择了action这个顶点后CDS的变化
        matrix[:, action]是与这个顶点相关的点的连接情况
        返回CDS变化的值
        '''
        # return -1 * new_vertices[action] * matmul(new_vertices.T, matrix[:, action])
        # return -1 * new_vertices[action] * (new_vertices.T @ matrix[:, action])
        '''
        如果顶点不能重复选择的话，这里是不是只会变大？
        '''
        # return 1
        '''
        action是一个顶点的索引，其顶点的取值是1或者2
            1）值为2的顶点选择后变成0，CDS减少1
            2）值为1的顶点选择后变成2，CDS变大1
        '''
        return 1 if new_vertices[action] == 1 else -1

    @staticmethod
    @jit(float64(float64[:],float64[:,:]), nopython=True)
    def _calculate_energy_jit(vertices, matrix):
        # return - matmul(vertices.T, matmul(matrix, vertices)) / 2
        return - (vertices.T @ (matrix @ vertices)) / 2

    @staticmethod
    @jit(parallel=True)
    def __calc_over_range_jit(list_vertices, matrix):
        energy = 1e50
        best_vertices = None

        for vertices in list_vertices:
            vertices = vertices.astype('float64')
            # This is self._calculate_energy_jit without calling to the class or self so jit can do its thing.
            current_energy = - matmul(vertices.T, matmul(matrix, vertices)) / 2
            if current_energy < energy:
                energy = current_energy
                best_vertices = vertices
        return energy, best_vertices

    @staticmethod
    @jit(float64[:](float64[:],float64[:,:]), nopython=True)
    def _get_immeditate_energies_avaialable_jit(vertices, matrix):
        return 2 * vertices * matmul(matrix, vertices)

    '''
    静态方法不接受self，只进行计算
    那判断割点和是否控制的逻辑应该放在外面进行
    '''
    
    @staticmethod
    @jit(float64[:](float64[:],float64[:,:]), nopython=True)
    def _get_immeditate_cds_avaialable_jit(vertices, matrix):
        '''
        假设vertex加入了S标为1，否则为-1
        matmul(matrix, vertices)得到一个列向量，其中每一个元素表示该顶点有几个邻居在S中
        vertices * matmul(matrix, vertices)算出在S中的顶点一共有多少个邻居在S中
        比如说有k个顶点在S中，x1,..., xk
        返回的标量是每一个xi在S中的邻居之和，所以有些顶点应该会被重复计算
        为什么不直接把S中的顶点个数当作奖励呢？
        '''
        # return vertices * matmul(matrix, vertices) # matrix和vertices相乘得到一个向量
        '''
        这里返回的是一个向量而不是标量
        每一个元素都表示选择这个顶点后立刻能得到的奖励
        这个奖励还真不太好设计，如果选标为1的顶点获得2，那么程序会倾向于把所有1的点都标为2
        '''
        # rew = np.zeros(len(vertices))
        # for idx, val in enumerate(vertices):
        #     if val == 1:
        #         rew[idx] = -0.1
        #     elif val == 0:
        #         rew[idx] = 1
        # return rew.astype('float64')
        '''
        值为0的顶点：不应该选，给一个较大的负值
        值为1的顶点：选择后加入控制集
        值为2的顶点：如果是割点不应该选，否则选择后移出控制集
        在CDS没有控制整个图，即state[0,:]中还有为0的顶点时，选1的奖励应该要更大一点
        '''
        rew = np.zeros(len(vertices))
        for idx, val in enumerate(vertices):
            if val == 0:
                rew[idx] = -10000
            elif val == 1:
                rew[idx] = 1
            elif val == 2:
                # rew[idx] = -10000 if is_cut_vertex(idx) else -1
                rew[idx] = -1
        return rew.astype('float64')

# class VertexSystemBiased(VertexSystemBase):

#     def calculate_energy(self, vertices=None):
#         if type(vertices) == type(None):
#             vertices = self._get_vertices()

#         vertices = vertices.astype('float64')
#         matrix = self.matrix.astype('float64')
#         bias = self.bias.astype('float64')

#         return self._calculate_energy_jit(vertices, matrix, bias)

#     def calculate_cds(self, vertices=None):
#         raise NotImplementedError("MaxCut not defined/implemented for biased VertexSystems.")

#     def get_best_cds(self):
#         raise NotImplementedError("MaxCut not defined/implemented for biased VertexSystems.")

#     def _calc_over_range(self, i0, iMax):
#         list_vertices = [2 * np.array([int(x) for x in list_string]) - 1
#                       for list_string in
#                       [list(np.binary_repr(i, width=self.n_vertices))
#                        for i in range(int(i0), int(iMax))]]
#         matrix = self.matrix.astype('float64')
#         bias = self.bias.astype('float64')
#         return self.__calc_over_range_jit(list_vertices, matrix, bias)

#     @staticmethod
#     @jit(nopython=True)
#     def _calculate_energy_change(new_vertices, matrix, bias, action):
#         return 2 * new_vertices[action] * (matmul(new_vertices.T, matrix[:, action]) + bias[action])

#     @staticmethod
#     @jit(nopython=True)
#     def _calculate_cds_change(new_vertices, matrix, bias, action):
#         raise NotImplementedError("MaxCut not defined/implemented for biased VertexSystems.")

#     @staticmethod
#     @jit(nopython=True)
#     def _calculate_energy_jit(vertices, matrix, bias):
#         return matmul(vertices.T, matmul(matrix, vertices))/2 + matmul(vertices.T, bias)

#     @staticmethod
#     @jit(parallel=True)
#     def __calc_over_range_jit(list_vertices, matrix, bias):
#         energy = 1e50
#         best_vertices = None

#         for vertices in list_vertices:
#             vertices = vertices.astype('float64')
#             # This is self._calculate_energy_jit without calling to the class or self so jit can do its thing.
#             current_energy = -( matmul(vertices.T, matmul(matrix, vertices))/2 + matmul(vertices.T, bias))
#             if current_energy < energy:
#                 energy = current_energy
#                 best_vertices = vertices
#         return energy, best_vertices

#     @staticmethod
#     @jit(nopython=True)
#     def _get_immeditate_energies_avaialable_jit(vertices, matrix, bias):
#         return - (2 * vertices * (matmul(matrix, vertices) + bias))

#     @staticmethod
#     @jit(nopython=True)
#     def _get_immeditate_cds_avaialable_jit(vertices, matrix, bias):
#         raise NotImplementedError("MaxCut not defined/implemented for biased VertexSystems.")