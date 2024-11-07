import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import src.envs.core as ising_env
from src.utils import load_graph_set, mk_dir
from src.agents.dqn.dqn import DQN
from src.agents.dqn.utils import TestMetric
from src.envs.utils import (SetGraphGenerator,
                            RandomBarabasiAlbertGraphGenerator,
                            EdgeType, RewardSignal, ExtraAction,
                            OptimisationTarget, VertexBasis,
                            DEFAULT_OBSERVABLES)
from src.networks.mpnn import MPNN
from pathlib import Path

try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
except ImportError:
    pass

import time

def run(n_vertices, timestep, step_factor, save_loc="checkpoints"):

    print("\n----- Running {} -----\n".format(os.path.basename(__file__))) # 获取当前文件名

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    gamma=0.95 # 折扣因子
    # step_fact = 2 # step_fact * len(vertices)是一个回合episode的长度，这里是顶点个数的两倍
    
    step_fact = step_factor
    '''
    由于现在设的点是不能重复选择的，那么控制数最大也不会超过n，这里的step_fact给的小一点
    更精确一点的话，\gamma_c(G) <= n - \Delta(G)，max_step应该设置的更精确一点
    不用更精确也行，不能重复选择的时候做一个停止的条件就行
    '''
    # step_fact = 1

    env_args = {'observables':DEFAULT_OBSERVABLES,
                'reward_signal':RewardSignal.BLS, # 宽度学习系统
                'extra_action':ExtraAction.NONE, # ExtraAction设为NONE
                'optimisation_target':OptimisationTarget.CDS,
                # 'vertex_basis':VertexBasis.BINARY, # vertex取0或1
                'vertex_basis':VertexBasis.TRINARY, # vertex取0, 1, 2
                'norm_rewards':True, # 使得reward不超过1
                'memory_length':None,
                'horizon_length':None,
                'stag_punishment':None,
                'basin_reward':1./n_vertices, # 中间奖励
                'reversible_vertices':True} # vertex可以反复被操作
                # 'reversible_vertices':False}

    ####################################################
    # SET UP TRAINING AND TEST GRAPHS
    ####################################################

    n_vertices_train = n_vertices
    # DISCRETE边的取值为-1, 0 ,1
    train_graph_generator = RandomBarabasiAlbertGraphGenerator(n_vertices=n_vertices_train,m_insertion_edges=4,edge_type=EdgeType.DISCRETE)

    ####
    # Pre-generated test graphs
    ####
    # graph_save_loc = "_graphs/testing/BA_20vertex_m4_50graphs.pkl"
    # graphs_test = load_graph_set(graph_save_loc)
    # n_tests = len(graphs_test) # 50个图的邻接矩阵，取值-1, 0, 1，这里是带权图

    # 改成随机生成的50个图
    graphs_test = []
    
    for _ in range(50):
       graphs_test.append(RandomBarabasiAlbertGraphGenerator(n_vertices=n_vertices_train,m_insertion_edges=4,edge_type=EdgeType.DISCRETE).get())
    
    n_tests = len(graphs_test)

    test_graph_generator = SetGraphGenerator(graphs_test, ordered=True)

    ####################################################
    # SET UP TRAINING AND TEST ENVIRONMENTS
    ####################################################

    # train_envs是一个list，里面只有一个VertexSystemUnbiased对象
    train_envs = [ising_env.make("VertexSystem",
                                 train_graph_generator,
                                 int(n_vertices_train*step_fact),
                                 **env_args)]

    # 测试的一个回合的长度和训练设为一致
    # 这里测试用的图也是20个顶点的
    n_vertices_test = train_graph_generator.get().shape[0]
    test_envs = [ising_env.make("VertexSystem",
                                test_graph_generator,
                                int(n_vertices_test*step_fact),
                                **env_args)]

    ####################################################
    # SET UP FOLDERS FOR SAVING DATA
    ####################################################

    if isinstance(save_loc, str):
        save_loc = Path(save_loc)
    
    # data_folder = os.path.join(save_loc,'data')
    # network_folder = os.path.join(save_loc, 'network')
    data_folder = save_loc / 'data'
    network_folder = save_loc / 'network'

    mk_dir(data_folder)
    mk_dir(network_folder)
    # print(data_folder)
    # network_save_path = os.path.join(network_folder,'network.pth')
    # test_save_path = os.path.join(network_folder,'test_scores.pkl')
    # loss_save_path = os.path.join(network_folder, 'losses.pkl')
    network_save_path = network_folder / 'network.pth'
    test_save_path = network_folder / 'test_scores.pkl'
    loss_save_path = network_folder / 'losses.pkl'

    ####################################################
    # SET UP AGENT
    ####################################################

    # 总回合数
    # nb_steps = 2500000
    # nb_steps = 10000
    nb_steps = timestep

    network_fn = lambda: MPNN(n_obs_in=train_envs[0].observation_space.shape[1], # 状态个数
                              n_layers=3,
                              n_features=64, # 每个点向量的长度为64
                              n_hid_readout=[],
                              tied_weights=False)

    agent = DQN(train_envs, # 这是一个VertexSystemUnbasised对象

                network_fn, # 这里是一个函数

                init_network_params=None,
                init_weight_std=0.01,

                double_dqn=True,
                clip_Q_targets=False, # 不对目标Q值裁剪

                replay_start_size=500,
                replay_buffer_size=5000,  # 20000
                gamma=gamma,  # discount factor
                # 每1000次更新目标网络的参数一次
                update_target_frequency=1000,  # 500

                update_learning_rate=False, # 不更新学习率
                initial_learning_rate=1e-4,
                peak_learning_rate=1e-4,
                peak_learning_rate_step=20000,
                final_learning_rate=1e-4,
                final_learning_rate_step=200000,

                # 每32回合更新一次主网络
                update_frequency=32,  # 1
                # 从回放区采样64个经验
                minibatch_size=64,  # 128
                max_grad_norm=None,
                weight_decay=0, # 权重衰退

                update_exploration=True, # 动态eplison贪心
                initial_exploration_rate=1,
                final_exploration_rate=0.05,  # 0.05
                final_exploration_step=150000,  # 40000

                adam_epsilon=1e-8,
                logging=False,
                # logging=True,
                loss="mse", # 损失函数

                save_network_frequency=100000, # 每100000次保存一次网络
                network_save_path=network_save_path,

                evaluate=True,
                test_envs=test_envs,
                test_episodes=n_tests, # 这里是50
                test_frequency=10000,  # 每一万次才输出一个Test score
                test_save_path=test_save_path,
                test_metric=TestMetric.MIN_CDS,

                seed=None
                )

    print("\n Created DQN agent with network:\n\n", agent.network)

    #############
    # TRAIN AGENT
    #############
    start = time.time()
    agent.learn(timesteps=nb_steps, verbose=True)
    print(time.time() - start)

    agent.save()

    ############
    # PLOT - learning curve
    ############
    data = pickle.load(open(test_save_path,'rb'))
    data = np.array(data)

    fig_fname = os.path.join(network_folder,"training_curve")
    # fig_fname = network_folder / "training_curve"

    plt.plot(data[:,0],data[:,1])
    plt.xlabel("Timestep")
    plt.ylabel("Mean reward")
    if agent.test_metric==TestMetric.ENERGY_ERROR:
      plt.ylabel("Energy Error")
    elif agent.test_metric==TestMetric.BEST_ENERGY:
      plt.ylabel("Best Energy")
    elif agent.test_metric==TestMetric.CUMULATIVE_REWARD:
      plt.ylabel("Cumulative Reward")
    elif agent.test_metric==TestMetric.MIN_CDS:
      plt.ylabel("MIN CDS")
    elif agent.test_metric==TestMetric.FINAL_CDS:
      plt.ylabel("Final CDS")

    plt.savefig(fig_fname + ".png", bbox_inches='tight')
    plt.savefig(fig_fname + ".pdf", bbox_inches='tight')

    plt.clf()

    ############
    # PLOT - losses
    ############
    data = pickle.load(open(loss_save_path,'rb'))
    data = np.array(data)

    fig_fname = os.path.join(network_folder,"loss")
    # fig_fname = network_folder / "loss"

    N=50
    data_x = np.convolve(data[:,0], np.ones((N,))/N, mode='valid')
    data_y = np.convolve(data[:,1], np.ones((N,))/N, mode='valid')

    plt.plot(data_x,data_y)
    plt.xlabel("Timestep")
    plt.ylabel("Loss")

    plt.yscale("log")
    plt.grid(True)

    plt.savefig(fig_fname + ".png", bbox_inches='tight')
    plt.savefig(fig_fname + ".pdf", bbox_inches='tight')

if __name__ == "__main__":
    run()