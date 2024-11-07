import os

import matplotlib.pyplot as plt
import torch

import src.envs.core as ising_env
from src.utils import test_network, load_graph_set
from src.envs.utils import (SingleGraphGenerator,
                            RewardSignal, ExtraAction,
                            OptimisationTarget, VertexBasis,
                            DEFAULT_OBSERVABLES,
                            RandomBarabasiAlbertGraphGenerator,
                            EdgeType, SetGraphGenerator)
from src.networks.mpnn import MPNN

from src.configs.common_configs import OtherConfig, PathConfig

from pathlib import Path

try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
except ImportError:
    pass

def run(n_vertices, step_factor, save_loc="checkpoints",
        # graph_save_loc="_graphs/validation/BA_20spin_m4_100graphs.pkl",
        batched=True,
        max_batch_size=None) -> None:

    print("\n----- Running {} -----\n".format(os.path.basename(__file__)))

    ####################################################
    # NETWORK LOCATION
    ####################################################

    # data_folder = os.path.join(save_loc, 'data')
    # network_folder = os.path.join(save_loc, 'network')
    if isinstance(save_loc, str):
        save_loc = Path(save_loc)
    
    data_folder = save_loc / 'data'
    network_folder = save_loc / 'network'

    print("data folder :", data_folder)
    print("network folder :", network_folder)

    # test_save_path = os.path.join(network_folder, 'test_scores.pkl')
    # network_save_path = os.path.join(network_folder, 'network_best.pth')
    
    test_save_path = network_folder / 'test_scores.pkl'
    network_save_path = network_folder / 'network_best.pth'

    print("network params :", network_save_path)

    ####################################################
    # NETWORK SETUP
    ####################################################

    network_fn = MPNN
    network_args = {
        'n_layers': 3,
        'n_features': 64,
        'n_hid_readout': [],
        'tied_weights': False
    }

    ####################################################
    # SET UP ENVIRONMENTAL AND VARIABLES
    ####################################################

    gamma = 0.95 # discount factor
    step_factor = step_factor

    env_args = {'observables': DEFAULT_OBSERVABLES,
                'reward_signal': RewardSignal.BLS,
                'extra_action': ExtraAction.NONE,
                'optimisation_target': OptimisationTarget.CDS,
                'vertex_basis': VertexBasis.TRINARY,
                'norm_rewards': True,
                'memory_length': None,
                'horizon_length': None,
                'stag_punishment': None,
                'basin_reward': 1. / n_vertices, # 这个奖励在什么情况下会被给出
                'reversible_vertices': True}

    ####################################################
    # LOAD VALIDATION GRAPHS
    ####################################################

    # 测试集里是100张图
    
    n_vertices_test = n_vertices
        
    # graphs_test = load_graph_set(graph_save_loc)
    
    graphs_test = []
    for _ in range(100):
       graphs_test.append(RandomBarabasiAlbertGraphGenerator(n_vertices=n_vertices_test,m_insertion_edges=4,edge_type=EdgeType.DISCRETE).get())

    test_graph_generator = SetGraphGenerator(graphs_test, ordered=True)
    
    ####################################################
    # SETUP NETWORK TO TEST
    ####################################################

    # test_env = ising_env.make("VertexSystem", 
    #                           SingleGraphGenerator(graphs_test[0]),
    #                           graphs_test[0].shape[0]*step_factor,
    #                           **env_args)
    
    test_env = ising_env.make("VertexSystem", 
                              test_graph_generator,
                              graphs_test[0].shape[0]*step_factor,
                              **env_args)

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = OtherConfig().device
    
    torch.device(device)
    print("Set torch default device to {}.".format(device))

    network = network_fn(n_obs_in=test_env.observation_space.shape[1], # 状态数
                         **network_args).to(device)

    network.load_state_dict(torch.load(network_save_path,map_location=device))
    for param in network.parameters():
        param.requires_grad = False
    network.eval()

    print("Sucessfully created agent with pre-trained MPNN.\nMPNN architecture\n\n{}".format(repr(network)))

    ####################################################
    # TEST NETWORK ON VALIDATION GRAPHS
    ####################################################

    results, results_raw, history = test_network(network, env_args, graphs_test, device, step_factor,
                                                 return_raw=True, return_history=True,
                                                 batched=batched, max_batch_size=max_batch_size)

    # results_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + ".pkl"
    # results_raw_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_raw.pkl"
    # history_fname = "results_" + os.path.splitext(os.path.split(graph_save_loc)[-1])[0] + "_history.pkl"
    
    results_fname = "results_BA_20spin_m4_100graphs.pkl"
    results_raw_fname = "results_BA_20spin_m4_100graphs_raw.pkl"
    history_fname = "results_BA_20spin_m4_100graphs_history.pkl"

    for res, fname, label in zip([results, results_raw, history],
                                 [results_fname, results_raw_fname, history_fname],
                                 ["results", "results_raw", "history"]):
        # save_path = os.path.join(data_folder, fname)
        save_path = data_folder / fname
        res.to_pickle(save_path)
        print("{} saved to {}".format(label, save_path))

if __name__ == "__main__":
    run()