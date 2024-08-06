"""
Implements a DQN learning agent.
"""

import os
import pickle
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from src.agents.dqn.utils import ReplayBuffer, Logger, TestMetric, set_global_seed
from src.envs.utils import ExtraAction

class DQN:
    """
    # Required parameters.
    envs : List of environments to use.
    network : Choice of neural network.

    # Initial network parameters.
    init_network_params : Pre-trained network to load upon initialisation.
    init_weight_std : Standard deviation of initial network weights.

    # DQN parameters
    double_dqn : Whether to use double DQN (DDQN).
    update_target_frequency : How often to update the DDQN target network.
    gamma : Discount factor.
    clip_Q_targets : Whether negative Q targets are clipped (generally True/False for irreversible/reversible agents).

    # Replay buffer.
    replay_start_size : The capacity of the replay buffer at which training can begin.
    replay_buffer_size : Maximum buffer capacity.
    minibatch_size : Minibatch size.
    update_frequency : Number of environment steps taken between parameter update steps.

    # Learning rate
    update_learning_rate : Whether to dynamically update the learning rate (if False, initial_learning_rate is always used).
    initial_learning_rate : Initial learning rate.
    peak_learning_rate : The maximum learning rate.
    peak_learning_rate_step : The timestep (from the start, not from when training starts) at which the peak_learning_rate is found.
    final_learning_rate : The final learning rate.
    final_learning_rate_step : The timestep of the final learning rate.

    # Optional regularization.
    max_grad_norm : The norm grad to clip gradients to (None means no clipping).
    weight_decay : The weight decay term for regularisation.

    # Exploration
    update_exploration : Whether to update the exploration rate (False would tend to be used with NoisyNet layers).
    initial_exploration_rate : Inital exploration rate.
    final_exploration_rate : Final exploration rate.
    final_exploration_step : Timestep at which the final exploration rate is reached.

    # Loss function
    adam_epsilon : epsilon for ADAM optimisation.
    loss="mse" : Loss function to use.

    # Saving the agent
    save_network_frequency : Frequency with which the network parameters are saved.
    network_save_path : Folder into which the network parameters are saved.

    # Testing the agent
    evaluate : Whether to test the agent during training.
    test_envs : List of test environments.  None means the training environments (envs) are used.
    test_episodes : Number of episodes at each test point.
    test_frequency : Frequency of tests.
    test_save_path : Folder into which the test scores are saved.
    test_metric : The metric used to quantify performance.

    # Other
    logging : Whether to log.
    seed : The global seed to set.  None means randomly selected.
    """
    def __init__(
        self,
        envs,
        network,

        # Initial network parameters.
        init_network_params = None,
        init_weight_std = None,

        # DQN parameters
        double_dqn = True,
        update_target_frequency=10000,
        gamma=0.99,
        clip_Q_targets=False,

        # Replay buffer.
        replay_start_size=50000,
        replay_buffer_size=1000000,
        minibatch_size=32,
        update_frequency=1,

        # Learning rate
        update_learning_rate = True,
        initial_learning_rate = 0,
        peak_learning_rate = 1e-3,
        peak_learning_rate_step = 10000,
        final_learning_rate = 5e-5,
        final_learning_rate_step = 200000,

        # Optional regularization.
        max_grad_norm=None,
        weight_decay=0,

        # Exploration
        update_exploration=True,
        initial_exploration_rate=1,
        final_exploration_rate=0.1,
        final_exploration_step=1000000,

        # Loss function
        adam_epsilon=1e-8,
        loss="mse",

        # Saving the agent
        save_network_frequency=10000,
        network_save_path='network',

        # Testing the agent
        evaluate=True,
        test_envs=None,
        test_episodes=20,
        test_frequency=10000,
        test_save_path='test_scores',
        test_metric=TestMetric.ENERGY_ERROR,

        # Other
        logging=True,
        seed=None
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.double_dqn = double_dqn

        self.replay_start_size = replay_start_size
        self.replay_buffer_size = replay_buffer_size
        self.gamma = gamma
        self.clip_Q_targets = clip_Q_targets
        self.update_target_frequency = update_target_frequency
        self.minibatch_size = minibatch_size

        self.update_learning_rate = update_learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.peak_learning_rate_step = peak_learning_rate_step
        self.final_learning_rate = final_learning_rate
        self.final_learning_rate_step = final_learning_rate_step

        self.max_grad_norm = max_grad_norm
        self.weight_decay=weight_decay
        self.update_frequency = update_frequency
        self.update_exploration = update_exploration,
        self.initial_exploration_rate = initial_exploration_rate
        self.epsilon = self.initial_exploration_rate # 1
        self.final_exploration_rate = final_exploration_rate
        self.final_exploration_step = final_exploration_step
        self.adam_epsilon = adam_epsilon
        self.logging = logging
        if callable(loss):
            self.loss = loss
        else:
            try:
                self.loss = {'huber': F.smooth_l1_loss, 'mse': F.mse_loss}[loss]
            except KeyError:
                raise ValueError("loss must be 'huber', 'mse' or a callable")

        if type(envs)!=list:
            envs = [envs]
        self.envs = envs # 这里envs应该是只有一个VertexSystemUnbiased对象的list
        # acting_in_reversible_vertex_env的值就是reversible_vertices的值，这里是True
        self.env, self.acting_in_reversible_vertex_env  = self.get_random_env() # 没传envs给函数

        self.replay_buffers = {}
        # 如果envs里有多个env的话，这个集合里就是所有env的顶点数
        # replay_buffers会根据不同的顶点数创建对应的ReplayBuffer对象
        for n_vertices in set([env.action_space.n for env in self.envs]):
            self.replay_buffers[n_vertices] = ReplayBuffer(self.replay_buffer_size)

        self.replay_buffer = self.get_replay_buffer_for_env(self.env)

        self.seed = random.randint(0, 1e6) if seed is None else seed

        for env in self.envs:
            set_global_seed(self.seed, env)

        self.network = network().to(self.device) # 执行函数产生一个MPNN对象
        self.init_network_params = init_network_params
        self.init_weight_std = init_weight_std
        if self.init_network_params != None:
            print("Pre-loading network parameters from {}.\n".format(init_network_params))
            self.load(init_network_params)
        else:
            if self.init_weight_std != None:
                def init_weights(m):
                    if type(m) == torch.nn.Linear:
                        print("Setting weights for", m)
                        m.weight.normal_(0, init_weight_std)
                with torch.no_grad():
                    self.network.apply(init_weights)

        self.target_network = network().to(self.device) # target_network也是一个MPNN
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.initial_learning_rate, eps=self.adam_epsilon,
                                    weight_decay=self.weight_decay)

        self.evaluate = evaluate
        if test_envs in [None,[None]]:
            # By default, test on the same environment(s) as are trained on.
            self.test_envs = self.envs
        else:
            if type(test_envs) != list:
                test_envs = [test_envs]
            self.test_envs = test_envs
        self.test_episodes = int(test_episodes)
        self.test_frequency = test_frequency
        self.test_save_path = test_save_path
        self.test_metric = test_metric

        self.losses_save_path = os.path.join(os.path.split(self.test_save_path)[0], "losses.pkl")

        if not self.acting_in_reversible_vertex_env:
            for env in self.envs:
                assert env.extra_action==ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."
            for env in self.test_envs:
                assert env.extra_action==ExtraAction.NONE, "For deterministic MDP, no extra action is allowed."

        self.allowed_action_state = self.env.get_allowed_action_states()

        self.save_network_frequency = save_network_frequency
        self.network_save_path = network_save_path

    def get_random_env(self, envs=None):
        '''
        从envs里随机采样k个元素，env是这里面的第0个元素
        当k=1时这里的env就是从envs随机选择的1个元素
        返回env和一个表示顶点是否可以重复选择的布尔变量
        '''
        if envs is None:
            env = random.sample(self.envs, k=1)[0]
        else:
            env = random.sample(envs, k=1)[0]

        return env, env.reversible_vertices

    def get_replay_buffer_for_env(self, env):
        return self.replay_buffers[env.action_space.n]

    def get_random_replay_buffer(self):
        return random.sample(self.replay_buffers.items(), k=1)[0][1]

    def learn(self, timesteps, verbose=False):

        if self.logging:
            logger = Logger()

        # Initialise the state
        # 假设graph有n个顶点
        # 这里的state的形状是(7+n)xn
        state = torch.as_tensor(self.env.reset()) # self.env的方法都是verticesystem.py里去找
        # 从这里开始state[:8,:]的取值跟SIGNED和BINARY约定的一致了
        
        score = 0
        losses_eps = []
        t1 = time.time()

        test_scores = []
        losses = [] # 这是总的losses

        is_training_ready = False

        for timestep in range(timesteps):

            if not is_training_ready:
                # 这里是判断replay_buffer里存储的经验是不是超过了replay_start_size
                if all([len(rb)>=self.replay_start_size for rb in self.replay_buffers.values()]):
                    print('\nAll buffers have {} transitions stored - training is starting!\n'.format(
                        self.replay_start_size))
                    # 只有当经验缓冲区里有足够的经验才会开始训练
                    is_training_ready=True

            # Choose action
            # action就是选择的顶点
            action = self.act(state.to(self.device).float(), is_training_ready=is_training_ready)

            # Update epsilon
            if self.update_exploration:
                self.update_epsilon(timestep)

            # Update learning rate
            if self.update_learning_rate: # False
                self.update_lr(timestep)

            # Perform action in environment
            # 得到执行当前动作后的状态和奖励
            # state_next的形状是(7+n)xn
            state_next, reward, done, _ = self.env.step(action)

            score += reward # score就是reward累加而成的

            # Store transition in replay buffer
            action = torch.as_tensor([action], dtype=torch.long)
            reward = torch.as_tensor([reward], dtype=torch.float)
            state_next = torch.as_tensor(state_next)

            done = torch.as_tensor([done], dtype=torch.float)

            # 缓冲区里的经验
            self.replay_buffer.add(state, action, reward, state_next, done)

            if done: # 一个episode完成
                # Reinitialise the state
                if verbose: # True
                    loss_str = "{:.2e}".format(np.mean(losses_eps)) if is_training_ready else "N/A"
                    print("timestep : {}, episode time: {}, score : {}, mean loss: {}, time : {} s".format(
                        (timestep+1),
                         self.env.current_step,
                         np.round(score,3),
                         loss_str,
                         round(time.time() - t1, 3)))

                if self.logging: # False
                    logger.add_scalar('Episode_score', score, timestep)
                self.env, self.acting_in_reversible_vertex_env = self.get_random_env()
                self.replay_buffer = self.get_replay_buffer_for_env(self.env)
                state = torch.as_tensor(self.env.reset())
                score = 0
                losses_eps = [] # 这是一个episode中的losses
                t1 = time.time()

            else:
                state = state_next

            if is_training_ready:

                # Update the main network
                if timestep % self.update_frequency == 0:

                    # Sample a batch of transitions
                    transitions = self.get_random_replay_buffer().sample(self.minibatch_size, self.device)

                    # Train on selected batch
                    loss = self.train_step(transitions)
                    losses.append([timestep,loss])
                    losses_eps.append(loss)

                    if self.logging: # False
                        logger.add_scalar('Loss', loss, timestep)

                # Periodically update target network
                if timestep % self.update_target_frequency == 0:
                    self.target_network.load_state_dict(self.network.state_dict())

            # evaluate agent
            if (timestep+1) % self.test_frequency == 0 and self.evaluate and is_training_ready:
                test_score = self.evaluate_agent()
                print('\nTest score: {}\n'.format(np.round(test_score,3)))

                if self.test_metric in [TestMetric.FINAL_CDS,TestMetric.MAX_CDS,TestMetric.CUMULATIVE_REWARD]:
                    best_network = all([test_score > score for t,score in test_scores])
                elif self.test_metric in [TestMetric.ENERGY_ERROR, TestMetric.BEST_ENERGY]:
                    best_network = all([test_score < score for t, score in test_scores])
                else:
                    raise NotImplementedError("{} is not a recognised TestMetric".format(self.test_metric))

                if best_network:
                    path = self.network_save_path
                    path_main, path_ext = os.path.splitext(path)
                    path_main += "_best"
                    if path_ext == '':
                        path_ext += '.pth'
                    self.save(path_main + path_ext)

                test_scores.append([timestep+1,test_score]) # 每一个元素都是一个list

            if (timestep + 1) % self.save_network_frequency == 0 and is_training_ready:
                path = self.network_save_path
                path_main, path_ext = os.path.splitext(path)
                path_main += str(timestep+1)
                if path_ext == '':
                    path_ext += '.pth'
                self.save(path_main+path_ext)
            '''
            顶点不能重复选择的情况下，如果最后timestep超过了timesteps也没有完成最后一次，会不会导致出问题？
            '''
            # if timestep + len(state[0]) > timesteps:
            #     break

        if self.logging:
            logger.save()

        path = self.test_save_path
        if os.path.splitext(path)[-1] == '':
            path += '.pkl'

        with open(path, 'wb+') as output:
            pickle.dump(np.array(test_scores), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('test_scores saved to {}'.format(path))

        with open(self.losses_save_path, 'wb+') as output:
            pickle.dump(np.array(losses), output, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print('losses saved to {}'.format(self.losses_save_path))


    @torch.no_grad()
    def __only_bad_actions_allowed(self, state, network):
        x = (state[0, :] == self.allowed_action_state).nonzero()
        q_next = network(state.to(self.device).float())[x].max()
        return True if q_next < 0 else False

    def train_step(self, transitions):
        '''
        transitions包含若干条经验
        这里的states是batchsize x (7+n) x n的
        返回loss
        '''

        states, actions, rewards, states_next, dones = transitions

        if self.acting_in_reversible_vertex_env:
            # Calculate target Q
            with torch.no_grad():
                if self.double_dqn: # True
                    # 保持维度不变，greedy_action的形状应该是batchsize x 1
                    # 得到一个batch里每一个state_next下使得Q值最大的action
                    greedy_actions = self.network(states_next.float()).argmax(1, True)
                    # target_network(states_next.float())得到每一个状态对应的Q值
                    # 得到greedy_actions对应的Q值向量，形状也是batchsize x 1
                    q_value_target = self.target_network(states_next.float()).gather(1, greedy_actions)
                else:
                    q_value_target = self.target_network(states_next.float()).max(1, True)[0]

        else:
            target_preds = self.target_network(states_next.float())
            # disallowed_actions_mask形状是batchsize x 1 x n的，表示所有不能选择的点
            disallowed_actions_mask = (states_next[:, 0, :] != self.allowed_action_state)
            # Calculate target Q, selecting ONLY ALLOWED ACTIONS greedily.
            with torch.no_grad():
                if self.double_dqn:
                    network_preds = self.network(states_next.float())
                    # Set the Q-value of disallowed actions to a large negative number (-10000) so they are not selected.
                    # 对于不能选择的点在MPNN预测Q值的时候是不处理的
                    # 算完之后再对那些点赋一个大负数表示不能选择
                    # 这样就不用在计算的时候去处理了，比较快
                    network_preds_allowed = network_preds.masked_fill(disallowed_actions_mask,-10000)
                    greedy_actions = network_preds_allowed.argmax(1, True)
                    q_value_target = target_preds.gather(1, greedy_actions)
                else:
                    q_value_target = target_preds.masked_fill(disallowed_actions_mask,-10000).max(1, True)[0]

        if self.clip_Q_targets: # False
            q_value_target[q_value_target < 0] = 0

        # Calculate TD target
        # 时序差分对象，R + \gamma * next state Q value
        td_target = rewards + (1 - dones) * self.gamma * q_value_target

        # Calculate Q value
        # 形状是batchsize x 1 
        q_value = self.network(states.float()).gather(1, actions)

        # Calculate loss
        loss = self.loss(q_value, td_target, reduction='mean') # loss是均方损失

        # Update weights
        self.optimizer.zero_grad()
        loss.backward() # 反向传播更新网络参数

        if self.max_grad_norm is not None: #Optional gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return loss.item()

    def act(self, state, is_training_ready=True):
        '''
        这里采用了epsilon贪心
        150000time_step后epsilon才为0.05，之前epsilon从1逐渐减小到0.05
        所以其实刚开始训练的时候基本都是随机选一个点，后面才是epsilon贪心
        在1-epsilon时，action是使得Q值最大的动作，就是选使得Q值最大的那个点
        在epsilon时:
        如果顶点可以重复选择，那么就从所有的顶点里随机选择一个顶点
        否则，
        如果是BINARY，取值0或1时，找到所有标记为0的顶点，从中随机选择一个作为action
        如果时SIGNED，取值-1或者1时，找到所有标记为1的顶点，从中随机选择一个作为action
        这里要注意，由于CDS算的是两个集合之间的边的总权值，把哪个集合看成是S其实都一样，算出来的CDS是一样的
        所以这个代码，BINARY的时候把标记为0的顶点看成是可选的，SIGNED是把标记为1的顶点看成是可选的
        返回值是个标量，代表选哪个顶点
        '''
        if is_training_ready and random.uniform(0, 1) >= self.epsilon: # 一开始epsilon=1
            # Action that maximises Q function
            action = self.predict(state)
        else:
            if self.acting_in_reversible_vertex_env: # 我们这里是True
                # Random random vertex.
                action = np.random.randint(0, self.env.action_space.n)
            else:
                # Flip random vertex from that hasn't yet been flipped.
                x = (state[0, :] == self.allowed_action_state).nonzero() # x中包含所有满足条件的顶点
                action = x[np.random.randint(0, len(x))].item() # 从所有满足条件的顶点里随机选一个顶点
        return action

    def update_epsilon(self, timestep):
        '''
        更新epsilon，每次减小一点点，最小为final_exploration_rate
        一开始是1，最后是0.05
        '''
        eps = self.initial_exploration_rate - (self.initial_exploration_rate - self.final_exploration_rate) * (
            timestep / self.final_exploration_step
        )
        self.epsilon = max(eps, self.final_exploration_rate)

    def update_lr(self, timestep):
        if timestep <= self.peak_learning_rate_step:
            lr = self.initial_learning_rate - (self.initial_learning_rate - self.peak_learning_rate) * (
                    timestep / self.peak_learning_rate_step
                )
        elif timestep <= self.final_learning_rate_step:
            lr = self.peak_learning_rate - (self.peak_learning_rate - self.final_learning_rate) * (
                    (timestep - self.peak_learning_rate_step) / (self.final_learning_rate_step - self.peak_learning_rate_step)
                )
        else:
            lr = None

        if lr is not None:
            for g in self.optimizer.param_groups:
                g['lr'] = lr


    @torch.no_grad()
    def predict(self, states, acting_in_reversible_vertex_env=None):
        '''
        states是(7+n)xn的矩阵
        qs是MPNN网络的输出
        具体来说，qs是一个长为n的向量，代表了选择每个顶点后产生的Q值
        从这里可以看出来MPNN压缩了所有的行，列数不变
        返回action，代表选哪个顶点可以使得Q值最大
        '''

        if acting_in_reversible_vertex_env is None:
            acting_in_reversible_vertex_env = self.acting_in_reversible_vertex_env # True

        qs = self.network(states) # network是MPNN，把states喂给MPNN

        if acting_in_reversible_vertex_env: # 每个顶点可以重复操作
            if qs.dim() == 1: # 在这个例子里dim应该只等于1
                actions = qs.argmax().item() 
            else:
                actions = qs.argmax(1, True).squeeze(1).cpu().numpy()
            return actions
        else:
            if qs.dim() == 1:
                x = (states[0, :] == self.allowed_action_state).nonzero()
                actions = x[qs[x].argmax().item()].item()
            else:
                disallowed_actions_mask = (states[:, :, 0] != self.allowed_action_state)
                qs_allowed = qs.masked_fill(disallowed_actions_mask, -10000)
                actions = qs_allowed.argmax(1, True).squeeze(1).cpu().numpy()
            return actions

    @torch.no_grad()
    def evaluate_agent(self, batch_size=None):
        '''
        Evaluates agent's current performance.  Run multiple evaluations at once
        so the network predictions can be done in batches.
        返回测试的平均分数
        '''
        if batch_size is None: # None
            batch_size = self.minibatch_size

        i_test = 0
        i_comp = 0
        test_scores = []
        batch_scores = [0]*batch_size # 长为batchsize的向量

        test_envs = np.array([None]*batch_size)
        obs_batch = []

        while i_comp < self.test_episodes: # 50，也就是测试集是50张图

            for i, env in enumerate(test_envs):
                if env is None and i_test < self.test_episodes: # 50
                    test_env, testing_in_reversible_vertex_env = self.get_random_env(self.test_envs)
                    obs = test_env.reset() # obs的形状是(7+n, n)
                    test_env = deepcopy(test_env)

                    test_envs[i] = test_env # 创建了50个test_env的list
                    obs_batch.append(obs)

                    i_test += 1

            actions = self.predict(torch.FloatTensor(np.array(obs_batch)).to(self.device),
                                   testing_in_reversible_vertex_env)

            obs_batch = []

            i = 0
            for env, action in zip(test_envs, actions):

                if env is not None:
                    obs, rew, done, info = env.step(action)

                    if self.test_metric == TestMetric.CUMULATIVE_REWARD:
                        batch_scores[i] += rew

                    if done:
                        if self.test_metric == TestMetric.BEST_ENERGY:
                            batch_scores[i] = env.best_energy
                        elif self.test_metric == TestMetric.ENERGY_ERROR:
                            batch_scores[i] = abs(env.best_energy - env.calculate_best()[0])
                        elif self.test_metric == TestMetric.MAX_CDS:
                            batch_scores[i] = env.get_best_cds()
                        elif self.test_metric == TestMetric.FINAL_CDS:
                            batch_scores[i] = env.calculate_cds()

                        test_scores.append(batch_scores[i])

                        if self.test_metric == TestMetric.CUMULATIVE_REWARD:
                            batch_scores[i] = 0

                        i_comp += 1
                        test_envs[i] = None
                    else:
                        obs_batch.append(obs)

                i += 1

        if self.test_metric == TestMetric.ENERGY_ERROR:
            print("\n{}/{} graphs solved optimally".format(np.count_nonzero(np.array(test_scores)==0),self.test_episodes), end="")

        return np.mean(test_scores)

    def save(self, path='network.pth'):
        if os.path.splitext(path)[-1]=='':
            path + '.pth'
        torch.save(self.network.state_dict(), path)

    def load(self,path):
        self.network.load_state_dict(torch.load(path,map_location=self.device))