import copy
import logging

import math
import os
import pickle
import random
from collections import deque

import numpy as np
from torch.distributions import Categorical
import torch
import torch.nn as nn
import torch.optim as optim

from Agent_model import AgentNetwork
from QMIXNet_model import QMIXNet
from multi_agent_environment import MultiAgentGraphEnvironment
import torch.nn.functional as F


class QMixAgent:
    def __init__(self, learning_rate, experience, discount_factor, model_update_frequency,
                 batch_size, config, device='cuda:1'):
        """
        初始化 QMIX Agent。

        参数:
        - state_dim1, state_dim2: 不同模型类型的状态输入维度。
        - learning_rate: 优化器的学习率。
        - experience: 经验回放缓冲区的最大长度。
        - discount_factor: 未来奖励的折扣因子。
        - agent_number: 环境中 Agent 的数量。
        - model_update_frequency: 目标 QMIX 网络更新的频率（以回合数计）。
        - batch_size: 每个训练批次的样本数量。
        - muti_step: 多步学习的步数。
        - device: 使用的设备（'cpu' 或 'cuda'）。
        """
        self.node_num = config['node_num']
        self.edge_num = config['edge_num']
        self.max_steps = config['max_steps']
        self.action_size = config['action_size']
        self.n_agents = config['n_agents']

        self.hyperparameters = config['hyperparameters']
        self.num_episodes_to_run = config['num_episodes_to_run']
        self.gradient_clipping = self.hyperparameters.get("gradient_clipping", True)
        # 整个工程路径
        self.root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.all_data = pickle.load(open(f'{self.root_path}/data/RL_data.pkl', 'rb'))

        self.qmix_hidden_units = config['qmix_hidden_units']
        self.experience = deque(maxlen=experience)
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.model_update_frequency = model_update_frequency
        self.batch_size = batch_size
        self.device = device
        self.rewards = []
        self.steps = []

        self.episode_number = 0

        # 初始化全局 QMIX 网络和目标网络
        self.qmix = self._build_qmix().to(self.device)
        self.target_qmix = self._build_qmix().to(self.device)
        self.target_qmix.load_state_dict(self.qmix.state_dict())  # 初始化目标网络权重

        # 为每个 Agent 初始化 Q 网络
        self.q_networks = [self._build_q_network().to(self.device) for _ in range(self.n_agents)]
        self.target_q_networks = [self._build_q_network().to(self.device) for _ in range(self.n_agents)]
        if config['loading_model']:
            self.load()

        # 优化器：QMIX 和所有 Q 网络的参数都会被优化
        qmix_params = list(self.qmix.parameters())
        qnet_params = []
        for qnet in self.q_networks:
            qnet_params += list(qnet.parameters())
        self.optimizer = optim.Adam(qmix_params + qnet_params, lr=learning_rate)
        # 损失函数
        self.loss_fn = nn.SmoothL1Loss()


    def episode_reset(self, tasks):
        """
        重置 Agent 的状态。
        """
        self.possible_actions = {}
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_reward = []
        self.episode_step = 0
        self.episode_loss = 0
        self.environment = MultiAgentGraphEnvironment(agent_ids=range(self.n_agents),
                                                      start_sides=config['start_sides'],
                                                      max_steps=self.max_steps, tasks=tasks,
                                                      all_data=self.all_data
                                                      )
        # if os.path.exists('TrainedModel/Mixing.pth'):
        #     self.qmix.load_state_dict(torch.load('TrainedModel/Mixing.pth'))
        # for i, qnet in enumerate(self.q_networks):
        #     if os.path.exists(f'TrainedModel/agent{i}.pth'):
        #         qnet.load_state_dict(torch.load(f'TrainedModel/agent{i}.pth'))
        # self.target_qmix.load_state_dict(self.qmix.state_dict())

    def _build_qmix(self):
        return QMIXNet(n_agents=self.n_agents, qmix_hidden_units=self.qmix_hidden_units, global_state_dim=self.n_agents * 214)

    def _build_q_network(self):
        return AgentNetwork(self.node_num, self.edge_num, self.action_size, max_steps=self.max_steps)

    def get_action(self, state, agent_id):
        """
        使用 epsilon-greedy 策略选择动作。

        参数:
        - epsilon: 选择随机动作的概率。
        - state: 当前 Agent 的状态。
        - agent_id: Agent 的 ID。

        返回:
        - action: 选择的动作。
        """
        current_node = torch.tensor(state[agent_id]['current_node']).long().unsqueeze(0).to(self.device)
        edges_traversed = state[agent_id]['visited_edges']
        edges_traversed_ = edges_traversed + [19] * (self.max_steps - len(edges_traversed))
        edges_traversed_ = torch.tensor([int(item) for item in edges_traversed_]).long().unsqueeze(0).to(self.device)
        rss = torch.tensor(state[agent_id]['rss_data']).long().unsqueeze(0).to(self.device)
        # 获取可能的动作
        possible_actions = self.environment.get_possible_actions(agent_id)
        possible_actions = torch.tensor(possible_actions, dtype=torch.long)

        # 创建动作掩码
        action_mask = torch.zeros(self.action_size, dtype=torch.float32).to(self.device)
        action_mask[possible_actions - 1] = 1  # 动作索引从1开始

        # 获取 Q 值
        q_values = self.q_networks[int(agent_id)](current_node, edges_traversed_, rss)
        # 获得q_values的最小值，并将mask中为0的位置的值设置为比最小值小5个数量级
        q_values = q_values.masked_fill(action_mask == 0, q_values.min() - 1e5)
        q_values = F.softmax(q_values/1.5, dim=1)

        epsilon = self.exponential_epsilon_decay()
        uniform_prob = epsilon / len(possible_actions)
        adjusted_probs = (1 - epsilon) * q_values + uniform_prob * action_mask
        adjusted_probs = adjusted_probs / adjusted_probs.sum()  # 归一化概率

        # 创建动作分布并采样动作
        action_distribution = Categorical(adjusted_probs)
        self.possible_actions[agent_id] = adjusted_probs
        action = action_distribution.sample().item()

        if action != 26:
            self.environment.pre_select.append(str(int(action) + 1))

        return int(action)

    def storage_experience(self, state, action, reward, next_state, done, stop_flag, step):
        # 将state, action, reward, next_state, done转化为数组
        for i in range(self.n_agents):
            state[str(i)]['visited_edges'] = [int(item) for item in state[str(i)]['visited_edges']]
            next_state[str(i)]['visited_edges'] = [int(item) for item in next_state[str(i)]['visited_edges']]
            state[str(i)]['visited_edges'] = state[str(i)]['visited_edges'][1:] + [0] * (self.max_steps + 1 - len(state[str(i)]['visited_edges']))
            next_state[str(i)]['visited_edges'] = next_state[str(i)]['visited_edges'][1:] + [0] * (self.max_steps + 1 - len(next_state[str(i)]['visited_edges']))

        state_current_node = [[state[str(i)]['current_node']] for i in range(self.n_agents)]
        state_visited_edges = [state[str(i)]['visited_edges'] for i in range(self.n_agents)]
        state_rss_data = [state[str(i)]['rss_data'] for i in range(self.n_agents)]
        next_state_current_node = [[next_state[str(i)]['current_node']] for i in range(self.n_agents)]
        next_state_visited_edges = [next_state[str(i)]['visited_edges'] for i in range(self.n_agents)]
        next_state_rss_data = [next_state[str(i)]['rss_data'] for i in range(self.n_agents)]

        action = [[action[i]] for i in range(self.n_agents)]
        reward = [[reward[i]] for i in range(self.n_agents)]
        done = [[int(done)] for _ in range(self.n_agents)]
        stop_flag = [[int(stop_flag)] for _ in range(self.n_agents)]
        self.episode_buffer['s_current_node'][step-1] = state_current_node
        self.episode_buffer['s_visited_edges'][step-1] = state_visited_edges
        self.episode_buffer['s_rss_data'][step-1] = state_rss_data

        self.episode_buffer['a'][step-1] = action
        self.episode_buffer['r'][step-1] = reward
        self.episode_buffer['s_current_node_next'][step-1] = next_state_current_node
        self.episode_buffer['s_visited_edges_next'][step-1] = next_state_visited_edges
        self.episode_buffer['s_rss_data_next'][step-1] = next_state_rss_data
        self.episode_buffer['done'][step-1] = done
        self.episode_buffer['stop_flag'][step-1] = stop_flag
        if done[0][0] == 1:
            self.experience.append(copy.deepcopy(self.episode_buffer))

    def reset_episode_buffer(self):
        """
        初始化（或重置）episode_buffer，用于存储一条完整序列（episode）中的数据。
        """
        self.episode_buffer = {'s_current_node': np.zeros((self.max_steps, self.n_agents, 1), dtype=np.float32),
                               's_visited_edges': np.zeros((self.max_steps, self.n_agents, self.max_steps), dtype=np.float32),
                               's_rss_data': np.zeros((self.max_steps, self.n_agents, 7, 30), dtype=np.float32),
                               'a': np.zeros((self.max_steps, self.n_agents, 1), dtype=np.float32),
                               'r': np.zeros((self.max_steps, self.n_agents, 1), dtype=np.float32),
                               's_current_node_next': np.zeros((self.max_steps, self.n_agents, 1), dtype=np.float32),
                               's_visited_edges_next': np.zeros((self.max_steps, self.n_agents, self.max_steps), dtype=np.float32),
                               's_rss_data_next': np.zeros((self.max_steps, self.n_agents, 7, 30), dtype=np.float32),
                               'done': np.zeros((self.max_steps, self.n_agents, 1), dtype=np.float32),
                               'stop_flag': np.zeros((self.max_steps, self.n_agents, 1), dtype=np.float32)}

    def update_target_model(self, episode, tau=0.2):
        if episode % self.model_update_frequency == 0:
            with torch.no_grad():
                for p, tp in zip(self.qmix.parameters(), self.target_qmix.parameters()):
                    tp.data.mul_(1 - tau).add_(tau * p.data)
                for net, tnet in zip(self.q_networks, self.target_q_networks):
                    for p, tp in zip(net.parameters(), tnet.parameters()):
                        tp.data.mul_(1 - tau).add_(tau * p.data)

    def save(self, path='./Trained_Model'):
        """
        保存训练好的模型权重。

        参数:
        - path: 保存模型的目录路径。
        """
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory {path} created.")
        for i, qnet in enumerate(self.q_networks):
            torch.save(qnet.state_dict(), os.path.join(path, f'agent{i}.pth'))
            print(f"Model of agent {i} saved.")
        torch.save(self.qmix.state_dict(), os.path.join(path, 'Mixing.pth'))
        print("Model of QMIX saved.")

    def load(self, path='./Trained_Model'):
        """
        加载训练好的模型权重。

        参数:
        - path: 加载模型的目录路径。
        """
        for i, qnet in enumerate(self.q_networks):
            qnet.load_state_dict(torch.load(os.path.join(path, f'agent{i}.pth'), map_location=self.device))
        self.qmix.load_state_dict(torch.load(os.path.join(path, 'Mixing.pth'), map_location=self.device))

    def sample_and_construct_batch(self):
        """
        从经验池中随机抽样 batch_size 个 episode，同时构建局部输入和全局状态信息。
        返回：
          一个字典，包含以下键：
             'inputs': 当前局部状态输入，形状 (batch_size, max_steps, n_agents, input_dim)
             'global_inputs': 当前全局状态，形状 (batch_size, max_steps, n_agents * input_dim)
             'next_inputs': 下一时刻局部状态输入，形状 (batch_size, max_steps, n_agents, input_dim)
             'global_next_inputs': 下一时刻全局状态，形状 (batch_size, max_steps, n_agents * input_dim)
             'actions': (batch_size, max_steps, n_agents, 1)
             'rewards': (batch_size, max_steps, n_agents, 1)
             'done': (batch_size, max_steps, n_agents, 1)
        """
        # 改为连续随机采样
        sampled_episodes = random.sample(self.experience, self.batch_size)
        # start_index = random.randint(0, len(self.experience) - self.batch_size)
        # sampled_episodes = [self.experience[i] for i in range(start_index, start_index + self.batch_size)]

        batch_local_inputs = []  # 局部输入数据
        batch_global_inputs = []  # 当前全局状态数据
        batch_next_local_inputs = []  # 下一状态局部输入
        batch_next_global_inputs = []  # 下一状态全局状态
        batch_actions = []  # 动作
        batch_rewards = []  # 奖励
        batch_done = []  # 终止标志

        for ep in sampled_episodes:
            # 假设 n_agents 可以从 s_current_node 的 shape 中获取
            n_agents = ep['s_current_node'].shape[1]

            # --- 构建当前状态局部输入 ---
            # s_rss_data 的形状 (max_steps, n_agents, 7, 30) 展平为 (max_steps, n_agents, 210)
            rss_flat = ep['s_rss_data'].reshape(self.max_steps, n_agents, -1)
            # 拼接各部分特征，沿最后一维连接，得到形状 (max_steps, n_agents, 214)
            local_input = np.concatenate([ep['s_current_node'], ep['s_visited_edges'], rss_flat], axis=-1)
            batch_local_inputs.append(local_input)
            # 全局状态：在每个时间步上将各智能体的局部输入拼接成一个整体向量
            # 形状变为 (max_steps, n_agents*214)
            global_input = local_input.reshape(self.max_steps, -1)
            batch_global_inputs.append(global_input)

            # --- 构建下一时刻状态输入 ---
            rss_flat_next = ep['s_rss_data_next'].reshape(self.max_steps, n_agents, -1)
            next_local_input = np.concatenate([ep['s_current_node_next'], ep['s_visited_edges_next'], rss_flat_next],
                                              axis=-1
                                              )
            batch_next_local_inputs.append(next_local_input)
            global_next_input = next_local_input.reshape(self.max_steps, -1)
            batch_next_global_inputs.append(global_next_input)

            # --- 动作、奖励、done ---
            batch_actions.append(ep['a'])  # (max_steps, n_agents, 1)
            batch_rewards.append(ep['r'])  # (max_steps, n_agents, 1)
            batch_done.append(ep['done'])  # (max_steps, n_agents, 1)

        # 将各列表堆叠为 numpy 数组，并增加 batch 维度
        batch_local_inputs = np.stack(batch_local_inputs, axis=0)  # (batch_size, max_steps, n_agents, 214)
        batch_global_inputs = np.stack(batch_global_inputs, axis=0)  # (batch_size, max_steps, n_agents*214)
        batch_next_local_inputs = np.stack(batch_next_local_inputs, axis=0)  # (batch_size, max_steps, n_agents, 214)
        batch_next_global_inputs = np.stack(batch_next_global_inputs, axis=0)  # (batch_size, max_steps, n_agents*214)
        batch_actions = np.stack(batch_actions, axis=0)
        batch_rewards = np.stack(batch_rewards, axis=0)
        batch_done = np.stack(batch_done, axis=0)

        # 转换为 torch.Tensor
        batch_local_inputs = torch.tensor(batch_local_inputs, dtype=torch.float32)
        batch_global_inputs = torch.tensor(batch_global_inputs, dtype=torch.float32)
        batch_next_local_inputs = torch.tensor(batch_next_local_inputs, dtype=torch.float32)
        batch_next_global_inputs = torch.tensor(batch_next_global_inputs, dtype=torch.float32)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        batch_done = torch.tensor(batch_done, dtype=torch.float32)

        return {'inputs': batch_local_inputs,  # (batch_size, max_steps, n_agents, 214)
            'global_inputs': batch_global_inputs,  # (batch_size, max_steps, n_agents*214)
            'next_inputs': batch_next_local_inputs,  # (batch_size, max_steps, n_agents, 214)
            'global_next_inputs': batch_next_global_inputs,  # (batch_size, max_steps, n_agents*214)
            'actions': batch_actions,  # (batch_size, max_steps, n_agents, 1)
            'rewards': batch_rewards,  # (batch_size, max_steps, n_agents, 1)
            'done': batch_done  # (batch_size, max_steps, n_agents, 1)
        }

    def train_agent(self):
        """
        从经验回放中采样一个批次 episode 数据，并使用整个 episode 数据进行一次整体 QMIX 更新。
        """
        # 如果经验不足，则不进行训练
        if len(self.experience) < self.batch_size:
            return 0

        # 构建输入数据，数据格式按照上述格式组织
        input_data = self.sample_and_construct_batch()

        # 清零梯度
        self.optimizer.zero_grad()

        # 存储所有时间步各智能体的局部 Q 值（当前网络和目标网络）
        q_evals_list = []
        q_targets_list = []

        # 遍历整个 episode 的每个时间步
        for step in range(self.max_steps):
            q_eval_agents = []  # 当前时刻各智能体的 Q 值（当前网络）
            q_target_agents = []  # 下一状态下各智能体的 Q 值（目标网络）
            for i in range(self.n_agents):
                # 获取当前局部输入（例如 214 维向量）
                obs = input_data['inputs'][:, step, i, :].to(self.device)
                # 获取下一状态对应的局部输入
                next_obs = input_data['next_inputs'][:, step, i, :].to(self.device)
                # 计算当前状态下 Q 网络的输出（形状: [batch_size, action_dim]）
                q_eval = self.q_networks[i](obs[:, :1].squeeze(1), obs[:, 1:4], obs[:, 4:].reshape(-1, 7, 30))
                # 计算下一状态下目标网络的输出
                q_target = self.target_q_networks[i](next_obs[:, :1].squeeze(1), next_obs[:, 1:4], next_obs[:, 4:].reshape(-1, 7, 30))
                q_eval_agents.append(q_eval)
                q_target_agents.append(q_target)
            # 将各智能体的 Q 值堆叠，得到形状 (batch_size, n_agents, action_dim)
            q_eval_agents = torch.stack(q_eval_agents, dim=1)
            q_target_agents = torch.stack(q_target_agents, dim=1)

            # 从当前网络输出中，根据实际动作采样 Q 值
            # actions 的 shape 为 (batch_size, max_steps, n_agents, 1)，这里取出当前时间步的动作
            actions = input_data['actions'][:, step, :, :].to(self.device)
            # 对每个智能体取出所执行动作对应的 Q 值，结果 shape 为 (batch_size, n_agents)
            q_eval_chosen = torch.gather(q_eval_agents, dim=2, index=actions).squeeze(-1)

            # 对目标网络输出，选取每个智能体的最大 Q 值
            with torch.no_grad():
                q_target_max, _ = q_target_agents.max(dim=2)  # (batch_size, n_agents)

            # 保存每个时间步各智能体的 Q 值
            q_evals_list.append(q_eval_chosen)
            q_targets_list.append(q_target_max)

        # 堆叠所有时间步，得到形状 (batch_size, max_steps, n_agents)
        q_evals = torch.stack(q_evals_list, dim=1)
        q_targets = torch.stack(q_targets_list, dim=1)

        # --- 使用混合网络计算全局 Q 值 ---
        # 当前全局状态：使用 'global_inputs'，形状 (batch_size, max_steps, n_agents*214)
        global_states = input_data['global_inputs'].to(self.device)
        # 下一状态全局状态：使用 'global_next_inputs'，形状 (batch_size, max_steps, n_agents*214)
        global_next_states = input_data['global_next_inputs'].to(self.device)
        # 计算全局 Q 值，混合网络接受局部 Q 值 (batch_size, max_steps, n_agents) 与全局状态
        q_total_eval = self.qmix(q_evals, global_states)  # (batch_size, max_steps, 1)
        q_total_target = self.target_qmix(q_targets, global_next_states)  # (batch_size, max_steps, 1)

        # --- 构造 TD Target 与损失计算 ---
        # 奖励和 done 的 shape 为 (batch_size, max_steps, n_agents, 1)
        # 通常 QMIX 使用全局奖励，如果每个智能体的奖励一致，可以选取第 0 个智能体的奖励
        rewards = input_data['rewards'][:, :, 0, :].to(self.device)  # (batch_size, max_steps, 1)
        done = input_data['done'][:, :, 0, :].to(self.device)  # (batch_size, max_steps, 1)
        # TD Target：reward + gamma * (1 - done) * q_total_target

        # targets = rewards + self.discount_factor * (1 - done) * q_total_target


        # r_t + γ * (1-done_t) * q_tot_target_{t+1}

        # loss = self.loss_fn(q_total_eval, targets.detach())

        targets = rewards.clone()  # r_t
        targets[:, :-1, :] += self.discount_factor * (1 - done[:, :-1, :]) * q_total_target[:, 1:, :]

        # --------- ⑥  有效步掩码 (屏蔽填充) ---------------
        cum_done = torch.cumsum(done, dim=1).clamp(max=1)  # done后全1
        valid_m = 1 - cum_done  # 0..last_valid=1
        valid_m[:, 0, :] = 1  # 首步始终有效

        td_err = F.smooth_l1_loss(q_total_eval, targets.detach(), reduction='none')  # (B,T,1)
        loss = (valid_m * td_err).sum() / valid_m.sum().clamp(min=1)


        # 反向传播和参数更新
        loss.backward()

        if self.gradient_clipping:
            gradient_clipping_norm = (self.hyperparameters.get("gradient_clipping_norm", 10))
            torch.nn.utils.clip_grad_norm_(self.qmix.parameters(), max_norm=gradient_clipping_norm)
            for qnet in self.q_networks:
                torch.nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=gradient_clipping_norm)

        self.optimizer.step()
        self.lr_decay()
        # return global_grad_norm
        return loss.item()

    def lr_decay(self):
        # 设置最小学习率
        lr_min = 1e-6
        # 使用余弦退火公式计算当前学习率
        # lr_now 在 [lr_min, self.learning_rate] 之间变化
        lr_now = lr_min + (self.learning_rate - lr_min) * (
                    1 + math.cos(math.pi * self.episode_number / self.num_episodes_to_run)) / 2
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now

    def run_n_episodes(self):
        tasks = []
        for test_id in range(1, 91):
            if self.n_agents == 2:
                tasks.append([[0, 2, test_id], [1, 16, test_id]])
            elif self.n_agents == 3:
                tasks.append([[0, 3, test_id], [1, 15, test_id], [2, 13, test_id]])
            else:
                tasks.append([[0, 2, test_id], [1, 10, test_id], [2, 6, test_id], [3, 11, test_id]])

        for episode in range(self.num_episodes_to_run):
            rewards, steps = [], []
            Loss = 0
            cumulative_reward = 0
            for task in tasks:
                self.episode_reset(task)  # 重置环境
                state = self.environment.reset_episode(tasks=task)
                done = False
                self.reset_episode_buffer()
                while not done:
                    state_copy = copy.deepcopy(state)
                    actions = {i: self.get_action(state_copy, str(i)) for i in range(self.n_agents)}

                    if any(a == 26 for a in actions.values()):
                        actions = {i: 26 for i in range(self.n_agents)}

                    next_state, reward, done, stop_flag = self.environment.step({i: actions[i] + 1 for i in range(self.n_agents)})
                    state = copy.deepcopy(next_state)
                    cumulative_reward += reward[0]
                    self.storage_experience(state_copy, actions, reward, next_state, done, stop_flag, self.environment.current_step)  # 存储经验


                steps.append(self.environment.current_step - 1 if stop_flag else self.environment.current_step)
                rewards.append(np.round(- self.environment.pre_loss, 2))

            for _ in range(5):
                loss = self.train_agent()
                Loss += loss

            self.update_target_model(episode)
            self.rewards.append(np.round(np.average(rewards), 2))
            if self.episode_number % 100 == 0 or self.episode_number == self.num_episodes_to_run - 1:
                print(steps)
                print(rewards)
            self.steps.append(np.round(np.average(steps), 2))
            self.episode_number += 1
            print(f"Episode {episode} completed with average precision: {self.rewards[-1]} and average reward: {np.round(cumulative_reward / 90, 3)} and average steps: {self.steps[-1]} and loss: {np.round(Loss / 5, 2)}")
            logging.info(f"Episode {episode} completed with average precision: {self.rewards[-1]} and average reward: {np.round(cumulative_reward / 90, 3)} and average steps: {self.steps[-1]}")

        self.save()

        print("Training completed.")

    def exponential_epsilon_decay(self):
        """
        指数衰减ε
        """
        epsilon_start = self.hyperparameters['epsilon_start']
        epsilon_min = self.hyperparameters['epsilon_min']
        epsilon_decay = self.hyperparameters['epsilon_decay']
        epsilon = epsilon_min + (epsilon_start - epsilon_min) * math.exp(-1.0 * self.episode_number / epsilon_decay)
        # epsilon = max(epsilon_min, epsilon_start - (epsilon_start - epsilon_min) * self.episode_number / epsilon_decay)
        return epsilon


if __name__ == "__main__":
    # ------------------------------2----------------------------------------
    logging.basicConfig(filename='./training_log.txt', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    config = {'hyperparameters': {'learning_rate': 1e-3,
                                  'gradient_clipping_norm': 10.0, 'gradient_clipping': True,
                                  'discount_rate': 0.95,
                                  'epsilon_start': 1, 'epsilon_min': 0.05, 'epsilon_decay': 500},
              'num_episodes_to_run': 2000,
              'experience': 50000,
              'n_agents': 2,
              'node_num': 19,
              'edge_num': 26,
              'max_steps': 3,
              'action_size': 27,
              'qmix_hidden_units': 512,
              'start_sides': [(1, 2), (6, 16)],
              'loading_model': False
              }

    # 初始化 Agent
    Q_Agent = QMixAgent(learning_rate=config['hyperparameters']['learning_rate'], experience=config['experience'],
                        discount_factor=config['hyperparameters']['discount_rate'], model_update_frequency=2,
                        batch_size=128, device='cuda:0', config=config)
    Q_Agent.run_n_episodes()
    #
    # # ------------------------------3----------------------------------------
    # logging.basicConfig(filename='./training_log.txt', level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s')
    # config = {'hyperparameters': {'learning_rate': 1e-3,
    #                               'gradient_clipping_norm': 10.0, 'gradient_clipping': True,
    #                               'discount_rate': 0.9,
    #                               'epsilon_start': 1, 'epsilon_min': 0.0001, 'epsilon_decay': 400},
    #           'num_episodes_to_run': 2000,
    #           'experience': 20000,
    #           'n_agents': 3,
    #           'node_num': 19,
    #           'edge_num': 26,
    #           'max_steps': 3,
    #           'action_size': 27,
    #           'qmix_hidden_units': 512,
    #           'start_sides': [(2, 3), (9, 15), (12, 13)],
    #           'loading_model': False
    #           }
    #
    # # 初始化 Agent
    # Q_Agent = QMixAgent(learning_rate=config['hyperparameters']['learning_rate'], experience=config['experience'],
    #                     discount_factor=config['hyperparameters']['discount_rate'], model_update_frequency=5,
    #                     batch_size=128, device='cuda:1', config=config)
    # Q_Agent.run_n_episodes()
    # ------------------------------4----------------------------------------

    # logging.basicConfig(filename='./training_log.txt', level=logging.INFO,
    #                     format='%(asctime)s - %(levelname)s - %(message)s'
    #                     )
    # config = {'hyperparameters': {'learning_rate': 1e-3,
    #                               'gradient_clipping_norm': 10.0,
    #                               'gradient_clipping': True,
    #                               'discount_rate': 0.9,
    #                               'epsilon_start': 1,
    #                               'epsilon_min': 0.0001,
    #                               'epsilon_decay': 400},
    #           'num_episodes_to_run': 2000,
    #           'experience': 20000,
    #           'n_agents': 4,
    #           'node_num': 19,
    #           'edge_num': 26,
    #           'max_steps': 3,
    #           'action_size': 27,
    #           'qmix_hidden_units': 512,
    #           'start_sides': [(1, 2), (8, 10), (16, 6), (7, 11)],
    #           'loading_model': False}
    #
    # # 初始化 Agent
    # Q_Agent = QMixAgent(learning_rate=config['hyperparameters']['learning_rate'], experience=config['experience'],
    #                     discount_factor=config['hyperparameters']['discount_rate'], model_update_frequency=5,
    #                     batch_size=128, device='cuda:1', config=config
    #                     )
    # Q_Agent.run_n_episodes()






