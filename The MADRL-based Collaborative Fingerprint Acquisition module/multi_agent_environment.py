import time

import math
import os
import pickle

import numpy as np
import torch
from torch.nn import functional as F
from reinforcement_learning.Config import Config
from classification import my_model, my_model_without_atten, my_model_without_LSTM, my_model_only_LSTM, LITE, WHEN
from classification.Config import PARAMETER_CONFIGURATION
from reinforcement_learning.environmental_data import *


class MultiAgentGraphEnvironment:
    def __init__(self, agent_ids, start_sides, max_steps=6, all_data=None, tasks=None):
        """
        初始化多智能体环境
        :param agent_ids: 智能体的ID列表
        :param start_sides: 每个智能体对应的起始边列表
        :param max_steps: 最大步数限制
        :param all_data: 所有数据的预加载（可选）
        """
        self.graph = Config().graph
        self.edges = Config().edges
        self.max_steps = max_steps
        self.all_data = all_data
        self.agent_ids = sorted(agent_ids)  # 确保智能体按ID排序
        self.start_sides = start_sides
        self.AllHistoricalRecord = {}  # 存储所有智能体的历史记录
        self.reset_episode(tasks=tasks)
        self.pre_select = []

    def reset_episode(self, tasks):
        """重置整个episode，初始化所有智能体的状态"""
        self.current_step = 0
        self.done = False
        self.agents = {}  # 存储每个智能体的状态
        self.visited_edges = set()  # 全局已访问的边
        self.task_queue = []  # 管理任务队列
        self.pre_loss = 0

        # 初始化每个智能体
        for agent_id, start_side in zip(self.agent_ids, self.start_sides):
            self.agents[agent_id] = {'current_node': None, 'visited_edges': [], 'test_num': None, 'preloss': 0,
                'historicalRecord': [], 'rss_data': np.ones(shape=(7, 30)), 'first_edge': start_side}

        for task in tasks:
            self.add_task(*task)
        state = self.reset_task()
        return state

    def add_task(self, agent_id, start_node, test_num):
        """为指定的智能体添加新任务到任务队列"""
        self.task_queue.append({'agent_id': agent_id, 'start_node': start_node, 'test_num': test_num})

    def reset_task(self):
        """重置当前任务"""
        if not self.task_queue:
            self.done = True
            return None
        state = {}
        while self.task_queue:
            self.current_task = self.task_queue.pop(0)
            agent_id = self.current_task['agent_id']
            start_node = self.current_task['start_node']
            test_num = self.current_task['test_num']
            state[f'{agent_id}'] = self.reset(agent_id, start_node, test_num)
        return state

    def reset(self, agent_id, start_node, test_num):
        """重置指定智能体的状态，并执行第一条边"""
        agent = self.agents[agent_id]
        agent['current_node'] = start_node
        agent['test_num'] = test_num
        agent['visited_edges'] = []
        agent['historicalRecord'] = []

        # 执行第一条边
        first_edge = agent['first_edge']
        first_edge = tuple(sorted(first_edge))
        first_edge_id = self.edges[first_edge]
        agent['visited_edges'].append(first_edge_id)
        agent['rss_data'] = self.load_rss_data(agent_id, first_edge_id)
        self.visited_edges.add(first_edge_id)  # 全局已选择的边
        agent['current_node'] = self.get_next_node_from_edge(first_edge, agent_id)
        self.current_step = 0
        self.get_reward()  # 计算不在继续前进的情况下的奖励

        return self.get_state(agent_id)

    def identification_node_pair(self, edge):
        """
        根据边的编号，返回边对应的节点对
        :param edge: 边的编号
        :return: 节点对
        """
        for e, edge_id in self.edges.items():
            if edge_id == str(edge):
                return e
        return

    def get_next_node_from_edge(self, edge, agent_id):
        """根据边找到下一个节点"""
        node1, node2 = edge
        return node2 if node1 == self.agents[agent_id]['current_node'] else node1

    def get_reward(self):
        # 模型加载与预测
        Model = my_model.MixedModel(PARAMETER_CONFIGURATION['input_size'], PARAMETER_CONFIGURATION['num_classes'])
        # Model = WHEN.WHEN(PARAMETER_CONFIGURATION['input_size'], PARAMETER_CONFIGURATION['num_classes'])
        # MaxAcc_Two_Agent.pth  MaxAcc_Two_Agent_LITE.pth  MaxAcc_Two_Agent_WHEN.pth
        # MaxAcc_Two_Agent_without_atten.pth MaxAcc_Two_Agent_without_LSTM.pth MaxAcc_Two_Agent_only_LSTM.pth
        model_path = rf'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/classification/TrainedModel_Single_Side/MaxAcc_Four_Agent.pth'
        Model.load_state_dict(torch.load(model_path, map_location=PARAMETER_CONFIGURATION['device']))
        Model = Model.to(device=PARAMETER_CONFIGURATION['device'])
        AllHistoricalData = merge_multiple_edge_data_dict(self.AllHistoricalRecord)
        _, output = self.return_probability(Model, AllHistoricalData, device=PARAMETER_CONFIGURATION['device'])
        probability = F.softmax(torch.tensor(output) / 1.5, dim=1).numpy()[0]
        self.pre_loss = self.return_position_error(probability)

        # 清空模型
        del Model
        torch.cuda.empty_cache()

        return -self.pre_loss

    def step(self, actions):
        """
        执行动作并返回下一个状态、奖励、是否结束等信息。
        :param actions: 字典，键为agent_id，值为动作（边编号）
        :return: 状态字典，奖励字典，done标志
        """
        if self.done:
            return None, None, self.done
        self.pre_select = []
        stop_flag = False  # 标记是否有智能体选择停止动作
        self.current_step += 1
        # 按照智能体ID顺序处理动作
        for agent_id in self.agent_ids:
            action = actions.get(agent_id, None)
            if action is None:
                print(f"Wrong: action for agent {agent_id} is None.")
                exit(1)

            if action == 27:  # 任意智能体选择停止动作，所有智能体停止
                stop_flag = True
                break

            if str(action) in self.visited_edges:  # 检查边是否已被其他智能体选择
                print(f"Wrong: edge {action} has been visited by another agent.")
                stop_flag = True

            # 处理合法动作
            self._process_action(agent_id, action)


        self.get_reward()  # 计算奖励
        pre_loss_reward = - self.pre_loss

        if stop_flag:
            # 只有在 loss 低时才鼓励停止
            reward = np.round(pre_loss_reward , 2)
            next_state = {str(agent_id): self.get_state(agent_id) for agent_id in self.agent_ids}
            self.done = True
            return next_state, {agent_id: reward for agent_id in self.agent_ids}, self.done, stop_flag

        # 步数惩罚：鼓励智能体尽快完成任务
        if self.current_step >= self.max_steps and not self.done:
            self.done = True
            reward = np.round(pre_loss_reward , 2)
        else:
            self.done = False
            step_penalty = -0.01 * self.current_step
            reward = step_penalty

        rewards = {agent_id: reward for agent_id in self.agent_ids}

        # 获取下一个状态
        next_states = {str(agent_id): self.get_state(agent_id) for agent_id in self.agent_ids}
        return next_states, rewards, self.done, stop_flag

    def _process_action(self, agent_id, action):
        """处理单个智能体的动作"""
        agent = self.agents[agent_id]
        current_node = agent['current_node']

        # 找到对应的边（节点对）
        edge = None
        for e, edge_id in self.edges.items():
            if edge_id == str(action):
                edge = e
                break

        if edge and (current_node in edge):
            next_node = self.get_next_node_from_edge(edge, agent_id)
            # 更新当前状态
            agent['visited_edges'].append(self.edges[edge])
            self.visited_edges.add(str(action))  # 全局已选择的边
            agent['current_node'] = next_node
            agent['rss_data'] = self.load_rss_data(agent_id, action)
            self.AllHistoricalRecord[str(action)] = agent['rss_data']

        else:
            print(f"Wrong action {action} for agent {agent_id} at node {current_node}.")
            exit(1)

    def get_possible_actions(self, agent_id):
        """获取指定智能体的可能动作"""
        agent_id = int(agent_id)
        agent = self.agents[agent_id]
        current_node = agent['current_node']
        possible_edges = []
        for neighbor in self.graph[current_node]:
            edge = tuple(sorted([current_node, neighbor]))
            edge_id = self.edges.get(edge, None)
            if edge_id not in self.visited_edges and edge_id not in self.pre_select:
                possible_edges.append(int(edge_id))  # 返回边的编号
        possible_edges.append(27)  # 停止动作
        return possible_edges

    def load_rss_data(self, agent_id, edge):
        """
        根据现在的历史记录和当前选择可走的边，加载数据
        :param edge: 当前边的编号
        :return: 所有边的累计数据， 当前边的单rss数据
        """
        test_num = int(self.agents[agent_id]['test_num'])
        if test_num > 54:
            random_num = np.random.randint(1, 6)
        else:
            random_num = np.random.choice([1, 10, 19, 28, 37])
            # random_num = np.random.randint(1, 41)
        if self.all_data is None:
            if test_num > 54:
                sub_path = f'test/test_side_{edge}_rp_{test_num - 54}.csv'
            else:
                sub_path = f'rp/side_{edge}_rp_{test_num}.csv'
            present_edge = load_single_dataset(os.path.join(rf'{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}Localization/data/', sub_path))
        else:
            present_edge = self.all_data[f'{edge}_{test_num}']
        present_edge = present_edge[np.random.choice(random_num)]

        self.AllHistoricalRecord[str(edge)] = present_edge
        self.agents[agent_id]['historicalRecord'].append(present_edge)
        self.agents[agent_id]['historicalRecord'] = merge_multiple_edge_data(self.agents[agent_id]['historicalRecord'])

        return present_edge

    def return_position_error(self, estimated_probability, TopK=5):
        test_num = self.agents[0]['test_num']
        truth_position = Config().RpPosition[int(test_num) - 1]
        TopK_index = np.argsort(estimated_probability)[::-1][:TopK]
        TopK_probability = estimated_probability[TopK_index]
        # 归一化
        TopK_probability = TopK_probability / np.sum(TopK_probability)
        TopK_position = Config().RpPosition[TopK_index]

        estimated_position = np.sum(TopK_position * TopK_probability[:, np.newaxis], axis=0)
        position_loss = np.linalg.norm(estimated_position - truth_position)
        return position_loss

    @staticmethod
    def return_probability(model, x_test, device=PARAMETER_CONFIGURATION['device']):
        model.eval()
        data = torch.tensor(np.array(x_test), dtype=torch.float32)
        data = data.to(device)
        with torch.no_grad():
            output = model(data)
            _, predicted = torch.max(output, 1)
            output = output.cpu().numpy()
            predicted = predicted.cpu().numpy()
        return predicted, output

    def get_state(self, agent_id):
        """获取指定智能体的当前状态"""
        agent = self.agents[agent_id]
        return {'current_node': agent['current_node'], 'visited_edges': agent['visited_edges'],
                'rss_data': agent['rss_data']}


