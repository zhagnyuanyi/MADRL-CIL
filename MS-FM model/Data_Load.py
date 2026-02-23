import gc
import os
import pickle
import pandas as pd
import numpy as np
from data_enhancement import *
from Config import *

class DatasetLoader:
    def __init__(self, data_path, side, test_rp_num, rp_num):
        self.data_path = data_path
        self.side = side
        self.test_rp_num = test_rp_num
        self.rp_num = rp_num 

    def load_dataset(self, additive_gaussian_noise=True):
        """加载参考点数据集"""
        x_train, y_train, x_test, y_test = [], [], [], []
        for side in self.side:
            for rp in range(1, self.rp_num + 1):
                path = os.path.join(self.data_path, 'rp', f'side_{side}_rp_{rp}.csv')
                single_rp_data = self.load_single_dataset(path)
                test_index = range(1, 40, 9)
                train_index = [i for i in range(40) if i not in test_index]
                test_single_rp_data = single_rp_data[test_index]
                train_single_rp_data = single_rp_data[train_index]  


                train_single_rp_data = start_enhancing(train_single_rp_data)

                if additive_gaussian_noise:
                    for i in range(1, 3):
                        test_single_rp_data = self.add_gaussian_noise(test_single_rp_data, 0, i/2)

                    for i in range(1, 3):
                        train_single_rp_data = self.add_gaussian_noise(train_single_rp_data, 0, i/2)

                x_train.append(train_single_rp_data)
                x_test.append(test_single_rp_data)
                y_train.append([rp - 1] * len(train_single_rp_data))
                y_test.append([rp - 1] * len(test_single_rp_data))

        # 将数据第一维度合并
        x_train, x_test = map(np.vstack, (x_train, x_test))
        # 将标签第0维度合并
        y_train, y_test = map(np.hstack, (y_train, y_test))
        x_train, y_train, x_test, y_test = map(np.array, (x_train, y_train, x_test, y_test))

        return x_train, y_train, x_test, y_test

    def load_test_dataset(self):
        """加载测试数据集——方便数据处理，x_train, y_train返回的数据是无效的"""
        x_train, y_train, x_test, y_test = [], [], [], []
        for side in self.side:
            for rp in range(1, self.test_rp_num + 1):
                path = os.path.join(self.data_path, 'test', f'test_side_{side}_rp_{rp}.csv')
                single_rp_data = self.load_single_dataset(path)
                x_test.append(single_rp_data[:5])
                x_train.append(single_rp_data[:5])
                y_test.append([rp + self.rp_num- 1] * 5)
                y_train.append([rp + self.rp_num - 1] * 5)
        x_train, x_test = map(np.vstack, (x_train, x_test))
        # 将标签第0维度合并
        y_train, y_test = map(np.hstack, (y_train, y_test))
        x_train, y_train, x_test, y_test = map(np.array, (x_train, y_train,x_test, y_test))
        return x_train, y_train, x_test, y_test


    @staticmethod
    def load_single_dataset(data_path):
        """读取单个数据集"""
        single_rp_data = []
        df = pd.read_csv(data_path)
        df['rss'] = pd.to_numeric(df['rss'], errors='coerce')
        df['length'] = pd.to_numeric(df['length'], errors='coerce')
        df['position_x'] = pd.to_numeric(df['position_x'], errors='coerce')
        df['position_y'] = pd.to_numeric(df['position_y'], errors='coerce')

        unique_times = np.sort(df['times'].unique())

        for time_value in unique_times:
            group = df[df['times'] == time_value]
            mean_rss = np.round(group.groupby('length')['rss'].mean(), 2)
            min_rss = np.round(group.groupby('length')['rss'].min(), 2)
            max_rss = np.round(group.groupby('length')['rss'].max(), 2)
            std_length_rss = np.round(group.groupby('length')['rss'].std(), 2).fillna(0)  # 将 NaN 值填充为 0
            std_rss = [np.round(group['rss'].std(), 2)] * 30
            mean_position_x = np.round(group.groupby('length')['position_x'].mean(), 2)
            mean_position_y = np.round(group.groupby('length')['position_y'].mean(), 2)
            # 按照维度拼接为一个二维向量
            feature = np.vstack((mean_rss, min_rss, max_rss, std_length_rss, std_rss, mean_position_x, mean_position_y))

            single_rp_data.append(feature)

        return np.array(single_rp_data)

    @staticmethod
    def add_gaussian_noise(data, mean, std):
        noise = np.random.normal(mean, std, data.shape)
        noise[:, 4:, :] = 0  # 后几行置为0，即不对后几行进行噪声添加
        noisy_data = data + noise
        noisy_data = np.round(noisy_data, 2)
        return np.vstack((data, noisy_data))


def merge_multiple_edge_data(side_data_list):
    """
    对多条边的数据进行横向合并，例如：两个1000*7*30 -> 1000*14*30
    input: [[side1], [side2], ...]
    """
    global x_trains, y_trains, x_tests, y_tests
    m_x_train, m_y_train, m_x_test, m_y_test = [], [], [], []
    for side in side_data_list:
        x_train, y_train, x_test, y_test = x_trains[side], y_trains[side], x_tests[side], y_tests[side]

        if len(m_x_train) == 0:
            m_x_train, m_y_train, m_x_test, m_y_test = x_train, y_train, x_test, y_test
        else:
            # 横向合并
            m_x_train = np.concatenate((m_x_train, x_train), axis=-1)
            m_x_test = np.concatenate((m_x_test, x_test), axis=-1)

    m_x_train, m_y_train, m_x_test, m_y_test = map(np.array, (m_x_train, m_y_train, m_x_test, m_y_test))
    return m_x_train, m_y_train, m_x_test, m_y_test


def merge_multiple_edge_sets_of_data(edge_data_list):
    m_x_train, m_y_train, m_x_test, m_y_test = [], [], [], []
    for edge in edge_data_list:
        x_train, y_train, x_test, y_test = merge_multiple_edge_data(edge)
        m_x_train.append(x_train)
        m_y_train.append(y_train)
        m_x_test.append(x_test)
        m_y_test.append(y_test)

    return m_x_train, m_y_train, m_x_test, m_y_test

def get_path_planning(starting_side, length):
    path = []
    for edge in which_path_planning:
        if edge[0] == str(starting_side) and len(edge) == length:
            path.append(edge)
    return path

def start_sort_out_data(path):
    x_train, y_train, x_test, y_test = merge_multiple_edge_sets_of_data(path)
    return x_train, y_train, x_test, y_test

def main(rp=True, side=1, additive_gaussian_noise=True):
    """
    :param rp: 是否加载参考点数据集
    :param side: 边编号
    :param additive_gaussian_noise: 是否添加高斯噪声
    """
    Data_Path = r'../data'
    side = side
    loader = DatasetLoader(Data_Path, side, Test_RP_Num, RP_Num)
    if rp:
        x_train, y_train, x_test, y_test = loader.load_dataset(additive_gaussian_noise=additive_gaussian_noise)
    else:
        x_train, y_train, x_test, y_test = loader.load_test_dataset()
    return x_train, y_train, x_test, y_test

def data_processing(x_data, y_data):
    """对数据进行进一步处理"""
    processed_x, processed_y = [], []
    for side_data in x_data:
        for single_rp_data in side_data:
            processed_x.append(single_rp_data)

    for side_data in y_data:
        for single_rp_data in side_data:
            processed_y.append(single_rp_data)

    return processed_x, processed_y

def data_loading(Starting_side, length, additive_gaussian_noise=True, test=False, file_path=r'/data/localization_data'):
    """一个起点的所有数据加载"""
    if additive_gaussian_noise:
        with open(f'{file_path}/StartingSide_{Starting_side}_num_{length}_noise.pkl', 'rb') as f:  # 训练集
            data = pickle.load(f)
    elif test:
        with open(f'{file_path}/StartingSide_{Starting_side}_num_{length}_test.pkl', 'rb') as f:  # 测试集
            data = pickle.load(f)
    else:
        with open(f'{file_path}/StartingSide_{Starting_side}_num_{length}_primitive.pkl', 'rb') as f:  # 用于评估网络
            data = pickle.load(f)

    # 加载数据
    x_train_loaded = data['X_train']
    y_train_loaded = data['Y_train']
    x_test_loaded = data['X_test']
    y_test_loaded = data['Y_test']

    # 处理数据
    x_train_processed, y_train_processed = data_processing(x_train_loaded, y_train_loaded)
    x_test_processed, y_test_processed = data_processing(x_test_loaded, y_test_loaded)

    # 释放不再使用的数据
    del data, x_train_loaded, y_train_loaded, x_test_loaded, y_test_loaded
    gc.collect()  # 手动垃圾回收

    return x_train_processed, y_train_processed, x_test_processed, y_test_processed

def data_loading_P2S1(Starting_side, length, additive_gaussian_noise=True, test=False, num=0, file_path=r'/data/Point_2_Side_1'):
    """一个起点的所有数据加载"""
    if additive_gaussian_noise:
        with open(f'{file_path}/StartingSide_{Starting_side}_length_{length}_num_{num}_noise.pkl', 'rb') as f:  # 训练集
            data = pickle.load(f)
    elif test:
        with open(f'{file_path}/StartingSide_{Starting_side}_length_{length}_num_{num}_test.pkl', 'rb') as f:
            data = pickle.load(f)
    else:
        with open(f'{file_path}/StartingSide_{Starting_side}_length_{length}_num_{num}_primitive.pkl', 'rb') as f:
            data = pickle.load(f)

    # 加载数据
    x_train_loaded = data['X_train']
    y_train_loaded = data['Y_train']
    x_test_loaded = data['X_test']
    y_test_loaded = data['Y_test']

    # 处理数据
    x_train_processed, y_train_processed = data_processing(x_train_loaded, y_train_loaded)
    x_test_processed, y_test_processed = data_processing(x_test_loaded, y_test_loaded)

    # 释放不再使用的数据
    del data, x_train_loaded, y_train_loaded, x_test_loaded, y_test_loaded
    gc.collect()  # 手动垃圾回收

    return x_train_processed, y_train_processed, x_test_processed, y_test_processed


def data_set_construction(rp, additive_gaussian_noise):
    """
    数据集构建
    rp：true + additive_gaussian_noise:true   : 训练集
    rp:false + additive_gaussian_noise:false   : 测试集
    rp:true + additive_gaussian_noise:false   : 用于评估网络
    """
    global x_trains, y_trains, x_tests, y_tests
    x_trains, y_trains, x_tests, y_tests = {}, {}, {}, {}
    for i in range(1, 27):  # 26条边
        x_train, y_train, x_test, y_test = main(rp=rp, side=[i], additive_gaussian_noise=additive_gaussian_noise)
        i = str(i)
        x_trains[i] = x_train
        y_trains[i] = y_train
        x_tests[i] = x_test
        y_tests[i] = y_test
    for i in range(1, 2):  # 起始边
        for j in range(1, 17):  # 路径长度
            path = get_path_planning(i, j)
            if path == []:
                continue
            if j > 5 and rp:
                paths = [path[i:i + max_length] for i in range(0, len(path), max_length)]
            else:
                paths = [path]
            for idx, path in enumerate(paths):
                x_train, y_train, x_test, y_test = start_sort_out_data(path=path)
                # 保存数据——起始边，对应边的长度，第几组数据
                if rp:
                    if additive_gaussian_noise:
                        with open(f'{sub_path}/StartingSide_{i}_length_{j}_num_{idx}_noise.pkl', 'wb') as f:
                            pickle.dump({'X_train': x_train, 'Y_train': y_train, 'X_test': x_test, 'Y_test': y_test}, f)
                    else:
                        with open(f'{sub_path}/StartingSide_{i}_length_{j}_num_{idx}_primitive.pkl', 'wb') as f:
                            pickle.dump({'X_train': x_train, 'Y_train': y_train, 'X_test': x_test, 'Y_test': y_test}, f)
                else:
                    with open(f'{sub_path}/StartingSide_{i}_length_{j}_num_{idx}_test.pkl', 'wb') as f:
                        pickle.dump({'X_train': x_train, 'Y_train': y_train, 'X_test': x_test, 'Y_test': y_test}, f)
                # 释放内存
                del x_train, y_train, x_test, y_test

                print(f"起点为{i}的第{j}组数据保存成功！")


def reinforcement_learning_data():
    """将参考点上的前5个数据以及测试点上的前5个数据作为测试集——写入一个字典中"""
    all_data = {}
    for i in range(1, 27):  # 26条边
        for j in range(1, 91):
            if j > 54:
                sub_path = f'test/test_side_{i}_rp_{j - 54}.csv'
            else:
                sub_path = f'rp/side_{i}_rp_{j}.csv'
            data = DatasetLoader.load_single_dataset(os.path.join('/home/YYZhang/project/Localization/data/', sub_path))
            all_data[f'{i}_{j}'] = data
    with open('/home/YYZhang/project/Localization/data/RL_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    print("Done!")


if __name__ == "__main__":
    # # X_train, Y_train, Y_test, Y_test = main(rp=True, side=[1], additive_gaussian_noise=True)
    # # X_train, Y_train, X_test, Y_test = start_sort_out_data(1)

    # ######################## 数据集分配 #############################
    global x_trains, y_trains, x_tests, y_tests
    max_length = 20  #一个数据集中最大的路径组合个数
    # sub_path = '/data/Point_2_Side_1'
    sub_path = '/data/4_Car'
    which_path_planning = path_planning_4
    data_set_construction(rp=True, additive_gaussian_noise=True)  # 训练集
    data_set_construction(rp=True, additive_gaussian_noise=False)  # 用于评估网络——比测试集多一个数据增强
    data_set_construction(rp=False, additive_gaussian_noise=False)  # 测试集
    #################################################################

    # 加载文件
    # x_train_processed, y_train_processed, x_test_processed, y_test_processed = data_loading(Starting_side=15, length=5, additive_gaussian_noise=False, test=True)
    # print("Done!")
    # 从x_train_processed中根据形状分类

    # reinforcement_learning_data()



