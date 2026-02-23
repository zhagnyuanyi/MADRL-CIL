import numpy as np


def add_noise(data, noise_factor=1):
    """
    给数据添加噪声。

    参数:
    - data: 输入的 N*7*30 数据
    - noise_factor: 控制噪声强度的因子

    返回:
    - augmented_data: 添加噪声后的数据
    """
    noise = np.random.normal(0, noise_factor, data.shape)
    # 后几行置为0
    noise[:, 4:, :] = 0
    noise = np.round(noise, 2)
    augmented_data = data + noise
    return augmented_data


def random_scale_data(data, scale_factor_range=(0.9, 1.1), p=0.3):
    """
    对每个数据点以一定的概率进行随机缩放。

    参数:
    - data: 输入的 N*7*30 数据
    - scale_factor_range: 缩放因子的范围 (min_scale, max_scale)，在此范围内随机选择缩放因子
    - p: 每个数据点被缩放的概率

    返回:
    - augmented_data: 数据增强后的数据
    """
    random_scale_factors = np.random.uniform(scale_factor_range[0], scale_factor_range[1], data.shape)
    random_scale_factors[:, 4:, :] = 1

    # 生成一个与 data 形状相同的掩码矩阵，决定每个点是否进行缩放
    random_mask = np.random.rand(*data.shape) < p

    # 根据掩码进行缩放
    augmented_data = np.where(random_mask, data * random_scale_factors, data)
    augmented_data = np.round(augmented_data, 2)

    return augmented_data

def data_translation():
    pass


def start_enhancing(data):
    """
    对输入数据进行两种增强操作，并返回增强后的数据集。

    参数:
    - data: 输入的 N*7*30 数据

    返回:
    - enhanced_data: 增强后的 2*N*7*30 数据
    """
    # 数据增强：加噪声

    data_with_noise = add_noise(data=data, noise_factor=0.5)

    # 数据增强：随机缩放
    data_with_random_scale = random_scale_data(data, scale_factor_range=(0.95, 1.05), p=0.6)

    # 合并增强后的数据
    enhanced_data = np.vstack([data, data_with_noise, data_with_random_scale])

    return enhanced_data

