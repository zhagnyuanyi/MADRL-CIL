# 模型搭建
import gc
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from Config import PARAMETER_CONFIGURATION, length_num, length_num_two, length_num_3
import Data_Load
import my_model, my_model_without_atten, my_model_without_LSTM, my_model_only_LSTM, LITE, WHEN
import numpy as np


def train_model(model, epochs=100, batch_size=128, learning_rate=1e-3,
                monitor='loss', device=PARAMETER_CONFIGURATION['device']):
    global max_acc

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    # 循环训练
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0  # 总的训练样本数

        # 按需加载训练数据
        for i in range(1, 9):
            if not SINGLE_AGENT and i % 2 != 0:
                continue
            if i > 5:
                num = _length_num[f'{i}']
            else:
                num = 1
            for n in range(num):
                x_train, y_train, _, _ = Data_Load.data_loading_P2S1(Starting_side=1, length=i, num=n,
                                                                     file_path=data_file_path)  # 加载训练数据
                x_train = torch.tensor(np.array(x_train))
                y_train = torch.tensor(np.array(y_train))

                train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True,
                                          drop_last=True, num_workers=1, pin_memory=True)

                for inputs, labels in train_loader:
                    inputs = inputs.float().to(device)
                    labels = labels.long().to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    correct_predictions += torch.sum(preds == labels.data).item()
                    total_samples += inputs.size(0)

                # 清理内存
                del x_train, y_train, train_loader
                torch.cuda.empty_cache()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_predictions / total_samples

        # 验证循环
        model.eval()
        running_loss_val = 0.0
        correct_predictions_val = 0
        total_samples_val = 0  # 总的验证样本数

        with torch.no_grad():
            # 按需加载验证数据
            for i in range(1, 9):
                if not SINGLE_AGENT and i % 2 != 0:
                    continue
                _, _, x_test, y_test = Data_Load.data_loading_P2S1(Starting_side=1, length=i, num=0,
                                                                   additive_gaussian_noise=False, test=False,
                                                                   file_path=data_file_path)  # 加载验证数据
                x_test = torch.tensor(np.array(x_test))
                y_test = torch.tensor(np.array(y_test))

                test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False,
                                         drop_last=False, num_workers=1, pin_memory=True)

                for inputs, labels in test_loader:
                    inputs = inputs.float().to(device)
                    labels = labels.long().to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    running_loss_val += loss.item() * inputs.size(0)
                    correct_predictions_val += torch.sum(preds == labels.data).item()
                    total_samples_val += inputs.size(0)

                # 清理内存
                del x_test, y_test, test_loader
                torch.cuda.empty_cache()

        epoch_loss_val = running_loss_val / total_samples_val
        epoch_acc_val = correct_predictions_val / total_samples_val

        # 更新学习率
        if monitor == 'loss':
            scheduler.step(epoch_loss_val)
        else:
            scheduler.step(epoch_acc_val)

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {epoch_loss_val:.4f}, Val Acc: {epoch_acc_val:.4f}')

        if epoch_acc_val > max_acc:
            max_acc = epoch_acc_val
            torch.save(model.state_dict(), 'MaxAcc_' + _model_save_name)
            print(f"模型保存成功：{_model_save_name}.pth")


    return model


def evaluate_model(model, batch_size=128, device=PARAMETER_CONFIGURATION['device']):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for i in range(1, 9):
            if not SINGLE_AGENT and i % 2 != 0:
                continue
            _, _, x_test, y_test = Data_Load.data_loading_P2S1(Starting_side=1, length=i,
                                                               additive_gaussian_noise=False, test=False,
                                                               file_path=data_file_path)  # 加载测试数据
            x_test = torch.tensor(np.array(x_test))
            y_test = torch.tensor(np.array(y_test))

            test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False,
                                     drop_last=False, num_workers=1, pin_memory=True, prefetch_factor=2)

            for inputs, labels in test_loader:
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data).item()
                total_samples += inputs.size(0)

            # 清理内存
            del x_test, y_test, test_loader
            torch.cuda.empty_cache()

    acc = correct_predictions / total_samples
    return acc


def start_train():
    """训练模型"""
    Model = my_model.MixedModel(input_size=PARAMETER_CONFIGURATION['input_size'],
                                num_classes=PARAMETER_CONFIGURATION['num_classes'])
    # 加载已有的模型参数（如果存在）
    model_path = os.path.join(PARAMETER_CONFIGURATION['model_save_path'],'last_' + _model_save_name)
    if os.path.exists(model_path):
        Model.load_state_dict(torch.load(model_path))
        print(f"已加载模型参数：{model_path}")

    Model = Model.to(PARAMETER_CONFIGURATION['device'])  # 模型加载到GPU

    model = train_model(Model, epochs=PARAMETER_CONFIGURATION['epochs'],
                        batch_size=PARAMETER_CONFIGURATION['batch_size'],
                        learning_rate=PARAMETER_CONFIGURATION['lr'], monitor='loss')
    torch.save(model.state_dict(), model_path)
    print(f"模型保存成功：{model_path}")


def start_eval():
    """评估模型"""

    Model = my_model.MixedModel(input_size=PARAMETER_CONFIGURATION['input_size'],
                                num_classes=PARAMETER_CONFIGURATION['num_classes'])  # 模型加载

    model_path = os.path.join(PARAMETER_CONFIGURATION['model_save_path'],
                              'MaxAcc_' + _model_save_name)
    Model.load_state_dict(torch.load(model_path))
    Model = Model.to(PARAMETER_CONFIGURATION['device'])  # 模型加载到GPU

    acc = evaluate_model(Model)
    print(f'模型准确率为：{acc:.4f}')


if __name__ == "__main__":
    SINGLE_AGENT = False
    Two_Agent = False
    max_acc = 0.95
    if SINGLE_AGENT:
        _length_num = length_num
        data_file_path = '/data/Point_2_Side_1'
        _model_save_name = 'Single_Agent.pth'

    elif Two_Agent:
        _length_num = length_num_two
        data_file_path = '/data/Two_Agent'
        _model_save_name = 'Two_Agent.pth'
    else:
        _length_num = length_num_3
        data_file_path = '/data/3_Car'
        _model_save_name = 'Three_Agent.pth'

    start_train()
    # 如果需要评估模型，可以调用 start_eval()
    # start_eval()
