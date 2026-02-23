import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MixedModel_(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MixedModel_, self).__init__()

        # Conv layers
        self.conv0 = nn.Conv1d(input_size, 32, 3, padding=1)
        self.bn0 = nn.BatchNorm1d(32)
        self.dropout0 = nn.Dropout(0.3)

        self.conv1 = nn.Conv1d(32, 64, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        # LSTM layer
        self.lstm = nn.LSTM(64, 16, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        # Additional Conv layers
        self.conv2 = nn.Conv1d(input_size, 64, 7, padding=3)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.se1 = SqueezeExcitation(64)
        self.conv3 = nn.Conv1d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.3)

        self.se2 = SqueezeExcitation(128)
        self.conv4 = nn.Conv1d(128, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm1d(64)
        self.dropout5 = nn.Dropout(0.3)

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(16 + 64, num_classes)

    def forward(self, x):
        # 输入形状为 (batch_size, num_of_feature, seq_length)
        # Pre-LSTM Convolutional layers
        x1 = F.relu(self.bn0(self.conv0(x)))
        x1 = self.dropout0(x1)
        x1 = F.relu(self.bn1(self.conv1(x1)))

        # LSTM branch
        x1, _ = self.lstm(x1.permute(0, 2, 1))  # 调整为 (batch_size, seq_length, channels)
        x1 = self.dropout2(x1[:, -1, :])  # 取最后一个时间步的输出

        # CNN branch
        x2 = F.relu(self.bn2(self.conv2(x)))
        x2 = self.se1(x2)
        x2 = self.dropout3(x2)
        x2 = F.relu(self.bn3(self.conv3(x2)))
        x2 = self.se2(x2)
        x2 = F.relu(self.bn4(self.conv4(x2)))
        x2 = self.dropout5(x2)

        x2 = self.gap(x2).squeeze(-1)  # 全局平均池化，输出为 (batch_size, channels)

        # Concatenate LSTM and CNN branches
        x = torch.cat([x1, x2], dim=-1)  # 拼接LSTM和CNN分支的输出
        x = self.fc(x)
        return x





class ResidualBlock1D(nn.Module):
    """
    一个用于1D卷积的残差块。
    包含两个卷积层，每个卷积层后跟批归一化和ReLU激活。
    如果输入和输出通道数不同，使用1x1卷积调整维度。
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.residual_conv is not None:
            identity = self.residual_conv(identity)

        out += identity
        out = self.relu(out)

        return out


class AttentionBlock1D(nn.Module):
    """
    一个用于1D数据的多头自注意力块。
    使用1x1卷积生成查询（Q）、键（K）和值（V）。
    计算注意力权重并应用于值（V），然后与输入进行残差连接。
    """

    def __init__(self, in_channels, attention_heads=4):
        super(AttentionBlock1D, self).__init__()
        assert in_channels % attention_heads == 0, "in_channels必须能被attention_heads整除"
        self.attention_heads = attention_heads
        self.head_dim = in_channels // attention_heads

        self.query_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        """
        前向传播。
        参数:
            x: 输入张量，形状为 (batch_size, channels, seq_length)
        返回:
            经过注意力机制处理后的张量，形状与输入相同。
        """
        batch_size, channels, seq_length = x.size()

        # 生成查询、键和值
        Q = self.query_conv(x).view(batch_size, self.attention_heads, self.head_dim, seq_length
                                    )  # (B, heads, head_dim, seq_length)
        K = self.key_conv(x).view(batch_size, self.attention_heads, self.head_dim, seq_length)
        V = self.value_conv(x).view(batch_size, self.attention_heads, self.head_dim, seq_length)

        # 转置以适应注意力计算
        Q = Q.permute(0, 1, 3, 2)  # (B, heads, seq_length, head_dim)
        K = K.permute(0, 1, 3, 2)  # (B, heads, seq_length, head_dim)
        V = V.permute(0, 1, 3, 2)  # (B, heads, seq_length, head_dim)

        # 计算注意力得分
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, heads, seq_length, seq_length)
        attn_weights = self.softmax(attn_scores)  # (B, heads, seq_length, seq_length)

        # 应用注意力权重到值上
        attn_output = torch.matmul(attn_weights, V)  # (B, heads, seq_length, head_dim)
        attn_output = attn_output.permute(0, 1, 3, 2).contiguous().view(batch_size, channels, seq_length
                                                                        )  # (B, channels, seq_length)

        # 残差连接
        out = x + attn_output

        return out


class MixedModel(nn.Module):
    """
    一个混合模型，结合了卷积神经网络（CNN）和长短期记忆网络（LSTM）分支。
    使用残差块和自注意力机制增强特征提取能力。
    """

    def __init__(self, input_size=7, num_classes=56):
        super(MixedModel, self).__init__()

        # CNN分支的残差块
        self.res_block0 = ResidualBlock1D(input_size, 32, kernel_size=3, padding=1)
        self.dropout0 = nn.Dropout(0.3)

        self.res_block1 = ResidualBlock1D(32, 64, kernel_size=5, padding=2)
        self.dropout1 = nn.Dropout(0.3)

        # LSTM层
        self.lstm = nn.LSTM(input_size=64, hidden_size=16, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)

        # CNN分支的另一部分，包含残差块和注意力机制
        self.res_block2 = ResidualBlock1D(input_size, 64, kernel_size=7, padding=3)
        self.dropout3 = nn.Dropout(0.2)

        self.attention1 = AttentionBlock1D(64, attention_heads=4)

        self.res_block3 = ResidualBlock1D(64, 128, kernel_size=5, padding=2)
        self.dropout4 = nn.Dropout(0.3)

        self.attention2 = AttentionBlock1D(128, attention_heads=4)

        self.res_block4 = ResidualBlock1D(128, 64, kernel_size=3, padding=1)
        self.dropout5 = nn.Dropout(0.3)

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc = nn.Linear(16 + 64, num_classes)

    def forward(self, x):
        """
        前向传播。
        参数:
            x: 输入张量，形状为 (batch_size, 7, N)
        返回:
            输出张量，形状为 (batch_size, 56)
        """
        # CNN分支的预处理卷积层
        x1 = self.res_block0(x)  # (B, 32, N)
        x1 = self.dropout0(x1)
        x1 = self.res_block1(x1)  # (B, 64, N)
        x1 = self.dropout1(x1)

        # LSTM分支
        # LSTM期望输入形状为 (batch_size, seq_length, feature)
        x1_perm = x1.permute(0, 2, 1)  # (B, N, 64)
        x1_lstm, _ = self.lstm(x1_perm)  # x1_lstm: (B, N, 16)
        x1_lstm = self.dropout2(x1_lstm[:, -1, :])  # 取最后一个时间步的输出，形状为 (B, 16)

        # CNN分支的另一部分
        x2 = self.res_block2(x)  # (B, 64, N)
        x2 = self.dropout3(x2)
        x2 = self.attention1(x2)  # (B, 64, N)

        x2 = self.res_block3(x2)  # (B, 128, N)
        x2 = self.dropout4(x2)
        x2 = self.attention2(x2)  # (B, 128, N)

        x2 = self.res_block4(x2)  # (B, 64, N)
        x2 = self.dropout5(x2)

        x2 = self.gap(x2).squeeze(-1)  # 全局平均池化，形状为 (B, 64)

        # 拼接LSTM和CNN分支的输出
        x = torch.cat([x1_lstm, x2], dim=-1)  # (B, 16 + 64)

        # 全连接层
        x = self.fc(x)  # (B, num_classes)

        return x

