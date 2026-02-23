import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class QMIXNet(nn.Module):
    def __init__(self, n_agents, qmix_hidden_units, global_state_dim=428, w_scale=0.01):
        super().__init__()
        self.n_agents = n_agents
        self.qmix_hidden_units = qmix_hidden_units
        self.global_state_dim = global_state_dim
        self.w_scale = w_scale     

        # -------- 状态特征提取 --------
        feature_dim = 224
        self.state_fc = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )

        # -------- 超网络 1 --------
        self.hyper_w1 = nn.Sequential(
            nn.Linear(feature_dim, qmix_hidden_units),
            nn.ReLU(),
            nn.Linear(qmix_hidden_units, n_agents * qmix_hidden_units)
        )
        self.hyper_b1 = nn.Linear(feature_dim, qmix_hidden_units)

        # -------- 超网络 2 --------
        self.hyper_w2 = nn.Sequential(
            nn.Linear(feature_dim, qmix_hidden_units),
            nn.ReLU(),
            nn.Linear(qmix_hidden_units, qmix_hidden_units)
        )
        self.hyper_b2 = nn.Linear(feature_dim, 1)

        # 非线性与 LayerNorm
        self.non_lin = nn.ELU()

        self.apply(self._init_weights)

    # 小范围初始化
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                init.constant_(m.bias, 0.0)

    def forward(self, agent_qs, global_state):
        """
        agent_qs:  (B, T, n_agents)
        global_state: (B, T, 428)
        """
        B, T, _ = agent_qs.size()
        agent_qs   = agent_qs.view(B * T, self.n_agents)
        global_state = global_state.view(B * T, self.global_state_dim)

        # -------- 状态特征 --------
        state_feat = self.state_fc(global_state)  

        # -------- 第一层权重与偏置 --------
        w1 = self.w_scale * F.softplus(self.hyper_w1(state_feat))
        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_units)  # (B*T, n, H)
        b1 = self.hyper_b1(state_feat).view(-1, 1, self.qmix_hidden_units)

        # -------- 隐藏层 --------
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1      
        hidden = self.non_lin(hidden)
        hidden = hidden * self.w_scale
        # hidden = F.layer_norm(hidden, hidden.shape[-1:])

        # -------- 第二层权重与偏置 --------
        w2 = self.w_scale * F.softplus(self.hyper_w2(state_feat))
        w2 = w2.view(-1, self.qmix_hidden_units, 1)               
        b2 = self.hyper_b2(state_feat).view(-1, 1, 1)

        # -------- Q_tot 计算 --------
        q_tot = torch.bmm(hidden, w2) + b2               

        return q_tot.view(B, T, 1)


