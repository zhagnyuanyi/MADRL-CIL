import torch
import torch.nn as nn
import torch.nn.functional as F

class AgentNetwork(nn.Module):
    def __init__(self, node_num, edge_num, action_size, max_steps=6):
        super(AgentNetwork, self).__init__()

        # 嵌入维度定义
        self.node_embedding_dim = 16
        self.edge_embedding_dim = 8
        self.rss_feature_dim = 16
        self.fusion_dim = 128
        self.hidden_dim = 64
        self.max_steps = max_steps

        # 当前节点嵌入
        self.node_embedding = nn.Embedding(node_num, self.node_embedding_dim)

        # 已遍历边嵌入
        self.edge_embedding = nn.Embedding(edge_num + 1, self.edge_embedding_dim, padding_idx=edge_num)

        # 信念状态处理（使用卷积层）
        self.rss_conv = nn.Sequential(
            nn.Conv1d(in_channels=7, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(224, self.rss_feature_dim),
            nn.ReLU()
        )

        # 初始化卷积层和线性层的权重
        for layer in self.rss_conv:
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        # 特征融合
        total_feature_dim = self.node_embedding_dim + (self.max_steps * self.edge_embedding_dim) + self.rss_feature_dim
        self.fusion_fc = nn.Linear(total_feature_dim, self.fusion_dim)
        self.hidden_fc = nn.Linear(self.fusion_dim, self.hidden_dim)
        self.action_output = nn.Linear(self.hidden_dim, action_size)

        # Dropout层
        self.dropout = nn.Dropout(0.3)

        # 使用 Xavier 初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier 初始化所有线性层"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, current_node, edges_traversed, rss_state):
        # 当前节点嵌入
        current_node = current_node.clone().detach().long()
        node_emb = self.node_embedding(current_node)  # [batch_size, node_embedding_dim]

        # 已遍历边嵌入
        edges_traversed = edges_traversed.long()
        edge_emb = self.edge_embedding(edges_traversed)  # [batch_size, max_steps, edge_embedding_dim]
        edge_emb_flat = edge_emb.view(edges_traversed.size(0), -1) 


        rss_state = rss_state.float()  
        rss_feature = self.rss_conv(rss_state)

        # 合并所有特征
        combined_feature = torch.cat([node_emb, edge_emb_flat, rss_feature], dim=1)

        # 特征融合
        fusion_feature = F.relu(self.fusion_fc(combined_feature)) 
        # fusion_feature = self.dropout(fusion_feature)

        # 隐藏层
        hidden_feature = F.relu(self.hidden_fc(fusion_feature)) 

        # 输出层
        action_logit = self.action_output(hidden_feature)  

        return action_logit
