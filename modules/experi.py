import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn.functional as F

def select_features(features, a):
    # 确保features是一个PyTorch张量
    features = torch.tensor(features, dtype=torch.float32)
    
    # 计算特征之间的余弦相似度矩阵
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
    
    # 初始化已选择的特征索引列表和相似度之和
    selected_indices = []
    min_sim_sum = float('inf')
    
    # 贪心算法选择特征
    for _ in range(a):
        if not selected_indices:
            # 如果还没有选择特征，随机选择一个特征
            selected_index = torch.randint(0, len(features), (1,)).item()
            selected_indices.append(selected_index)
        else:
            # 计算当前未选择的特征与已选择特征之间的相似度之和
            sim_sums = sim_matrix[selected_indices, :].sum(dim=0)
            # 选择相似度之和最小的特征
            selected_index = sim_sums.argmin().item()
            selected_indices.append(selected_index)
    
    return selected_indices

# # 示例：生成n个[1, 1024]的特征
# np.random.seed(2)
# n = 20  # 假设有20个特征
# features = np.random.rand(n, 1024)

# # 选择a个特征
# a = 5  # 假设我们只需要选择5个特征
# selected_indices = select_features(features, a)

# print("选中的特征索引：", selected_indices)


# # 示例：生成n个[1, 1024]的特征
# np.random.seed(2)
# n = 20  # 假设有10个特征
# features = np.random.rand(n, 1024)

# # 选择a个特征
# a = 30
# selected_indices = select_features(features, a)

# print("选中的特征索引：", selected_indices)
