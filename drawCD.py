# # -*- coding: utf-8 -*-
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import rankdata
# from scipy import stats

# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# # 输入数据：各算法在每个数据集上的五折交叉验证AUC值
# # 数据结构：字典，键为算法名，值为二维数组[数据集数][折数]
# # 示例数据结构（请替换为你的实际数据）
# algorithm_results = {
#     'k-means': [
#         [92.75,96.27,94.92,95.19,97.04],  # 数据集1的5折AUC值
#         [90.57,90.84,90.77,94.36,89.3],  # 数据集2的5折AUC值
#     ],
#     'SC': [
#         [93.5,96.3,93.7,95.1,97.4],
#         [91,90.42,91.68,94.96,88.96],
#     ],
#     'UDFS': [
#         [94.27,96.81,94.35,96.26,96.98],
#         [89.52,93.35,91.6,94.89,89.63],
#     ],
#     'NDFS': [
#         [93.34,95.6,94.35,95,96.31],
#         [92.18,93.22,89.52,94.08,91.23],
#     ],
#     'RUFS': [
#         [97.34,95.52,93.61,96.37,97.45],
#         [91.73,93.72,90.16,94.52,91.08],
#     ],
#     'RSFS': [
#         [93.7,95.64,95.28,95.4,97.4],
#         [90.7,90.01,91.64,95.5,89.19],
#     ],
#     'KSC': [
#         [93.83,96.65,92.45,95.09,96.78],
#         [89.85,92.3,90.67,95.84,92.01],
#     ],
#     'KDBA': [
#         [95.11,96.71,94.99,96.04,98.16],
#         [90.93,93.41,92.22,95.52,90.84],
#     ],
#     'K-shape': [
#         [93.72,96.16,92.14,95.63,97.23],
#         [92.58,91.76,90.91,94.68,92.37],
#     ],
#     'u-shapelet': [
#         [95.35,97.76,96.32,97.12,96.87],
#         [91.82,94.73,92.73,95.52,92.79],
#     ]
# }

# def calculate_average_ranks(algorithm_results):
#     """
#     计算各算法的平均排名
    
#     参数:
#     algorithm_results: 字典，键为算法名，值为二维数组[数据集数][折数]
    
#     返回:
#     names: 算法名列表
#     avranks: 对应的平均排名列表
#     """
#     algorithm_names = list(algorithm_results.keys())
#     num_algorithms = len(algorithm_names)
    
#     # 计算每个算法在每个数据集上的平均AUC
#     dataset_means = {}
#     num_datasets = len(algorithm_results[algorithm_names[0]])
    
#     for alg_name in algorithm_names:
#         dataset_means[alg_name] = []
#         for dataset_idx in range(num_datasets):
#             # 计算该算法在该数据集上的5折平均AUC
#             mean_auc = np.mean(algorithm_results[alg_name][dataset_idx])
#             dataset_means[alg_name].append(mean_auc)
    
#     # 计算每个数据集上各算法的排名
#     all_ranks = []
#     for dataset_idx in range(num_datasets):
#         # 获取该数据集上所有算法的平均AUC
#         dataset_aucs = []
#         for alg_name in algorithm_names:
#             dataset_aucs.append(dataset_means[alg_name][dataset_idx])
        
#         # 计算排名（AUC越大排名越靠前，即排名数字越小）
#         ranks = rankdata([-auc for auc in dataset_aucs], method='average')  # 负号使得大值排名小
#         all_ranks.append(ranks)
    
#     # 计算各算法的平均排名
#     all_ranks = np.array(all_ranks)
#     average_ranks = np.mean(all_ranks, axis=0)
    
#     return algorithm_names, average_ranks.tolist()

# # 尝试导入Orange3的函数
# try:
#     from Orange.evaluation import compute_CD, graph_ranks
#     ORANGE_AVAILABLE = True
# except ImportError:
#     # 如果Orange3不可用，使用自定义实现
#     ORANGE_AVAILABLE = False
    
#     def compute_CD(avranks, N, alpha='0.05', test='nemenyi'):
#         """
#         计算临界差异值
        
#         参数:
#         avranks: 平均排名列表
#         N: 数据集数量
#         alpha: 显著性水平
#         test: 检验类型
        
#         返回:
#         CD: 临界差异值
#         """
#         k = len(avranks)  # 算法数量
        
#         if test == 'nemenyi':
#             # Nemenyi检验的临界值
#             if alpha == '0.05':
#                 q_alpha = 2.344  # k=12时的临界值，可根据实际算法数量调整
#             else:
#                 q_alpha = 2.576  # alpha=0.01时的近似值
                
#             CD = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
        
#         return CD
    
#     def graph_ranks(avranks, names, cd, width=8, textspace=1.5, reverse=True):
#         """
#         绘制临界差异图
        
#         参数:
#         avranks: 平均排名列表
#         names: 算法名列表
#         cd: 临界差异值
#         width: 图宽
#         textspace: 文本间距
#         reverse: 是否反转排名顺序
#         """
#         fig, ax = plt.subplots(figsize=(width, len(names) * 0.5 + 1))
        
#         # 排序
#         sorted_indices = np.argsort(avranks)
#         if reverse:
#             sorted_indices = sorted_indices[::-1]
        
#         sorted_ranks = [avranks[i] for i in sorted_indices]
#         sorted_names = [names[i] for i in sorted_indices]
        
#         # 绘制排名线
#         y_positions = np.arange(len(names))
#         ax.barh(y_positions, sorted_ranks, height=0.6, alpha=0.7)
        
#         # 添加算法名称
#         ax.set_yticks(y_positions)
#         ax.set_yticklabels(sorted_names)
#         ax.set_xlabel('Average Rank')
#         ax.set_title('Critical Difference Diagram')
        
#         # 添加临界差异线
#         for i, rank in enumerate(sorted_ranks):
#             ax.text(rank + 0.1, i, f'{rank:.3f}', va='center')
        
#         # 显示临界差异值
#         ax.text(0.02, 0.98, f'CD = {cd:.3f}', transform=ax.transAxes, 
#                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
#         plt.tight_layout()
#         return fig

# # 计算平均排名
# names, avranks = calculate_average_ranks(algorithm_results)

# # 数据集数量
# datasets_num =15

# # 计算临界差异
# CD = compute_CD(avranks, datasets_num, alpha='0.05', test='bonferroni-dunn')
# # 'bonferroni-dunn' 'nemenyi'

# # 绘制临界差异图
# if ORANGE_AVAILABLE:
#     # 使用Orange3的函数
#     graph_ranks(avranks, names, cd=CD, width=8, textspace=1.5, reverse=True)
# else:
#     # 使用自定义函数
#     graph_ranks(avranks, names, CD, width=8, textspace=1.5, reverse=True)

# plt.savefig('critical_difference_diagram.png', dpi=300, bbox_inches='tight')
# plt.show()

# # 输出排名结果
# print("算法平均排名:")
# for name, rank in zip(names, avranks):
#     print(f"{name}: {rank:.4f}")
# print(f"临界差异 CD: {CD:.4f}")

# if ORANGE_AVAILABLE:
#     print("使用Orange3库绘制")
# else:
#     print("使用自定义函数绘制")

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 输入数据：各算法在每个数据集上的指标值
# 数据结构：字典，键为算法名，值为一维数组[数据集1, 数据集2, ..., 数据集10]
# 每个数字代表该算法在对应数据集上的性能指标（如AUC值）
algorithm_results = {
    'AB-MIL': [92.75, 96.27, 94.92, 95.19, 97.04, 90.57, 90.84, 90.77, 94.36, 89.3],  # 10个数据集的指标值
    'CLAM': [93.5, 96.3, 93.7, 95.1, 97.4, 91, 90.42, 91.68, 94.96, 88.96],
    'DSMIL': [94.27, 96.81, 94.35, 96.26, 96.98, 89.52, 93.35, 91.6, 94.89, 89.63],
    'TransMIL': [93.34, 95.6, 94.35, 95, 96.31, 92.18, 93.22, 89.52, 94.08, 91.23],
    'MHIM-MIL': [97.34, 95.52, 93.61, 96.37, 97.45, 91.73, 93.72, 90.16, 94.52, 91.08],
    'IB-MIL': [93.7, 95.64, 95.28, 95.4, 97.4, 90.7, 90.01, 91.64, 95.5, 89.19],
    'WikG': [93.83, 96.65, 92.45, 95.09, 96.78, 89.85, 92.3, 90.67, 95.84, 92.01],
    'RRT-MIL': [95.11, 96.71, 94.99, 96.04, 98.16, 90.93, 93.41, 92.22, 95.52, 90.84],
    'H-MIL': [93.72, 96.16, 92.14, 95.63, 97.23, 92.58, 91.76, 90.91, 94.68, 92.37],
    'Ours': [95.35, 97.76, 96.32, 97.12, 96.87, 91.82, 94.73, 92.73, 95.52, 92.79]
}

def calculate_average_ranks(algorithm_results):
    """
    计算各算法的平均排名
    
    参数:
    algorithm_results: 字典，键为算法名，值为一维数组[数据集1, 数据集2, ..., 数据集N]
    
    返回:
    names: 算法名列表
    avranks: 对应的平均排名列表
    """
    algorithm_names = list(algorithm_results.keys())
    num_algorithms = len(algorithm_names)
    num_datasets = len(algorithm_results[algorithm_names[0]])
    
    print(f"算法数量: {num_algorithms}")
    print(f"数据集数量: {num_datasets}")
    print()
    
    # 计算每个数据集上各算法的排名
    all_ranks = []
    for dataset_idx in range(num_datasets):
        # 获取该数据集上所有算法的指标值
        dataset_scores = []
        for alg_name in algorithm_names:
            dataset_scores.append(algorithm_results[alg_name][dataset_idx])
        
        # 计算排名（指标值越大排名越靠前，即排名数字越小）
        ranks = rankdata([-score for score in dataset_scores], method='average')  # 负号使得大值排名小
        all_ranks.append(ranks)
        
        # 打印每个数据集的排名情况
        print(f"数据集 {dataset_idx + 1}:")
        for i, (alg_name, score, rank) in enumerate(zip(algorithm_names, dataset_scores, ranks)):
            print(f"  {alg_name}: {score:.2f} (排名: {rank:.1f})")
        print()
    
    # 计算各算法的平均排名
    all_ranks = np.array(all_ranks)
    average_ranks = np.mean(all_ranks, axis=0)
    
    return algorithm_names, average_ranks.tolist()

# 尝试导入Orange3的函数

from Orange.evaluation import compute_CD, graph_ranks
ORANGE_AVAILABLE = True

# 计算平均排名
names, avranks = calculate_average_ranks(algorithm_results)
avranks=[7.3164, 6.3555, 5.3523, 7.4985, 5.1334, 6.25, 6.3888, 3.1835, 5.23, 1.15]
# 数据集数量（从数据中自动获取）
datasets_num = len(algorithm_results[names[0]])

# 计算临界差异
CD = compute_CD(avranks, 35, alpha='0.05', test='nemenyi')
# 可选择 'bonferroni-dunn' 或 'nemenyi'

# 绘制临界差异图
if ORANGE_AVAILABLE:
    # 使用Orange3的函数
    graph_ranks(avranks, names, cd=CD, width=6, textspace=1.0, reverse=True)


plt.savefig('critical_difference_diagram.png', dpi=300, bbox_inches='tight')
plt.show()

# 输出排名结果
print("=" * 50)
print("算法平均排名:")
print("=" * 50)
sorted_indices = np.argsort(avranks)
for i, idx in enumerate(sorted_indices):
    print(f"{i+1:2d}. {names[idx]:12s}: {avranks[idx]:.4f}")
print("=" * 50)
print(f"临界差异 CD: {CD:.4f}")
print(f"数据集数量: {datasets_num}")

if ORANGE_AVAILABLE:
    print("使用Orange3库绘制")
else:
    print("使用自定义函数绘制")

# 进行显著性检验
print("\n显著性检验结果:")
print("=" * 50)
best_rank = min(avranks)
best_algorithm = names[avranks.index(best_rank)]
print(f"最佳算法: {best_algorithm} (平均排名: {best_rank:.4f})")
print(f"与最佳算法有显著差异的算法 (差异 > CD = {CD:.4f}):")
for name, rank in zip(names, avranks):
    if rank - best_rank > CD:
        print(f"  {name}: {rank:.4f} (差异: {rank - best_rank:.4f})")