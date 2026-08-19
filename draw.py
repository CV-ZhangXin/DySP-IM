# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # 确保保存路径存在
# save_dir = '/data3/shihuazhan/DiT/draw'
# os.makedirs(save_dir, exist_ok=True)

# # 数据定义
# datasets = ['CAMELYON', 'TCGA-NSCLC', 'TCGA-BRCA', 'BRACS-3']
# timesteps = ['t=2', 't=5', 't=10', 't=20', 't=100']
# x = np.arange(len(timesteps))  # X轴坐标

# # AUC 数据
# auc_data = {
#     'CAMELYON': [92.62, 91.71, 90.72, 91.29, 91.65],
#     'TCGA-NSCLC': [96.68, 96.03, 96.28, 96.53, 96.30],
#     'TCGA-BRCA': [93.51, 91.57, 92.28, 92.70, 92.02],
#     'BRACS-3': [88.41, 88.68, 86.99, 87.23, 86.86]
# }

# # 时间 数据 (秒)
# time_data = {
#     'CAMELYON': [2.63, 2.95, 5.91, 13.02, 49.86],
#     'TCGA-NSCLC': [2.69, 6.13, 6.57, 11.49, 53.98],
#     'TCGA-BRCA': [1.44, 2.57, 5.14, 10.45, 53.67],
#     'BRACS-3': [1.43, 2.70, 6.31, 11.14, 66.23]
# }

# # 全局绘图参数设置
# plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# # 循环为每个数据集画图
# for dataset in datasets:
#     fig, ax1 = plt.subplots(figsize=(8, 5))

#     # --- 绘制时间（柱状图） ---
#     color_time = '#ff9999' # 浅红色
#     bars = ax1.bar(x, time_data[dataset], color=color_time, alpha=0.7, width=0.5, label='Time per WSI (s)')
#     ax1.set_xlabel('Inference Steps ($t$)', fontsize=14)
#     ax1.set_ylabel('Average Time (s)', color='#cc0000', fontsize=14)
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(timesteps, fontsize=12)
#     ax1.tick_params(axis='y', labelcolor='#cc0000')
    
#     # 给柱状图加数值标签
#     for bar in bars:
#         yval = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}', ha='center', va='bottom', fontsize=10, color='#cc0000')

#     # --- 绘制AUC（折线图） ---
#     ax2 = ax1.twinx()  # 创建共享X轴的第二个Y轴
#     color_auc = '#0055aa' # 深蓝色
#     line = ax2.plot(x, auc_data[dataset], color=color_auc, marker='o', linewidth=2.5, markersize=8, label='AUC (%)')
#     ax2.set_ylabel('AUC Performance (%)', color=color_auc, fontsize=14)
#     ax2.tick_params(axis='y', labelcolor=color_auc)
    
#     # 根据数据动态调整Y轴范围，让折线图看起来更居中、波动更明显
#     min_auc = min(auc_data[dataset])
#     max_auc = max(auc_data[dataset])
#     ax2.set_ylim(min_auc - 1.5, max_auc + 1.5)

#     # 给折线图加数值标签 (重点高亮 t=2 的值)
#     for i, txt in enumerate(auc_data[dataset]):
#         fontweight = 'bold' if i == 0 else 'normal' # t=2 加粗
#         ax2.annotate(f'{txt:.2f}', (x[i], auc_data[dataset][i]), 
#                      textcoords="offset points", xytext=(0,-15), ha='center', 
#                      fontsize=11, color=color_auc, fontweight=fontweight)

#     # --- 图表整体美化 ---
#     plt.title(f'Performance vs. Time Cost ({dataset})', fontsize=16, pad=15)
    
#     # 合并图例
#     lines_1, labels_1 = ax1.get_legend_handles_labels()
#     lines_2, labels_2 = ax2.get_legend_handles_labels()
#     ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    
#     plt.grid(axis='y', linestyle='--', alpha=0.3)
#     plt.tight_layout()

#     # --- 保存图表 ---
#     save_path = os.path.join(save_dir, f'{dataset}_tradeoff.pdf') # 保存为高质量PDF，也可改为.png
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.close()

# print(f"四张图表已成功生成并保存至 {save_dir}")



import os
import torch
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. 全局配置区 (如果路径有误，请在此微调)
# ==========================================
SAVE_ROOT = "/data3/shihuazhan/DiT/draw/draw_tsne"
T_STEP = 2  # 假设使用最优步数 t=2 生成的原型
NUM_SAMPLES = 15  # 每个数据集抽取的切片数量

# 定义数据集名称及其对应的 (原始特征根目录, 原型保存子路径名)
# 注意：原型子路径名需要和你生成时 parser.add_argument('--datasets') 的输入一致
DATASETS_INFO = {
    "BRACS-3": {
        "raw_root": "/data/zhangxiaoxian/BRACS_WSI/r50_zft",
        "raw_sub_dir": "pt_files",
        "proto_dataset_name": "bracs"
    },
    "TCGA-BRCA": {
        "raw_root": "/data2/zhangxiaoxian/tcga/brca",
        "raw_sub_dir": "pt_files",
        "proto_dataset_name": "brca"
    },
    "TCGA-NSCLC": {
        "raw_root": "/data/zhangxiaoxian/tcga/zft",
        "raw_sub_dir": "pt_files",
        "proto_dataset_name": "tcga"
    },
    "CAMELYON": {
        "raw_root": "/data2/zhangxiaoxian/camelyon_all/r50_bioseg",
        "raw_sub_dir": "pt", # 根据你之前的代码，camelyon 是 pt 文件夹
        "proto_dataset_name": "camelyon16" 
    }
}

os.makedirs(SAVE_ROOT, exist_ok=True)

# ==========================================
# 2. 绘图核心函数
# ==========================================
def plot_single_tsne(wsi_features, prototypes, save_path, title):
    """为单张 WSI 及其原型绘制 t-SNE 并保存"""
    num_wsi = wsi_features.shape[0]
    all_features = np.vstack((wsi_features, prototypes))
    
    # 降维
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    all_features_2d = tsne.fit_transform(all_features)
    
    wsi_2d = all_features_2d[:num_wsi, :]
    proto_2d = all_features_2d[num_wsi:, :]
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.scatter(wsi_2d[:, 0], wsi_2d[:, 1], c='#87CEFA', alpha=0.5, s=15, label='WSI Instance Features')
    plt.scatter(proto_2d[:, 0], proto_2d[:, 1], c='#DC143C', marker='o', edgecolors='black', s=60, label='Generated Prototypes')
    
    plt.title(title, fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. 批量处理逻辑
# ==========================================
def main():
    print(f"[*] 开始批量化 t-SNE 绘图任务...")
    print(f"[*] 统一保存目录: {SAVE_ROOT}")
    
    for dataset_name, info in DATASETS_INFO.items():
        print(f"\n[{dataset_name}] 正在处理...")
        
        raw_dir = os.path.join(info["raw_root"], info["raw_sub_dir"])
        proto_dir = f"/data3/shihuazhan/DiT/generate/{info['proto_dataset_name']}/t_{T_STEP}"
        
        # 验证路径是否存在
        if not os.path.exists(raw_dir):
            print(f"  [!] 找不到原始特征路径: {raw_dir}，跳过该数据集。")
            continue
        if not os.path.exists(proto_dir):
            print(f"  [!] 找不到原型特征路径: {proto_dir}，跳过该数据集。")
            continue
            
        # 获取两边都有的有效 pt 文件（取交集确保不出错）
        raw_files = set(f for f in os.listdir(raw_dir) if f.endswith('.pt'))
        proto_files = set(f for f in os.listdir(proto_dir) if f.endswith('.pt'))
        valid_files = list(raw_files.intersection(proto_files))
        
        if len(valid_files) == 0:
            print(f"  [!] 在 {dataset_name} 中没有找到匹配的 .pt 文件！")
            continue
            
        # 随机抽取 15 张
        sample_size = min(NUM_SAMPLES, len(valid_files))
        sampled_files = random.sample(valid_files, sample_size)
        print(f"  [*] 找到 {len(valid_files)} 个可用文件，已随机抽取 {sample_size} 个，开始绘图...")
        
        # 使用 tqdm 显示单个数据集的进度
        for pt_name in tqdm(sampled_files, desc=f"Plotting {dataset_name}", leave=False):
            raw_path = os.path.join(raw_dir, pt_name)
            proto_path = os.path.join(proto_dir, pt_name)
            
            # 规范化命名方案: 数据集名_切片名_tsne.pdf (去掉 .pt 后缀)
            clean_name = pt_name.replace('.pt', '')
            save_filename = f"{dataset_name}_{clean_name}_tsne.pdf"
            save_path = os.path.join(SAVE_ROOT, save_filename)
            
            try:
                # 加载特征并转移到 CPU numpy
                wsi_feat = torch.load(raw_path, weights_only=True).cpu().numpy()
                proto_feat = torch.load(proto_path, weights_only=True).cpu().numpy()
                
                # 兼容维度
                if wsi_feat.ndim == 3:
                    wsi_feat = wsi_feat.squeeze(0)
                if proto_feat.ndim == 3:
                    proto_feat = proto_feat.squeeze(0)
                    
                title = f"{dataset_name} - {clean_name}"
                plot_single_tsne(wsi_feat, proto_feat, save_path, title)
                
            except Exception as e:
                print(f"  [!] 处理 {pt_name} 时发生错误: {e}")
                
    print(f"\n[*] 所有任务执行完毕！图片已保存在 {SAVE_ROOT}")

if __name__ == "__main__":
    main()