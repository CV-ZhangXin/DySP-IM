import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numpy as np
import copy


def exists(val):
    return val is not None

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m,nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class MultiLayerDeformableAttention(nn.Module):
    def __init__(self, dim=512, layer_num_points=[2, 2, 2], n_heads=8, dropout=0.1, 
                 residual=True, act='tanh', deformable_layers=3, anchor_num = 2):
        super().__init__()
        self.dim = dim
        self.anchor_num =anchor_num
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.layer_num_points = layer_num_points
        self.deformable_layers = deformable_layers
        
        # 参数验证
        assert len(self.layer_num_points) == self.deformable_layers, \
            f"layer_num_points长度{len(self.layer_num_points)}必须等于deformable_layers{self.deformable_layers}"
        assert self.n_heads == 8, f"n_heads必须为8，当前{self.n_heads}"
        assert dim % n_heads == 0, f"dim {dim} 必须能被 n_heads {n_heads} 整除"
        assert self.dim % self.n_heads == 0, f"dim={self.dim}需被n_heads={self.n_heads}整除"
        
        self.to_qkv = nn.Linear(self.dim, self.dim * 3, bias=False)


    

        # 为每一层创建独立的偏移量网络
        self.layer_point_offsets = nn.ModuleList()
        self.adapt_weight = nn.ModuleList()
        for layer_idx in range(deformable_layers):
            # 每层使用对应的layer_num_points
            current_layer_points = self.layer_num_points[layer_idx]

            point_offset = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.LayerNorm(self.dim),
                    nn.Tanh() if act == 'tanh' else nn.ReLU(),
                    nn.Linear(self.dim, self.n_heads * current_layer_points * 2),
            )

            weight_layer = nn.Sequential(
                    nn.Linear(self.dim, self.dim),
                    nn.LayerNorm(self.dim),
                    nn.Tanh() if act == 'tanh' else nn.ReLU(),
                    nn.Linear(self.dim, 1),
                    nn.Sigmoid()
 
            )
            self.layer_point_offsets.append(point_offset)
            self.adapt_weight.append(weight_layer)
        
        self.residual = residual
        if self.residual:
            self.res_conv = nn.Conv2d(
                in_channels=self.head_dim,
                out_channels=self.head_dim,
                kernel_size=(3, 1),
                padding=(1, 0),
                groups=self.head_dim,
                bias=False
            )
            self.res_norm = nn.GroupNorm(self.head_dim, self.head_dim)
            
        self.to_out = nn.Sequential(nn.Linear(self.dim, self.dim), nn.Dropout(dropout))

    def generate_all_layer_grids(self, x, base_size=3):
        """
        预先生成所有层的采样点网格
        返回: 所有层的网格列表，每层形状为 [b, h, n, tp, 2]
        """
        b, n, c = x.shape
        h = self.n_heads
        
        # 第一层：生成基础网格
        anchor_grid = torch.linspace(-1., 1., steps=base_size, device=x.device)
        anchor_grid = torch.stack(torch.meshgrid(anchor_grid, anchor_grid, indexing='ij'), dim=-1)
        anchor_grid = rearrange(anchor_grid, 'p1 p2 o -> 1 1 1 (p1 p2) o')
        anchor_grid = anchor_grid.expand(b, h, n, -1, -1)
        pool_x = x.mean(dim=1,keepdim = True)
        all_layer_grids = []
        # 为后续层生成网格
        for layer_idx in range(self.deformable_layers):
            prev_grid = anchor_grid  # 前一层的网格
    
 
            # 获取当前层的点数配置
            current_layer_points = self.layer_num_points[layer_idx]

            # 使用当前层的偏移量网络生成偏移
            offsets = self.layer_point_offsets[layer_idx](x)  
            weight = self.adapt_weight[layer_idx](pool_x)
            offsets = weight * offsets

            offsets = rearrange(offsets, 'b n (h ep o) -> b h n ep o', 
                              h=h, ep=current_layer_points, o=2)

            expanded_prev_grid = rearrange(prev_grid, 'b h n tp o -> b h n tp 1 o')
            expanded_offsets = rearrange(offsets, 'b h n ep o -> b h n 1 ep o')

            grid = expanded_prev_grid + expanded_offsets
            grid = rearrange(grid, 'b h n tp ep o -> b h n (tp ep) o')
            
            all_layer_grids.append(grid)

            anchor_grid =  grid
        
        return all_layer_grids

    def forward(self, x, return_attn=False):
        """
        前向传播 - 单次调用处理所有层
        """
        b, n, c = x.shape
        h, d = self.n_heads, self.head_dim
        
        # 预先生成所有层的采样点网格
        all_layer_grids = self.generate_all_layer_grids(x,base_size=self.anchor_num)
        
        # 拆分QKV
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v_original = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h, d=d), qkv)
        
        # 初始化特征
        current_feat = k  # 使用K作为初始特征进行变形
        q_layer = rearrange(q, 'b h n d -> b h n 1 d')
        # 逐层应用可变形注意力
        all_attentions = []
        for layer_idx in range(self.deformable_layers):
            grid = all_layer_grids[layer_idx]
            current_tp = grid.shape[3]  # 当前层总采样点数
            
            # 调整网格形状以适应grid_sample
            grid_sampler = rearrange(grid, 'b h n tp o -> (b h) tp n o')
            
            # 对当前特征进行变形采样
            feat_grid = rearrange(current_feat, 'b h n d -> (b h) d n 1')
            deformed_feat = F.grid_sample(feat_grid, grid_sampler, mode='nearest', 
                                        padding_mode='border', align_corners=True)
            deformed_feat = rearrange(deformed_feat, '(b h) d tp n -> b h n tp d', 
                                    b=b, h=h, tp=current_tp)
            
            # 计算注意力

            attn = torch.einsum('b h n s d, b h n p d -> b h n p', q_layer, deformed_feat)
            attn = attn.softmax(dim=-1)
            
            # 聚合特征
            if layer_idx == 0:
                # 第一层使用原始V值
                v_grid = rearrange(v_original, 'b h n d -> (b h) d n 1')
                v_deformed = F.grid_sample(v_grid, grid_sampler, mode='bilinear', 
                                         padding_mode='border', align_corners=True)
                v_deformed = rearrange(v_deformed, '(b h) d tp n -> b h n tp d', 
                                     b=b, h=h, tp=current_tp)
            else:
                # 后续层使用变形后的特征作为V
                v_deformed = deformed_feat
            
            layer_out = torch.einsum('b h n p, b h n p d -> b h n d', attn, v_deformed)
            
            # 更新当前特征用于下一层
            current_feat = layer_out
            all_attentions.append(attn)
        
        # 残差连接
        if self.residual:
            v_reshaped = rearrange(v_original, 'b h n d -> (b h) d 1 n')
            conv_out = self.res_conv(v_reshaped)
            conv_out = self.res_norm(conv_out)
            conv_out = rearrange(conv_out, '(b h) d 1 n -> b h n d', b=b, h=h)
            current_feat = current_feat + conv_out
        
        # 重组输出维度
        out = rearrange(current_feat, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        if return_attn:
            return out, all_attentions, all_layer_grids
        return out, all_layer_grids[-1]  # 返回最后一层的网格

class ABMILAggregator(nn.Module):
    """独立的ABMIL聚合模块"""
    def __init__(self, in_dim=512, hidden_dim=256, n_classes=2):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        

    def forward(self, x):
        b, n, _ = x.shape
        attn_weights = self.attention(x)  # [b, n, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # 实例级权重归一化
        bag_feat = torch.matmul(attn_weights.transpose(1, 2), x).squeeze(1)  # [b, in_dim]
        return bag_feat, attn_weights
    



def partion(x):
    B, L, C = x.shape
    H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
    add_length = H ** 2 - L
    x = torch.cat((x, torch.zeros((B, add_length, C),device=x.device)), dim=1)
    return x.view(B, H, W, C).permute(0, 3, 1, 2), add_length

def anti_partion(x, add_lenth):
    B, C, H, W = x.shape
    get_length = H * W - add_lenth
    x = x.permute(0, 2, 3, 1).view(B, H * W, C)
    return (x[:, :get_length, :])


class ChannelAttention(nn.Module):
    def __init__(self, Channel_nums):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)  # C1D 一维卷积
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # 根据通道数求一维卷积大卷积核大小 odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = self.avg_pool(x)
        F_max = self.max_pool(x)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        F_add_ = F_add.squeeze(-1).permute(0, 2, 1)
        F_add_ = self.conv1d(F_add_).permute(0, 2, 1).unsqueeze(-1)
        out = self.sigmoid(F_add_)
        return out





class AdaptiveChannel(nn.Module):
    def __init__(self, Channel_nums):
        super(AdaptiveChannel, self).__init__()

        self.alpha = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.beta = nn.Parameter(data=torch.FloatTensor([0.5]), requires_grad=True)
        self.gamma = 2
        self.b = 1
        self.k = self.get_kernel_num(Channel_nums)
        self.conv1d = nn.Conv1d(kernel_size=self.k, in_channels=1, out_channels=1, padding=self.k // 2)  # C1D 一维卷积
        self.sigmoid = nn.Sigmoid()

    def get_kernel_num(self, C):  # 根据通道数求一维卷积大卷积核大小 odd|t|最近奇数
        t = math.log2(C) / self.gamma + self.b / self.gamma
        floor = math.floor(t)
        k = floor + (1 - floor % 2)
        return k

    def forward(self, x):
        F_avg = x.mean(dim=1,keepdim=True)
        _,F_max = x.max(dim=1,keepdim=True)
        F_add = 0.5 * (F_avg + F_max) + self.alpha * F_avg + self.beta * F_max
        out = self.sigmoid(self.conv1d(F_add)) * x
        return out















class SpatialAttention(nn.Module):
    def __init__(self, Channel_num,Lambda):
        super(SpatialAttention, self).__init__()
        self.channel = Channel_num
        self.Lambda = Lambda
        self.C_im = self.get_important_channelNum(Channel_num)
        self.C_subim = Channel_num - self.C_im
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.norm_active = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def get_important_channelNum(self, C):  # 根据通道数以及分离率确定重要通道的数量 even|t|最近偶数
        t = self.Lambda * C
        floor = math.floor(t)
        C_im = floor + floor % 2
        return C_im

    
    def get_im_subim_channels(self, C_im, M): # 根据Channel_Attention_Map得到重要通道以及不重要的通道
        _, topk = torch.topk(M, dim=1, k=C_im)
        important_channels = torch.zeros_like(M)
        subimportant_channels = torch.ones_like(M)
        important_channels = important_channels.scatter(1, topk, 1)
        subimportant_channels = subimportant_channels.scatter(1, topk, 0)
        return important_channels, subimportant_channels

    def get_features(self, im_channels, subim_channels, channel_refined_feature):
        import_features = im_channels * channel_refined_feature
        subimportant_features = subim_channels * channel_refined_feature
        return import_features, subimportant_features

    def forward(self, x, M):
        important_channels, subimportant_channels = self.get_im_subim_channels(self.C_im, M)
        important_features, subimportant_features = self.get_features(important_channels, subimportant_channels, x)

        im_AvgPool = torch.mean(important_features, dim=1, keepdim=True) * (self.channel / self.C_im)
        im_MaxPool, _ = torch.max(important_features, dim=1, keepdim=True)

        subim_AvgPool = torch.mean(subimportant_features, dim=1, keepdim=True) * (self.channel / self.C_subim)
        subim_MaxPool, _ = torch.max(subimportant_features, dim=1, keepdim=True)

        im_x = torch.cat([im_AvgPool, im_MaxPool], dim=1)
        subim_x = torch.cat([subim_AvgPool, subim_MaxPool], dim=1)

        A_S1 = self.norm_active(self.conv(im_x))
        A_S2 = self.norm_active(self.conv(subim_x))

        F1 = important_features * A_S1
        F2 = subimportant_features * A_S2

        refined_feature = F1 + F2

        return refined_feature



class ResBlock_HAM(nn.Module):
    def __init__(self, Channel_nums,ratio):
        super(ResBlock_HAM, self).__init__()
        self.channel = Channel_nums
        self.ChannelAttention = ChannelAttention(self.channel)
        self.SpatialAttention = SpatialAttention(self.channel,ratio)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x_part,add_length = partion(x_in)
        residual = x_part
        channel_attention_map = self.ChannelAttention(x_part)
        channel_refined_feature = channel_attention_map * x_part
        final_refined_feature = self.SpatialAttention(channel_refined_feature, channel_attention_map)
        out = self.relu(final_refined_feature + residual)
        return anti_partion(out,add_length)


class DyToP(nn.Module):
    def __init__(self, 
                 input_dim=1024, 
                 n_classes=4, 
                 dropout=0.25, 
                 act='gelu', 
                 rrt=None, 
                 deformable=True, 
                 deformable_layers=1, 
                 layer_num_points=[4], 
                 n_heads=8, 
                 abmil_hidden_dim=512,
                 partial_deform=False,           # ✅ 新增：是否部分通道 deformable
                 anchor_num = 1,
                 partial_ratio=0.25):            # ✅ 新增：部分 deformable 比例（默认 1/4）
        super(DyToP, self).__init__()
        
        self.L = 512
        self.D = 128
        self.K = 1
        self.deformable = deformable
        self.partial_deform = partial_deform
        self.partial_ratio = partial_ratio
        self.deformable_layers = deformable_layers
        self.layer_num_points = layer_num_points
        self.n_heads = n_heads

        # 特征降维层
        self.feature = [nn.Linear(input_dim, self.L)]
        if act.lower() == 'gelu':
            self.feature += [nn.GELU()]
        else:
            self.feature += [nn.ReLU()]
        if dropout:
            self.feature += [nn.Dropout(dropout)]
        if rrt is not None:
            self.feature += [rrt] 
        self.feature = nn.Sequential(*self.feature)


        # self.ac= AdaptiveChannel(self.L)

        # self.ham = ResBlock_HAM(self.L,self.partial_ratio)

        if self.partial_deform:
            self.fused_lienar = nn.Sequential(nn.Linear(self.L,self.L),nn.Dropout(dropout))
        if self.deformable:
            self.deformable_attn = MultiLayerDeformableAttention(
                dim=int(self.L * self.partial_ratio) if partial_deform else self.L,
                layer_num_points=self.layer_num_points,
                n_heads=n_heads,
                dropout=dropout,
                residual=True,
                act=act,
                anchor_num = anchor_num,
                deformable_layers=deformable_layers
            )

            self.abmil_agg = ABMILAggregator(
                in_dim=self.L,
                hidden_dim=abmil_hidden_dim,
                n_classes=n_classes
            )


        self.classifier = nn.Sequential(nn.Linear(self.L * self.K, n_classes))
        self.apply(initialize_weights)

    def forward(self, x, return_attn=False):
        feature = self.feature(x)  # [B, N, 512]
        # feature = self.ham(feature)
        if self.deformable:
            # ✅ 情况1：部分通道 deformable
            if self.partial_deform:
                deform_dim = int(self.L * self.partial_ratio)
                deform_feat, static_feat = torch.split(feature, [deform_dim, self.L - deform_dim], dim=-1)

                if return_attn:
                    deform_out, attn_list, all_grids = self.deformable_attn(deform_feat, return_attn=True)
                else:
                    deform_out, _ = self.deformable_attn(deform_feat, return_attn=False)

                # 拼接回 512 维
                fused_feat = torch.cat([deform_out, static_feat], dim=-1)
                fused_feat = self.fused_lienar(fused_feat)
                # 聚合分类
                bag_feat, abmil_attn = self.abmil_agg(fused_feat)
                Y_prob = self.classifier(bag_feat)
                if return_attn:
                    return Y_prob, abmil_attn, attn_list
                return Y_prob


            else:
                if return_attn:
                    current_feat, attn_list, all_grids = self.deformable_attn(feature, return_attn=True)
                else:
                    current_feat, _ = self.deformable_attn(feature, return_attn=False)

                bag_feat, abmil_attn = self.abmil_agg(current_feat)
                Y_prob = self.classifier(bag_feat)
                return Y_prob

        # ✅ 情况3：不使用 deformable
        else:
            bag_feat, abmil_attn = self.abmil_agg(current_feat)
            Y_prob = self.classifier(bag_feat)
            if return_attn:
                return Y_prob, abmil_attn
            return Y_prob

