import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import ema_update 
import torchvision.models as models
from utils import ema_update 
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
from modules.experi import select_features
from modules.rrt import *
from modules.transmil import *

class HoMPool(nn.Module):
    def __init__(self):
        super(HoMPool, self).__init__()
        self.eps = 1e-5
        
    def forward(self, x):
        cls_token = x[:, 0, :]
        patch = x[:, 1:, :]  # [BS, N_patch, D]
        mean = torch.mean(patch, dim=1)  # [BS, D]
        
        # [BS, D], unbiased=False means the variance is calculated by 1/N instead of 1/(N-1)
        variance = torch.var(patch, dim=1, unbiased=False) 
        std = torch.sqrt(variance + self.eps)  # [BS, D]
        # Calculate third central moment
        centered_patch = patch - mean.unsqueeze(1)  # [BS, N_patch, D]
        third_central_moment = torch.mean(centered_patch ** 3, dim=1)  # [BS, D]
                
        # This approach ensures the resulting cube root retains the direction (positive or negative) of the o
        # riginal third central moment, which is important when dealing with higher-order statistics
        # that may involve negative values.
        third_central_moment = torch.sign(third_central_moment) * (torch.abs(third_central_moment)+ self.eps) ** (1/3)  # [BS, D]
        gauss_embed = torch.cat([cls_token, mean, std, third_central_moment], dim=-1)  # [BS, 4*D]
        return gauss_embed
class SoftTargetCrossEntropy_v2(nn.Module):

    def __init__(self,temp_t=1.,temp_s=1.):
        super(SoftTargetCrossEntropy_v2, self).__init__()
        self.temp_t = temp_t
        self.temp_s = temp_s

    def forward(self, x: torch.Tensor, target: torch.Tensor, mean: bool= True) -> torch.Tensor:
        loss = torch.sum(-F.softmax(target / self.temp_t,dim=-1) * F.log_softmax(x / self.temp_s, dim=-1), dim=-1)
        if mean:
            return loss.mean()
        else:
            return loss


def D(p, z, version='simplified'): # negative cosine similarity
    if version == 'original':
        z = z.detach() # stop gradient
        p = F.normalize(p, dim=1) # l2-normalize 
        z = F.normalize(z, dim=1) # l2-normalize 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':# same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception
    



class AttentionFusionModel(nn.Module):
    def __init__(self, L=1024, D=256):
        super(AttentionFusionModel, self).__init__()
        self.L = L  # 输入向量的维度
        self.D = D  # 中间层的维度

        # 定义注意力层
        self.query_layer = nn.Linear(self.L, self.D, bias=False)
        self.key_layer = nn.Linear(self.L, self.D, bias=False)
        self.value_layer = nn.Linear(self.L, self.D, bias=False)

    def forward(self, a, b):
        # 将 a 视为查询向量，b 视为键和值向量
        # a 的形状是 [1, 1024]，b 的形状是 [n, 1024]

        # 通过查询层、键层和值层
        Q = self.query_layer(a)  # [1, D]
        K = self.key_layer(b)    # [n, D]
        V = self.value_layer(b)  # [n, D]

        # 计算注意力权重
        attention_weights = F.softmax(torch.matmul(Q, K.transpose(0, 1)), dim=1)
        # 使用注意力权重对 b 中的值向量进行加权平均
        c = torch.matmul(attention_weights, V)
        return c

class AttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, embed_dim)  # 可选的全连接层进行后处理

    def forward(self, a, b):
        # 准备输入数据
        # a的形状是[1, 1024]，需要变为[1, 1, 1024]以符合注意力模块的输入要求
        a = a.unsqueeze(1)
        # b的形状是[n, 1024]，需要变为[n, 1, 1024]以符合注意力模块的输入要求
        b = b.unsqueeze(1)

        # 应用注意力机制
        # a作为查询向量，b同时作为键向量和值向量
        c, _ = self.attention(a, b, b)

        # 后处理，例如使用全连接层
        #c = self.fc(c)

        # 最终的c形状应该是[1, 1024]
        return c.squeeze(1)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # ref from meituan
            # fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            # fan_out //= m.groups
            # m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Adapter(nn.Module):
    def __init__(self, c_in,c_out,reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    

def cosine_similarity_loss(c, b):
    # 计算c与b中每个向量的余弦相似度
    cosine_similarities = F.cosine_similarity(c, b, dim=1)
    # 计算平均余弦相似度
    mean_cosine_similarity = torch.mean(cosine_similarities)
    # 取负值作为损失
    loss = 1 - mean_cosine_similarity
    return loss

class FCLayer(nn.Module):
    def __init__(self, dropout=0.25,act='relu',in_size=1024):
        super(FCLayer, self).__init__()
        self.embed = [nn.Linear(in_size, 512)]
        # self.embed.append(SwinEncoder(attn='swin',pool='none'))
        # self.embed = nn.ModuleList([nn.Linear(1024, 512)])
        
        if act.lower() == 'gelu':
            self.embed += [nn.GELU()]
        else:
            self.embed += [nn.ReLU()]

        if dropout:
            self.embed += [nn.Dropout(dropout)]

        self.embed = nn.Sequential(*self.embed)

    def forward(self, feats):
        feats = self.embed(feats)
        return feats

class FCLayer1024(nn.Module):
    def __init__(self, dropout=0.25,act='relu',in_size=1024):
        super(FCLayer1024, self).__init__()
        self.embed = [nn.Linear(in_size, 1024)]
        # self.embed.append(SwinEncoder(attn='swin',pool='none'))
        # self.embed = nn.ModuleList([nn.Linear(1024, 512)])
        
        if act.lower() == 'gelu':
            self.embed += [nn.GELU()]
        else:
            self.embed += [nn.ReLU()]

        if dropout:
            self.embed += [nn.Dropout(dropout)]

        self.embed = nn.Sequential(*self.embed)

    def forward(self, feats):
        feats = self.embed(feats)
        return feats    
class FCLayer512_1024(nn.Module):
    def __init__(self, dropout=0.25,act='relu',in_size=512):
        super(FCLayer512_1024, self).__init__()
        self.embed = [nn.Linear(in_size, 1024)]
        # self.embed.append(SwinEncoder(attn='swin',pool='none'))
        # self.embed = nn.ModuleList([nn.Linear(1024, 512)])
        
        if act.lower() == 'gelu':
            self.embed += [nn.GELU()]
        else:
            self.embed += [nn.ReLU()]

        if dropout:
            self.embed += [nn.Dropout(dropout)]

        self.embed = nn.Sequential(*self.embed)

    def forward(self, feats):
        feats = self.embed(feats)
        return feats  

class DAttention(nn.Module):
    def __init__(self,out_dim=2,n_robust=0):
        super(DAttention, self).__init__()
        self.embedding = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        
        self.head = nn.Linear(512,out_dim)

    def forward(self, x):
        # b,p,n = x.size()
        x = self.embedding(x) # 1024->512
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        x = self.head(x.squeeze(1))

        return x,A
    
    def forward_test(self, x):
        # b,p,n = x.size()
        x = self.embedding(x) # 1024->512
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A,x)

        x = self.head(x.squeeze(1))

        return x,A

class DAttentionWithDiff1(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0):
        super(DAttentionWithDiff1, self).__init__()
        self.embedding = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        
        self.head = nn.Linear(512,out_dim)
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

    @torch.no_grad()
    def Diffusion_reembed(self,x): #实现思路：随机噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        x_pooled = x_pooled.repeat(1, 4, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=x_pooled.shape,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
    
    def drawTsne200(self,a,b):
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))
        b = b.reshape(-1, 1024)
        data = np.vstack((a, b))
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)
        plt.figure(figsize=(10, 5))
        # plt.scatter(data_tsne[:-1, 0], data_tsne[:-1, 1], c='blue', label='A Data',s=1)
        # plt.scatter(data_tsne[-1, 0], data_tsne[-1, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[:-200, 0], data_tsne[:-200, 1], c='blue', label='A Data',s=1)
        plt.scatter(data_tsne[-200:, 0], data_tsne[-200:, 1], c='red', label='B Data', s=1)
        plt.legend()
        plt.title('t-SNE Visualization')
        #output_path = '/nas/zhangxiaoxian/output/mil_shz/tsne_200/tsne_visualization_{}.png'.format(random_number)
        output_path = '/data/shihuazhan/output_wsi/tsne_200/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))
        
        return random_number


     
    def drawTsne(self, a, b, c, d):
        # 将数据转移到CPU并转换为numpy数组
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        c = c.cpu().numpy()
        d = d.cpu().numpy()

        # 生成一个随机数作为文件名的一部分
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))

        # 将数据重塑为二维数组
        c = c.reshape(-1, 1024)
        b = b.reshape(-1, 1024)
        a = a.reshape(-1, 1024)
        d = d.reshape(-1, 1024)

        # 计算每个数据集的大小
        size_a = a.shape[0]
        size_b = b.shape[0]
        size_c = c.shape[0]
        size_d = d.shape[0]

        # 合并数据
        data = np.vstack((a, b, c, d))

        # 应用t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)

        # 绘制t-SNE结果
        plt.figure(figsize=(10, 5))
        plt.scatter(data_tsne[:size_a, 0], data_tsne[:size_a, 1], c='blue', label='A Data', s=1)
        plt.scatter(data_tsne[size_a:size_a+size_b, 0], data_tsne[size_a:size_a+size_b, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[size_a+size_b:size_a+size_b+size_c, 0], data_tsne[size_a+size_b:size_a+size_b+size_c, 1], c='green', label='C Data', s=1)
        plt.scatter(data_tsne[-size_d:, 0], data_tsne[-size_d:, 1], c='yellow', label='D Data', s=10)
        plt.legend()
        plt.title('t-SNE Visualization')
        
        # 保存图像
        output_path = '/data/shihuazhan/output_wsi/tsne4/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))

        return random_number

       



    @torch.no_grad()
    def random_reembed(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) 
        x_pooled = x[random_index]
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled

    @torch.no_grad()
    def average_reembed(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        x_pooled = torch.mean(x, dim=0)  # 在第0维上计算平均值 
        
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled
    
   
    
    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a = self.Diffusion_reembed(x)
            elif self.ifrand == 1:
                a = self.random_reembed(x)
            else: #ifrand==2
                a = self.average_reembed(x)
            x = x.squeeze()
            tsnex= x.squeeze()
            """
            此处为tsne画图
            """
            # results = []
            # for _ in range(200):
            #     aa = self.Diffusion_reembed(x)
            #     results.append(aa)
            # b = torch.stack(results, dim=0)
            # nothing=self.drawTsne200(tsnex,b)
            """
            此处为tsne画图
            """
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]

            """
            此处为tsne画图
            """
 
            # nothing=self.drawTsne(tsnex,result,result_high,a)
            """
            此处为tsne画图
            """

            #boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
            x = result.view(-1, 1024)
            # b,p,n = x.size()
            x = self.embedding(x) # 1024->512
            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N
            x = torch.matmul(A,x)
            x = self.head(x.squeeze(1))
        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x = self.embedding(x) # 1024->512
                A = self.attention(x)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N
                x = torch.matmul(A,x)
                x = self.head(x.squeeze(1))
            else:
                if self.ifrand == 0:
                    a = self.Diffusion_reembed(x)
                elif self.ifrand == 1:
                    a = self.random_reembed(x)
                else: #ifrand==2
                    a = self.average_reembed(x)        
                #a = self.random_reembed(x)
                x = x.squeeze()
                b = x
                cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
                sorted_indices = torch.argsort(cosine_similarity, dim=0)
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                lowest_k_indices = sorted_indices[:num_elements]
                result = b[lowest_k_indices]
                boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
                x = result.view(-1, 1024)
                # b,p,n = x.size()
                x = self.embedding(x) # 1024->512
                A = self.attention(x)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N
                x = torch.matmul(A,x)
                x = self.head(x.squeeze(1))

        return x


class DAttentionWithRandomAbandon(nn.Module): #这个是跑对比试验用的，看看随即丢弃的效果如何
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0):
        super(DAttentionWithRandomAbandon, self).__init__()
        self.embedding = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        
        self.head = nn.Linear(512,out_dim)
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()
   
    
    
    def forward(self, x):
        if self.training:
            x = x.squeeze()
            b = x
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)    
            random_indices = torch.randperm(b.size(0))[:num_elements]
            result = b[random_indices]
            x = result.view(-1, 1024)
            # b,p,n = x.size()
            x = self.embedding(x) # 1024->512
            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N
            x = torch.matmul(A,x)
            x = self.head(x.squeeze(1))
        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x = self.embedding(x) # 1024->512
                A = self.attention(x)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N
                x = torch.matmul(A,x)
                x = self.head(x.squeeze(1))
            else:
                x = x.squeeze()
                b = x
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                random_indices = torch.randperm(b.size(0))[:num_elements]
                result = b[random_indices]
                x = result.view(-1, 1024)
                # b,p,n = x.size()
                x = self.embedding(x) # 1024->512
                A = self.attention(x)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N
                x = torch.matmul(A,x)
                x = self.head(x.squeeze(1))

        return x
    
class DAttentionWithDiffchose(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,n_robust=0,ifTrain=1,ifrand=0,ifEma=0,ifType=1,ifClose=0):
        super(DAttentionWithDiffchose, self).__init__()
        self.embedding = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.ifEma = ifEma
        self.ifType= ifType
        self.ifClose = ifClose
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        #ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

    # def updateEma(self,a):
    #     self.attention_ema -= (1 - a) * (self.attention_ema - self.attention.weight.data)

    def drawTsne(self,a,b):
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))
        b = b.reshape(-1, 1024)
        data = np.vstack((a, b))
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)
        plt.figure(figsize=(10, 5))
        # plt.scatter(data_tsne[:-1, 0], data_tsne[:-1, 1], c='blue', label='A Data',s=1)
        # plt.scatter(data_tsne[-1, 0], data_tsne[-1, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[:-200, 0], data_tsne[:-200, 1], c='blue', label='A Data',s=1)
        plt.scatter(data_tsne[-200:, 0], data_tsne[-200:, 1], c='red', label='B Data', s=1)
        plt.legend()
        plt.title('t-SNE Visualization')
        output_path = '/nas/zhangxiaoxian/output/mil_shz/tsne_50/tsne_visualization_{}.png'.format(random_number)
        output_path = '/data/shihuazhan/output_wsi/tsne_200_diffchose/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))
        
        return random_number


    def choseInstanceByValue(self,x,attention,score,value):
        
        if value==1: #max attention
            max_att = attention.argmax()
            index=max_att
            a=x[max_att]
        elif value==2: #min attention
            min_att = attention.argmin()
            index=min_att
            a=x[min_att]
        elif value==3: #max score 0
            max_sco0 = score[:, 0].argmax()
            a=x[max_sco0]
            index=max_sco0
        elif value==4: #min score 0
            min_sco0 = score[:, 0].argmin()
            a=x[min_sco0]
            index=min_sco0
        elif value==5: #max score 1
            max_sco1 = score[:, 1].argmax()
            a=x[max_sco1]
            index=max_sco1
        elif value==6: #min score 1          
            min_sco1 = score[:, 1].argmin()
            a=x[min_sco1]
            index=min_sco1

        # print(index)
        return a

    @torch.no_grad()
    def Diffusion_reembed_shareWeights(self,x): #实现思路：用abmil来合成一个当作输入diffusion的特征，然后生成特征当锚点
        
        A = self.attention(x)
        return A
    

    @torch.no_grad()
    def Diffusion_reembed_ChoseScoreMax(self,x): #实现思路：用abmil的head来对特征打分，选择得分最高的特征当初始输入
        x_ori = x.squeeze() #(n,1024)
        x = self.embedding(x) # n,512
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        attentionScore = F.softmax(A, dim=-1)  # softmax over N
        ascore=attentionScore.squeeze(0)
        ascore=ascore.squeeze(0) # n
        scores = self.head(x) # n,2
        chose = self.choseInstanceByValue(x_ori,ascore,scores,self.ifType)
        chose = chose.view(1, 32, 32)
        chose=chose.unsqueeze(1)
        chose=chose.repeat(1, 4, 1, 1)
        z = torch.randn(1, 4, 32, 32, device=A.device) #shz 4.26
        final=chose+z #shz 4.26
        diffusion = create_diffusion(str(self.t_steps))
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,final.shape,final,clip_denoised=False, progress=True,device=A.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                #a = self.Diffusion_reembed_shareWeights(x)
                a =self.Diffusion_reembed_ChoseScoreMax(x)
            x = x.squeeze()
            tsnex= x.squeeze()
            """
            此处为tsne画图
            """
            # results = []
            # for _ in range(200):
            #     aa = self.Diffusion_reembed_ChoseScoreMax(x)
            #     results.append(aa)
            # b = torch.stack(results, dim=0)
            # nothing=self.drawTsne(tsnex,b)
            """
            此处为tsne画图
            """
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            if self.ifClose == 0:
                sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            else:
                sorted_indices = torch.argsort(cosine_similarity, dim=0, descending=True)  #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            lowest_k_indices = sorted_indices[:num_elements]
            result = b[lowest_k_indices]
            # boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
            x = result.view(-1, 1024)
            # b,p,n = x.size()
            x = self.embedding(x)
            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N
            x = torch.matmul(A,x)
            x = self.head(x.squeeze(1))
        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x = self.embedding(x)
                A = self.attention(x)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N
                x = torch.matmul(A,x)
                x = self.head(x.squeeze(1))

        return x

class DAttentionWithDiffTune(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0):
        super(DAttentionWithDiffTune, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        #self.head2 = nn.Linear(512,out_dim)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        # ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

    @torch.no_grad()
    def Diffusion_reembed(self,x): #实现思路：随机噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        x_pooled = x_pooled.repeat(1, 4, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=x_pooled.shape,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
    


    @torch.no_grad()
    def Diffusion_reembed_withInfo(self,x): #实现思路：max-min当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector-min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
    

    @torch.no_grad()
    def Diffusion_reembed_withInfomeanIndex0(self,x): #实现思路：max-min当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        img_splits = torch.split(img.cuda(), 1, dim=1)
        concatenated_vectors = [torch.cat([img_splits[0], vector, img_splits[1], img_splits[2]], dim=1),torch.cat([img_splits[0], img_splits[1], vector, img_splits[2]], dim=1),torch.cat([img_splits[0], img_splits[1], img_splits[2], vector], dim=1),torch.cat([vector, img_splits[0], img_splits[1], img_splits[2]], dim=1)]
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vectors[0],clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
    

    
    @torch.no_grad()
    def Diffusion_reembed_withInfomeanIndex1(self,x): #实现思路：max-min当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        img_splits = torch.split(img.cuda(), 1, dim=1)
        concatenated_vectors = [torch.cat([img_splits[0], vector, img_splits[1], img_splits[2]], dim=1),torch.cat([img_splits[0], img_splits[1], vector, img_splits[2]], dim=1),torch.cat([img_splits[0], img_splits[1], img_splits[2], vector], dim=1),torch.cat([vector, img_splits[0], img_splits[1], img_splits[2]], dim=1)]
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vectors[1],clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
    
    @torch.no_grad()
    def Diffusion_reembed_withInfomeanIndex2(self,x): #实现思路：max-min当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        img_splits = torch.split(img.cuda(), 1, dim=1)
        concatenated_vectors = [torch.cat([img_splits[0], vector, img_splits[1], img_splits[2]], dim=1),torch.cat([img_splits[0], img_splits[1], vector, img_splits[2]], dim=1),torch.cat([img_splits[0], img_splits[1], img_splits[2], vector], dim=1),torch.cat([vector, img_splits[0], img_splits[1], img_splits[2]], dim=1)]
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vectors[2],clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfomeanIndex3(self,x): #实现思路：max-min当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        img_splits = torch.split(img.cuda(), 1, dim=1)
        concatenated_vectors = [torch.cat([img_splits[0], vector, img_splits[1], img_splits[2]], dim=1),torch.cat([img_splits[0], img_splits[1], vector, img_splits[2]], dim=1),torch.cat([img_splits[0], img_splits[1], img_splits[2], vector], dim=1),torch.cat([vector, img_splits[0], img_splits[1], img_splits[2]], dim=1)]
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vectors[3],clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfo1(self,x): #实现思路：min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfo2(self,x): #实现思路：max噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples

    @torch.no_grad()
    def Diffusion_reembed_withInfo3(self,x): #实现思路：max和min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector1=max_vector
        vector1=vector1.unsqueeze(1)
        vector2=min_vector
        vector2=vector2.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 2, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        img = torch.cat([img.cuda(), vector1], dim=1)
        concatenated_vector = torch.cat([img.cuda(), vector2], dim=1)        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    def drawTsne200(self,a,b):
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))
        b = b.reshape(-1, 1024)
        data = np.vstack((a, b))
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)
        plt.figure(figsize=(10, 5))
        # plt.scatter(data_tsne[:-1, 0], data_tsne[:-1, 1], c='blue', label='A Data',s=1)
        # plt.scatter(data_tsne[-1, 0], data_tsne[-1, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[:-200, 0], data_tsne[:-200, 1], c='blue', label='A Data',s=1)
        plt.scatter(data_tsne[-200:, 0], data_tsne[-200:, 1], c='red', label='B Data', s=1)
        plt.legend()
        plt.title('t-SNE Visualization')
        #output_path = '/nas/zhangxiaoxian/output/mil_shz/tsne_200/tsne_visualization_{}.png'.format(random_number)
        output_path = '/data/shihuazhan/output_wsi/tsne_200/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))
        
        return random_number


     
    def drawTsne(self, a, b, c, d):
        # 将数据转移到CPU并转换为numpy数组
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        c = c.cpu().numpy()
        d = d.cpu().numpy()

        # 生成一个随机数作为文件名的一部分
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))

        # 将数据重塑为二维数组
        c = c.reshape(-1, 1024)
        b = b.reshape(-1, 1024)
        a = a.reshape(-1, 1024)
        d = d.reshape(-1, 1024)

        # 计算每个数据集的大小
        size_a = a.shape[0]
        size_b = b.shape[0]
        size_c = c.shape[0]
        size_d = d.shape[0]

        # 合并数据
        data = np.vstack((a, b, c, d))

        # 应用t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)

        # 绘制t-SNE结果
        plt.figure(figsize=(10, 5))
        plt.scatter(data_tsne[:size_a, 0], data_tsne[:size_a, 1], c='blue', label='A Data', s=1)
        plt.scatter(data_tsne[size_a:size_a+size_b, 0], data_tsne[size_a:size_a+size_b, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[size_a+size_b:size_a+size_b+size_c, 0], data_tsne[size_a+size_b:size_a+size_b+size_c, 1], c='green', label='C Data', s=1)
        plt.scatter(data_tsne[-size_d:, 0], data_tsne[-size_d:, 1], c='yellow', label='D Data', s=10)
        plt.legend()
        plt.title('t-SNE Visualization')
        
        # 保存图像
        output_path = '/data/shihuazhan/output_wsi/tsne4/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))

        return random_number

       



    @torch.no_grad()
    def random_reembed(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) 
        x_pooled = x[random_index]
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled

    @torch.no_grad()
    def average_reembed(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        x_pooled = torch.mean(x, dim=0)  # 在第0维上计算平均值 
        
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled
    
    def getLoss(self,x1,x2):
        f, h = self.encoder, self.predictor
        z1, z2 = f(x1), f(x2)
        p1, p2 = h(z1), h(z2)
        L = D(p1, z2) / 2 + D(p2, z1) / 2
        return {'loss': L}

    
    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a = self.Diffusion_reembed_withInfomeanIndex2(x)
            x = x.squeeze()
            # print(x.shape)

            #a=0.8*a+0.2*self.adapter(a)  
            tsnex= x.squeeze()
            # """
            # 此处为tsne画图
            # """
            # results = []
            # for _ in range(200):
            #     aa = self.Diffusion_reembed(x)
            #     results.append(aa)
            # b = torch.stack(results, dim=0)
            # nothing=self.drawTsne200(tsnex,b)
            # """
            # 此处为tsne画图
            # """
            b = x
            # cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            
            # mask = torch.zeros_like(cosine_similarity)
            # probabilities = F.softmax(-cosine_similarity, dim=0)
            # k = int(b.size(0) * self.k_ratio)

            # selected_indices = torch.multinomial(probabilities, num_samples=k, replacement=False)
            # mask = torch.zeros_like(b, dtype=torch.bool)
            # mask[selected_indices] = True
            # result = b * mask
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
                        
            # k = self.k_ratio
            # sorted_cosine_similarity, _ = torch.sort(cosine_similarity)
            # threshold_index = int(k * len(cosine_similarity))
            # threshold = sorted_cosine_similarity[threshold_index]
            # mask_high = cosine_similarity > threshold
            # mask_low = cosine_similarity < threshold
            # result_igh = x[mask_high.unsqueeze(1).expand_as(x)].contiguous()
            # result = x[mask_low.unsqueeze(1).expand_as(x)].contiguous()




            # """
            # 此处为tsne画图
            # """
 
            # nothing=self.drawTsne(tsnex,result,result_high,a)
            # """
            # 此处为tsne画图
            # """

            #boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
            x = result.view(-1, 1024)
            x2=result_high.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)
             

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)
             
            x1 = self.head(x.squeeze(1))
            x2 = self.head(x2.squeeze(1))
            max_index_x1 = torch.argmax(x1, dim=1)
            max_index_x2 = torch.argmax(x2, dim=1)
            same_max_dimension = max_index_x1 == max_index_x2
            result = torch.where(same_max_dimension, x1 - x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))

            # if same_max_dimension:
            #     # 获取最大维度的索引
            #     max_dim_index = max_index_x1
            #     # print(x1)
            #     # print(x2)
                
            #     # 在最大维度上比较x1和x2的对应值
            #     x1_max_dim_values = x1[:, max_dim_index]
            #     x2_max_dim_values = x2[:, max_dim_index]
            #     #comparison = x1_max_dim_values > x2_max_dim_values
            #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
            #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
            #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
            #                   x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
            #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
            #     # 使用较大的值减去较小的值
            #     # print(result)
            #     # print("_________________________________________________________")
            # else:
            #     # 如果最大维度不同，保持原来的操作
            #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)







            # print("x1 is ",end='')
            # print(x1)
            # print("x2 is ",end='')
            # print(x2)
            # print("是否相同 is ",end='')
            # print(same_max_dimension)
            # print("最终结果")
            # print(result)
            # print("_______________________________________________")
            x=result


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))

                # if same_max_dimension:
                # # 获取最大维度的索引
                #     max_dim_index = max_index_x1
                #     # print(x1)
                #     # print(x2)
                    
                #     # 在最大维度上比较x1和x2的对应值
                #     x1_max_dim_values = x1[:, max_dim_index]
                #     x2_max_dim_values = x2[:, max_dim_index]
                #     #comparison = x1_max_dim_values > x2_max_dim_values
                #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
                #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
                #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
                #                 x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
                #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
                # # 使用较大的值减去较小的值
                # # print(result)
                # # print("_________________________________________________________")
                # else:
                #     # 如果最大维度不同，保持原来的操作
                #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)



                x=result


            else:
                a = self.Diffusion_reembed(x)
                # x = x.squeeze() 
                # b = x
                # cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
                # sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
                # sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
                # k = self.k_ratio
                # num_elements = int(torch.tensor(b.size(0)) * k)
                # k_indices = sorted_indices[:num_elements]
                # k_indices_high = sorted_indices_high[:num_elements]
                # result = b[k_indices]
                # result_high = b[k_indices_high]
                # x = result.view(-1, 1024)
                # x2=result_high.view(-1, 1024)
                # # b,p,n = x.size()
                # x = self.embedding1(x) # 1024->512
                # x2 = self.embedding1(x2)
                # A = self.attention(x)
                # A = torch.transpose(A, -1, -2)  # KxN
                # A = F.softmax(A, dim=-1)  # softmax over N
                # A2 = self.attention2(x2)
                # A2 = torch.transpose(A2, -1, -2)  # KxN
                # A2 = F.softmax(A2, dim=-1)  # softmax over N
                # x = torch.matmul(A,x)
                # x2 = torch.matmul(A2,x2)
                # x1 = self.head(x.squeeze(1))
                # x2 = self.head(x2.squeeze(1))
                # max_index_x1 = torch.argmax(x1, dim=1)
                # max_index_x2 = torch.argmax(x2, dim=1)
                # same_max_dimension = max_index_x1 == max_index_x2
                # result = torch.where(same_max_dimension, x1 + x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1))
                # x=result


        return x
    



class DAttentionWithDiffSimSiam(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,a_num=1):
        super(DAttentionWithDiffSimSiam, self).__init__()
        self.a_num=a_num
        self.dims=1024
        self.embedding1 = FCLayer1024()
        self.embedding2 = FCLayer1024()
        self.L = self.dims
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.temp_s=1.
        self.fusion_model = AttentionFusion(1024,1)
        self.temp_t=1.
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )

        self.cl_loss = SoftTargetCrossEntropy_v2(self.temp_t,self.temp_s)
        self.head = nn.Linear(self.dims,out_dim)
        #self.head2 = nn.Linear(512,out_dim)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        #ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

    @torch.no_grad()
    def Diffusion_reembed(self,x): #实现思路：随机噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        x_pooled = x_pooled.repeat(1, 4, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=x_pooled.shape,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
    
    def drawTsne200(self,a,b):
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
        #b = b.cpu().numpy()
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))
        b = b.reshape(-1, 1024)
        data = np.vstack((a, b))
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)
        plt.figure(figsize=(10, 5))
        # plt.scatter(data_tsne[:-1, 0], data_tsne[:-1, 1], c='blue', label='A Data',s=1)
        # plt.scatter(data_tsne[-1, 0], data_tsne[-1, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[:-200, 0], data_tsne[:-200, 1], c='blue', label='A Data',s=1)
        plt.scatter(data_tsne[-200:, 0], data_tsne[-200:, 1], c='red', label='B Data', s=1)
        plt.legend()
        plt.title('t-SNE Visualization')
        output_path = '/nas/zhangxiaoxian/output/mil_shz/attentsne200/tsne_visualization_{}.png'.format(random_number)
        #output_path = '/data/shihuazhan/output_wsi/tsne_200/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))
        plt.close()
        return random_number


     
    def drawTsne(self, a, b, c, d):
        # 将数据转移到CPU并转换为numpy数组
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        c = c.cpu().numpy()
        d = d.cpu().numpy()

        # 生成一个随机数作为文件名的一部分
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))

        # 将数据重塑为二维数组
        c = c.reshape(-1, 1024)
        b = b.reshape(-1, 1024)
        a = a.reshape(-1, 1024)
        d = d.reshape(-1, 1024)

        # 计算每个数据集的大小
        size_a = a.shape[0]
        size_b = b.shape[0]
        size_c = c.shape[0]
        size_d = d.shape[0]

        # 合并数据
        data = np.vstack((a, b, c, d))

        # 应用t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)

        # 绘制t-SNE结果
        plt.figure(figsize=(10, 5))
        plt.scatter(data_tsne[:size_a, 0], data_tsne[:size_a, 1], c='blue', label='A Data', s=1)
        plt.scatter(data_tsne[size_a:size_a+size_b, 0], data_tsne[size_a:size_a+size_b, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[size_a+size_b:size_a+size_b+size_c, 0], data_tsne[size_a+size_b:size_a+size_b+size_c, 1], c='green', label='C Data', s=1)
        plt.scatter(data_tsne[-size_d:, 0], data_tsne[-size_d:, 1], c='yellow', label='D Data', s=10)
        plt.legend()
        plt.title('t-SNE Visualization')
        
        # 保存图像
        output_path = '/data/shihuazhan/output_wsi/tsne4/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))

        return random_number


    
    def getLoss(self,a,x):
        L=cosine_similarity_loss(a,x)
        return L
    
    def forward_loss(self, student_cls_feat, teacher_cls_feat):
        if teacher_cls_feat is not None:
            #cls_loss = self.cl_loss(student_cls_feat,teacher_cls_feat.detach())
            cls_loss = self.cl_loss(student_cls_feat,teacher_cls_feat)
        else:
            cls_loss = 0.
        
        return cls_loss
    
    
    def forward(self, x):
        if self.training:
            a = self.Diffusion_reembed(x)
            x = x.squeeze()
            x3=x
            # tsnex= x.squeeze()
            # a=self.fusion_model(a,x)  
            # """
            # 此处为tsne画图
            # """
            # results = []
            # for _ in range(10):
            #     aa = self.Diffusion_reembed(x)
            #     aa=self.fusion_model(a,x) 
            #     results.append(aa)
            # b = torch.stack(results, dim=0)
            # nothing=self.drawTsne200(tsnex,b)
            # """
            # 此处为tsne画图
            # """
            #Loss = self.getLoss(a=a,x=x)
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]

            x = result.view(-1, 1024)
            x2= result_high.view(-1, 1024)
            x2 = x
            
            #x2= result.view(-1, 1024)
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)
            x3=self.embedding1(x3)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            A3 = self.attention(x3)
            A3 = torch.transpose(A3,-1,-2)
            A3 = F.softmax(A3, dim=-1)


            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)
            x3 = torch.matmul(A3,x3)

            x1 = self.head(x.squeeze(1))
            x2 = self.head(x2.squeeze(1))
            x3 = self.head(x3.squeeze(1))
            Loss=self.forward_loss(x1, x3)
            max_index_x1 = torch.argmax(x1, dim=1)
            max_index_x2 = torch.argmax(x2, dim=1)
            same_max_dimension = max_index_x1 == max_index_x2
            # result = torch.where(same_max_dimension, x1 - x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1))
            result = torch.where(same_max_dimension, x1 - x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))
            x=result


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))
                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                # result = torch.where(same_max_dimension, x1 - x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1))
                result = torch.where(same_max_dimension, x1 - x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))
                x=result
                Loss=0.

            else:
                a = self.Diffusion_reembed(x)
                x = x.squeeze()
                
                a=self.fusion_model(a,x)  
                Loss = self.getLoss(a=a,x=x)
                b = x
                cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
                sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
                sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                k_indices = sorted_indices[:num_elements]
                k_indices_high = sorted_indices_high[:num_elements]
                result = b[k_indices]
                result_high = b[k_indices_high]

                x = result.view(-1, 1024)
                x2=result_high.view(-1, 1024)
                # b,p,n = x.size()
                x = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x2)

                A = self.attention(x)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x = torch.matmul(A,x)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x.squeeze(1))
                x2 = self.head(x2.squeeze(1))
                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1 + x2, torch.where(x1[:, 1] > x2[:, 1], x1 + x2, x2 + x1))
                x=result


        # return {'loss': Loss,'output': x}
        return x,Loss
    

class DAttentionWithDiffEnd(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEnd, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        self.head2 = nn.Linear(512,out_dim)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/home/huangsheng/shihuazhan/pretrained/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

        # for conch
        self.emdproj=FCLayer(dropout=0.25,act='relu',in_size=512)



    @torch.no_grad()
    def Diffusion_reembed_withInfomin(self,x): #实现思路：min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfomax(self,x): #实现思路：max噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples




    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples

    def drawTsne200(self,a,b):
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))
        b = b.reshape(-1, 1024)
        data = np.vstack((a, b))
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)
        plt.figure(figsize=(10, 5))
        # plt.scatter(data_tsne[:-1, 0], data_tsne[:-1, 1], c='blue', label='A Data',s=1)
        # plt.scatter(data_tsne[-1, 0], data_tsne[-1, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[:-200, 0], data_tsne[:-200, 1], c='blue', label='A Data',s=1)
        plt.scatter(data_tsne[-200:, 0], data_tsne[-200:, 1], c='red', label='B Data', s=1)
        plt.legend()
        plt.title('t-SNE Visualization')
        #output_path = '/nas/zhangxiaoxian/output/mil_shz/tsne_200/tsne_visualization_{}.png'.format(random_number)
        output_path = '/data/shihuazhan/output_wsi/tsne_200/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))
        
        return random_number


     
    def drawTsne(self, a, b, c, d):
        # 将数据转移到CPU并转换为numpy数组
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        c = c.cpu().numpy()
        d = d.cpu().numpy()

        # 生成一个随机数作为文件名的一部分
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))

        # 将数据重塑为二维数组
        c = c.reshape(-1, 1024)
        b = b.reshape(-1, 1024)
        a = a.reshape(-1, 1024)
        d = d.reshape(-1, 1024)

        # 计算每个数据集的大小
        size_a = a.shape[0]
        size_b = b.shape[0]
        size_c = c.shape[0]
        size_d = d.shape[0]

        # 合并数据
        data = np.vstack((a, b, c, d))

        # 应用t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)

        # 绘制t-SNE结果
        plt.figure(figsize=(10, 5))
        plt.scatter(data_tsne[:size_a, 0], data_tsne[:size_a, 1], c='blue', label='A Data', s=1)
        plt.scatter(data_tsne[size_a:size_a+size_b, 0], data_tsne[size_a:size_a+size_b, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[size_a+size_b:size_a+size_b+size_c, 0], data_tsne[size_a+size_b:size_a+size_b+size_c, 1], c='green', label='C Data', s=1)
        plt.scatter(data_tsne[-size_d:, 0], data_tsne[-size_d:, 1], c='yellow', label='D Data', s=10)
        plt.legend()
        plt.title('t-SNE Visualization')
        
        # 保存图像
        output_path = '/data/shihuazhan/output_wsi/tsne4/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))

        return random_number

       


    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a=self.Diffusion_reembed_withInfomin(x)
            elif self.ifrand ==1:
                a=self.Diffusion_reembed_withInfomax(x)
            else:
                a=self.Diffusion_reembed_withInfoMean(x)
            x = x.squeeze()
            # print(x.shape)

            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result=self.head(a_embed)
            for param in self.embedding2.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
            #tsnex= x.squeeze()
            # """
            # 此处为tsne画图
            # """
            # results = []
            # for _ in range(200):
            #     aa = self.Diffusion_reembed(x)
            #     results.append(aa)
            # b = torch.stack(results, dim=0)
            # nothing=self.drawTsne200(tsnex,b)
            # """
            # 此处为tsne画图
            # """
            b = x
            # cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            
            # mask = torch.zeros_like(cosine_similarity)
            # probabilities = F.softmax(-cosine_similarity, dim=0)
            # k = int(b.size(0) * self.k_ratio)

            # selected_indices = torch.multinomial(probabilities, num_samples=k, replacement=False)
            # mask = torch.zeros_like(b, dtype=torch.bool)
            # mask[selected_indices] = True
            # result = b * mask
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
                        
            # k = self.k_ratio
            # sorted_cosine_similarity, _ = torch.sort(cosine_similarity)
            # threshold_index = int(k * len(cosine_similarity))
            # threshold = sorted_cosine_similarity[threshold_index]
            # mask_high = cosine_similarity > threshold
            # mask_low = cosine_similarity < threshold
            # result_igh = x[mask_high.unsqueeze(1).expand_as(x)].contiguous()
            # result = x[mask_low.unsqueeze(1).expand_as(x)].contiguous()




            # """
            # 此处为tsne画图
            # """
 
            # nothing=self.drawTsne(tsnex,result,result_high,a)
            # """
            # 此处为tsne画图
            # """

            #boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
            x = result.view(-1, 1024)
            x2=result_high.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x1 = self.head(x.squeeze(1))
            x2 = self.head(x2.squeeze(1))
            max_index_x1 = torch.argmax(x1, dim=1)
            max_index_x2 = torch.argmax(x2, dim=1)
            same_max_dimension = max_index_x1 == max_index_x2
            result = torch.where(same_max_dimension, x1 - x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))

            # if same_max_dimension:
            #     # 获取最大维度的索引
            #     max_dim_index = max_index_x1
            #     # print(x1)
            #     # print(x2)
                
            #     # 在最大维度上比较x1和x2的对应值
            #     x1_max_dim_values = x1[:, max_dim_index]
            #     x2_max_dim_values = x2[:, max_dim_index]
            #     #comparison = x1_max_dim_values > x2_max_dim_values
            #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
            #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
            #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
            #                   x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
            #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
            #     # 使用较大的值减去较小的值
            #     # print(result)
            #     # print("_________________________________________________________")
            # else:
            #     # 如果最大维度不同，保持原来的操作
            #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)







            # print("x1 is ",end='')
            # print(x1)
            # print("x2 is ",end='')
            # print(x2)
            # print("是否相同 is ",end='')
            # print(same_max_dimension)
            # print("最终结果")
            # print(result)
            # print("_______________________________________________")
            x=result


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))




                x=result
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result



    def forward_tsne(self, x,b_label):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x=x.to(device)
        self.to(device)
        a=self.Diffusion_reembed_withInfoMean(x)
        x = x.squeeze()
        # print(x.shape)
        a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
        b = x

        cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
        sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
        sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
        k = self.k_ratio
        num_elements = int(torch.tensor(b.size(0)) * k)
        k_indices = sorted_indices[:num_elements]
        k_indices_high = sorted_indices_high[:num_elements]


        result = b[k_indices]
        result_high = b[k_indices_high]
        x = result.view(-1, 1024)
        #x2=result_high.view(-1, 1024)
        result_label = [b_label[idx] for idx in k_indices]
        #result_high_label = [b_label[idx] for idx in k_indices_high]

       
        b_new = np.full_like(b_label, 2)
        b_new[k_indices.cpu().numpy()] = b_label[k_indices.cpu().numpy()]

        #

        return x,result_label,b_new



    def forward_test(self, x):
        if self.training:
            return 'error'
        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))
                #result = x1
                # if same_max_dimension:
                # # 获取最大维度的索引
                #     max_dim_index = max_index_x1
                #     # print(x1)
                #     # print(x2)
                    
                #     # 在最大维度上比较x1和x2的对应值
                #     x1_max_dim_values = x1[:, max_dim_index]
                #     x2_max_dim_values = x2[:, max_dim_index]
                #     #comparison = x1_max_dim_values > x2_max_dim_values
                #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
                #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
                #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
                #                 x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
                #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
                # # 使用较大的值减去较小的值
                # # print(result)
                # # print("_________________________________________________________")
                # else:
                #     # 如果最大维度不同，保持原来的操作
                #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)



                x=result
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result, A, A2
    





class DAttentionWithDiffEndTransmil(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndTransmil, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.Transmodel2 = TransMIL(n_classes=out_dim,dropout=False,act='relu')
        self.Transmodel1 = TransMIL(n_classes=out_dim,dropout=False,act='relu')
        self.head = nn.Linear(512,out_dim)

        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/data3/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/home/huangsheng/shihuazhan/pretrained/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()



    @torch.no_grad()
    def Diffusion_reembed_withInfomin(self,x): #实现思路：min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfomax(self,x): #实现思路：max噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples




    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
 
 

    @torch.no_grad()
    def Diffusion_reembed_withRandom(self, x):
        output_channels = 4
        # 实例维度转换
        x = x.squeeze()  # (n, 1024)
        n = x.size(0)
        x = x.view(n, 32, 32)

        # 创建扩散模型实例
        diffusion = create_diffusion(str(self.t_steps))

        # 生成随机噪声
        noise = torch.randn((1, output_channels, 32, 32), device=x.device)

        # 执行扩散模型的前向传播，生成样本
        samples = diffusion.p_sample_loop(
            self.Dit.forward_unconditional_for_wsi2,
            shape=noise.shape,
            noise=noise,
            clip_denoised=False,
            progress=True,
            device=x.device
        )

        # 调整样本形状并计算平均值
        new_shape = (1, 4, 1024)
        samples = samples.view(new_shape)
        samples = samples.mean(dim=1)  # (1, 1024)

        return samples

     
 

       

    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a=self.Diffusion_reembed_withRandom(x)
            elif self.ifrand ==1:
                a=self.Diffusion_reembed_withInfomax(x)
            elif self.ifrand ==3:
                a=self.Diffusion_reembed_withInfomin(x)
            else:
                a=self.Diffusion_reembed_withInfoMean(x)
            x = x.squeeze()
            # print(x.shape)
            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            

            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed = self.Transmodel2._fc1(a)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result=self.head(a_embed)
 
            b = x
 
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]

            x = result.view(-1, 1024)
            x2= result_high.view(-1, 1024)
            x2= x
            result=self.head(self.Transmodel1.forward_feature(x))-self.head(self.Transmodel2.forward_feature(x))
            # result=self.head(self.Transmodel2.forward_feature(x))
            # print(result.shape)
            x=result


        else:
            if self.ifTrain == 1:
                
                result = self.head(self.Transmodel1.forward_feature(x))-self.head(self.Transmodel2.forward_feature(x))
                # result=self.head(self.Transmodel2.forward_feature(x))
                x=result
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result



    def forward_tsne(self, x,b_label):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x=x.to(device)
        self.to(device)
        a=self.Diffusion_reembed_withInfoMean(x)
        x = x.squeeze()
        # print(x.shape)
        a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
        b = x

        cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
        sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
        sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
        k = self.k_ratio
        num_elements = int(torch.tensor(b.size(0)) * k)
        k_indices = sorted_indices[:num_elements]
        k_indices_high = sorted_indices_high[:num_elements]


        result = b[k_indices]
        result_high = b[k_indices_high]
        x = result.view(-1, 1024)
        #x2=result_high.view(-1, 1024)
        result_label = [b_label[idx] for idx in k_indices]
        #result_high_label = [b_label[idx] for idx in k_indices_high]

       
        b_new = np.full_like(b_label, 2)
        b_new[k_indices.cpu().numpy()] = b_label[k_indices.cpu().numpy()]

        #

        return x,result_label,b_new



    def forward_test(self, x):
        if self.training:
            return 'error'
        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))
                #result = x1
                # if same_max_dimension:
                # # 获取最大维度的索引
                #     max_dim_index = max_index_x1
                #     # print(x1)
                #     # print(x2)
                    
                #     # 在最大维度上比较x1和x2的对应值
                #     x1_max_dim_values = x1[:, max_dim_index]
                #     x2_max_dim_values = x2[:, max_dim_index]
                #     #comparison = x1_max_dim_values > x2_max_dim_values
                #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
                #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
                #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
                #                 x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
                #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
                # # 使用较大的值减去较小的值
                # # print(result)
                # # print("_________________________________________________________")
                # else:
                #     # 如果最大维度不同，保持原来的操作
                #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)



                x=result
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result, A, A2
 



class TransmilWithMining(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifrand=0,ifTrain=1):
        super(TransmilWithMining, self).__init__()
        self.Transmodel2 = TransMIL(n_classes=out_dim,dropout=False,act='relu')
        self.Transmodel1 = TransMIL(n_classes=out_dim,dropout=False,act='relu')
        self.head = nn.Linear(512,out_dim)
        self.ifrand = ifrand
        self.k_ratio=k_ratio
        self.ifTrain = ifTrain




    def random(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) 
        x_pooled = x[random_index]
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled

    @torch.no_grad()
    def mean(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        x_pooled = torch.mean(x, dim=0)  # 在第0维上计算平均值 
        
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled

    @torch.no_grad()
    def max(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        x_pooled,_ = torch.max(x, dim=0)  # 在第0维上计算平均值 
        
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled

    @torch.no_grad()
    def minum(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        x_pooled,_ = torch.min(x, dim=0)  # 在第0维上计算平均值 
        
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled

    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a=self.random(x)
            elif self.ifrand ==1:
                a=self.mean(x)
            elif self.ifrand ==3:
                a=self.minum(x)
            else:
                a=self.max(x)
            x = x.squeeze()
            # a_embed = self.Transmodel2._fc1(a)
            # a_result=self.head(a_embed)
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
            x = result.view(-1, 1024)

            result=self.head(self.Transmodel1.forward_feature(x))-self.head(self.Transmodel2.forward_feature(x))
            x=result
            
        else:
            if self.ifTrain == 1:
                
                result = self.head(self.Transmodel1.forward_feature(x))-self.head(self.Transmodel2.forward_feature(x))
                x=result
                
            else:
                a = self.Diffusion_reembed(x)
        return x




class DualTransmilforExperiment(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DualTransmilforExperiment, self).__init__()
        self.Transmodel2 = TransMIL(n_classes=out_dim,dropout=False,act='relu')
        self.Transmodel1 = TransMIL(n_classes=out_dim,dropout=False,act='relu')
        self.head = nn.Linear(512,out_dim)
    
    def forward(self, x):
        if self.training:
            result = self.head(self.Transmodel1.forward_feature(x))-self.head(self.Transmodel2.forward_feature(x))
            x=result

        else:
            result = self.head(self.Transmodel1.forward_feature(x))-self.head(self.Transmodel2.forward_feature(x))
            x=result
        return x








class DAttentionWithDiffTwoEnd(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0):
        super(DAttentionWithDiffTwoEnd, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter_p = Adapter(1024, 4)
        self.adapter_n = Adapter(1024, 4)
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        #self.head2 = nn.Linear(512,out_dim)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        # ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

    @torch.no_grad()
    def Diffusion_reembed(self,x): #实现思路：随机噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        x_pooled = x_pooled.repeat(1, 4, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=x_pooled.shape,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
    




    def drawTsne200(self,a,b):
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))
        b = b.reshape(-1, 1024)
        data = np.vstack((a, b))
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)
        plt.figure(figsize=(10, 5))
        # plt.scatter(data_tsne[:-1, 0], data_tsne[:-1, 1], c='blue', label='A Data',s=1)
        # plt.scatter(data_tsne[-1, 0], data_tsne[-1, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[:-200, 0], data_tsne[:-200, 1], c='blue', label='A Data',s=1)
        plt.scatter(data_tsne[-200:, 0], data_tsne[-200:, 1], c='red', label='B Data', s=1)
        plt.legend()
        plt.title('t-SNE Visualization')
        #output_path = '/nas/zhangxiaoxian/output/mil_shz/tsne_200/tsne_visualization_{}.png'.format(random_number)
        output_path = '/data/shihuazhan/output_wsi/tsne_200/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))
        
        return random_number


     
    def drawTsne(self, a, b, c, d):
        # 将数据转移到CPU并转换为numpy数组
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        c = c.cpu().numpy()
        d = d.cpu().numpy()

        # 生成一个随机数作为文件名的一部分
        random_number = random.randint(1, 20000)
        plt.title('t-SNE Visualization ({})'.format(random_number))

        # 将数据重塑为二维数组
        c = c.reshape(-1, 1024)
        b = b.reshape(-1, 1024)
        a = a.reshape(-1, 1024)
        d = d.reshape(-1, 1024)

        # 计算每个数据集的大小
        size_a = a.shape[0]
        size_b = b.shape[0]
        size_c = c.shape[0]
        size_d = d.shape[0]

        # 合并数据
        data = np.vstack((a, b, c, d))

        # 应用t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
        data_tsne = tsne.fit_transform(data)

        # 绘制t-SNE结果
        plt.figure(figsize=(10, 5))
        plt.scatter(data_tsne[:size_a, 0], data_tsne[:size_a, 1], c='blue', label='A Data', s=1)
        plt.scatter(data_tsne[size_a:size_a+size_b, 0], data_tsne[size_a:size_a+size_b, 1], c='red', label='B Data', s=1)
        plt.scatter(data_tsne[size_a+size_b:size_a+size_b+size_c, 0], data_tsne[size_a+size_b:size_a+size_b+size_c, 1], c='green', label='C Data', s=1)
        plt.scatter(data_tsne[-size_d:, 0], data_tsne[-size_d:, 1], c='yellow', label='D Data', s=10)
        plt.legend()
        plt.title('t-SNE Visualization')
        
        # 保存图像
        output_path = '/data/shihuazhan/output_wsi/tsne4/tsne_visualization_{}.png'.format(random_number)
        plt.savefig(output_path)
        print("Image saved to {}".format(output_path))

        return random_number

       


    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a = self.Diffusion_reembed(x)
                a2=self.Diffusion_reembed(x)
            # save_path = '/data/zhangxiaoxian/output/pretrained_models/'
            # file_name1 = 'tensor_a.pt'
            # file_name2 = 'tensor_a2.pt'
            # full_path1 = os.path.join(save_path, file_name1)
            # full_path2 = os.path.join(save_path, file_name2)
            # a=torch.load(full_path1)
            # a2=torch.load(full_path2)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            a=a.to(device)
            a2=a2.to(device)

            x = x.squeeze()
            # print(x.shape)

            a_p=a+self.adapter_p(a)
            a_n=a+self.adapter_n(a2)   
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed_p=self.embedding1(a_p)
            a_embed_n=self.embedding2(a_n)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result_p=self.head(a_embed_p)
            a_result_n=self.head(a_embed_n)
            # for param in self.embedding2.parameters():
            #     param.requires_grad = True
            # for param in self.head.parameters():
            #     param.requires_grad = True
            #tsnex= x.squeeze()
            # """
            # 此处为tsne画图
            # """
            # results = []
            # for _ in range(200):
            #     aa = self.Diffusion_reembed(x)
            #     results.append(aa)
            # b = torch.stack(results, dim=0)
            # nothing=self.drawTsne200(tsnex,b)
            # """
            # 此处为tsne画图
            # """
            b = x
            # cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            
            # mask = torch.zeros_like(cosine_similarity)
            # probabilities = F.softmax(-cosine_similarity, dim=0)
            # k = int(b.size(0) * self.k_ratio)

            # selected_indices = torch.multinomial(probabilities, num_samples=k, replacement=False)
            # mask = torch.zeros_like(b, dtype=torch.bool)
            # mask[selected_indices] = True
            # result = b * mask
            cosine_similarity_p = F.cosine_similarity(a_p.expand_as(b), b, dim=1)
            cosine_similarity_n = F.cosine_similarity(a_n.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity_n, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity_p, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            common_indices_set = set(k_indices.tolist()).intersection(set(k_indices_high.tolist()))
            k_indices_share = torch.tensor(list(common_indices_set))
            # num_common_indices = len(k_indices_share)
            # proportion_common_indices = num_common_indices / (b.size(0))
            # print(num_common_indices)
            # print(b.size(0))
            # print(proportion_common_indices)
            # print("_______________")
            result = b[k_indices_share]
            # result_high = b[k_indices_high]
                        
            # k = self.k_ratio
            # sorted_cosine_similarity, _ = torch.sort(cosine_similarity)
            # threshold_index = int(k * len(cosine_similarity))
            # threshold = sorted_cosine_similarity[threshold_index]
            # mask_high = cosine_similarity > threshold
            # mask_low = cosine_similarity < threshold
            # result_igh = x[mask_high.unsqueeze(1).expand_as(x)].contiguous()
            # result = x[mask_low.unsqueeze(1).expand_as(x)].contiguous()




            # """
            # 此处为tsne画图
            # """
 
            # nothing=self.drawTsne(tsnex,result,result_high,a)
            # """
            # 此处为tsne画图
            # """

            #boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
            x = result.view(-1, 1024)
             
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x1 = self.head(x.squeeze(1))
            x2 = self.head(x2.squeeze(1))
            max_index_x1 = torch.argmax(x1, dim=1)
            max_index_x2 = torch.argmax(x2, dim=1)
            same_max_dimension = max_index_x1 == max_index_x2
            result = torch.where(same_max_dimension, x1 - x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))

            # if same_max_dimension:
            #     # 获取最大维度的索引
            #     max_dim_index = max_index_x1
            #     # print(x1)
            #     # print(x2)
                
            #     # 在最大维度上比较x1和x2的对应值
            #     x1_max_dim_values = x1[:, max_dim_index]
            #     x2_max_dim_values = x2[:, max_dim_index]
            #     #comparison = x1_max_dim_values > x2_max_dim_values
            #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
            #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
            #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
            #                   x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
            #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
            #     # 使用较大的值减去较小的值
            #     # print(result)
            #     # print("_________________________________________________________")
            # else:
            #     # 如果最大维度不同，保持原来的操作
            #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)







            # print("x1 is ",end='')
            # print(x1)
            # print("x2 is ",end='')
            # print(x2)
            # print("是否相同 is ",end='')
            # print(same_max_dimension)
            # print("最终结果")
            # print(result)
            # print("_______________________________________________")
            x=result


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))
                x=result
                a_result_p=result
                a_result_n=result

            

        return x,a_result_p,a_result_n
        # return x,a_result_p



class DAttentionWithDiff(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiff, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        self.head2 = nn.Linear(512,out_dim)
        



    def random(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) 
        x_pooled = x[random_index]
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled

    @torch.no_grad()
    def mean(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        x_pooled = torch.mean(x, dim=0)  # 在第0维上计算平均值 
        
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled

    @torch.no_grad()
    def max(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        x_pooled,_ = torch.max(x, dim=0)  # 在第0维上计算平均值 
        
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled

    @torch.no_grad()
    def minum(self,x):
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        x = x.view(n, 32, 32)
        x_pooled,_ = torch.min(x, dim=0)  # 在第0维上计算平均值 
        
        # print(x_pooled.shape)
        x_pooled = x_pooled.view(1,1024)
        # print(x_pooled.shape)
        # print("****************************")
        return x_pooled
    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))
                return result
            elif self.ifrand ==1:
                a=self.random(x)
            elif self.ifrand ==2:
                a=self.max(x)
            elif self.ifrand ==3:
                a=self.mean(x)
            elif self.ifrand ==4:
                a=self.minum(x)
             
            x = x.squeeze()

            b = x

            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]

            x = result.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N
            
            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x1 = self.head(x.squeeze(1))
            x2 = self.head(x2.squeeze(1))
            result = x1-x2
            x= result
                        
          


        else:

            # b,p,n = x.size()
            x1  = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x)

            A = self.attention(x1)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            x1 = torch.matmul(A,x1)
            x2 = torch.matmul(A2,x2)

            x1 = self.head(x1.squeeze(1))
            x2 = self.head(x2.squeeze(1))

            max_index_x1 = torch.argmax(x1, dim=1)
            max_index_x2 = torch.argmax(x2, dim=1)
            same_max_dimension = max_index_x1 == max_index_x2
            result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))



            x=result


            

        return x



    



class DAttentionWithDiffEndSur(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndSur, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        self.head2 = nn.Linear(512,out_dim)
        self.head3 = nn.Linear(512,out_dim)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()



    @torch.no_grad()
    def Diffusion_reembed_withInfomin(self,x): #实现思路：min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfomax(self,x): #实现思路：max噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples




    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples



       


    
    def forward(self, x):
        if self.training:
            a_model_results=[]
            if self.ifrand == 0:
                a=self.Diffusion_reembed_withInfomin(x)
            elif self.ifrand ==1:
                a=self.Diffusion_reembed_withInfomax(x)
            else:
                for i in range (self.a_num):
                    a=self.Diffusion_reembed_withInfoMean(x)
                    a_model_results.append(a)
            x = x.squeeze()
            # print(x.shape)
            a=a_model_results[0] #shz 这里只是暂时的权衡之计，在生成很多个a时，只取第一个用于微调
            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result=self.head(a_embed)
            for param in self.embedding2.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
            #tsnex= x.squeeze()
            # """
            # 此处为tsne画图
            # """
            # results = []
            # for _ in range(200):
            #     aa = self.Diffusion_reembed(x)
            #     results.append(aa)
            # b = torch.stack(results, dim=0)
            # nothing=self.drawTsne200(tsnex,b)
            # """
            # 此处为tsne画图
            # """
            b = x
            # cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            
            # mask = torch.zeros_like(cosine_similarity)
            # probabilities = F.softmax(-cosine_similarity, dim=0)
            # k = int(b.size(0) * self.k_ratio)

            # selected_indices = torch.multinomial(probabilities, num_samples=k, replacement=False)
            # mask = torch.zeros_like(b, dtype=torch.bool)
            # mask[selected_indices] = True
            # result = b * mask
            #a_model=select_features(a_model_results,self.a_num) 
            #a=a_model_results[0]
            a_model = torch.stack(a_model_results).squeeze(1)
            a_model= self.a_ratio * a_model + self.adapter_ratio * self.adapter(a_model)  
            if self.a_num == 1:
                cosine_similarity = F.cosine_similarity(a_model.expand_as(b), b, dim=1)
                sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
                sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                k_indices = sorted_indices[:num_elements]
                k_indices_high = sorted_indices_high[:num_elements]
            else:
                cosine_similarity_matrix = F.cosine_similarity(a_model.unsqueeze(1), b.unsqueeze(0), dim=2)  # [n, 1000]
                # 综合考量a中的每个向量对于b的相似度
                # 例如，我们可以取平均值或者最大值，这里我们取平均值
                average_similarity = torch.mean(cosine_similarity_matrix, dim=0, keepdim=True)  # [1, 1000]
                # 对综合相似度矩阵进行排序，以找到最相似和最不相似的b中向量
                sorted_indices = torch.argsort(average_similarity, dim=1)  # [1, 1000]
                sorted_indices_high = torch.argsort(average_similarity, dim=1, descending=True)  # [1, 1000]

                # 计算需要选择的元素数量
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)

                # 获取最不相似和最相似b中向量的索引
                k_indices = sorted_indices[0, :num_elements]
                k_indices_high = sorted_indices_high[0, :num_elements]

            result = b[k_indices]
            result_high = b[k_indices_high]
                        
            # k = self.k_ratio
            # sorted_cosine_similarity, _ = torch.sort(cosine_similarity)
            # threshold_index = int(k * len(cosine_similarity))
            # threshold = sorted_cosine_similarity[threshold_index]
            # mask_high = cosine_similarity > threshold
            # mask_low = cosine_similarity < threshold
            # result_igh = x[mask_high.unsqueeze(1).expand_as(x)].contiguous()
            # result = x[mask_low.unsqueeze(1).expand_as(x)].contiguous()




            # """
            # 此处为tsne画图
            # """
 
            # nothing=self.drawTsne(tsnex,result,result_high,a)
            # """
            # 此处为tsne画图
            # """

            #boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
            x = result.view(-1, 1024)
            x2=result_high.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N
            
            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x1 = self.head(x.squeeze(1))
            x2 = self.head(x2.squeeze(1))
          
            result = x1-x2
            result_sur=self.head3(x.squeeze(1))
            # if same_max_dimension:
            #     # 获取最大维度的索引
            #     max_dim_index = max_index_x1
            #     # print(x1)
            #     # print(x2)
                
            #     # 在最大维度上比较x1和x2的对应值
            #     x1_max_dim_values = x1[:, max_dim_index]
            #     x2_max_dim_values = x2[:, max_dim_index]
            #     #comparison = x1_max_dim_values > x2_max_dim_values
            #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
            #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
            #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
            #                   x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
            #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
            #     # 使用较大的值减去较小的值
            #     # print(result)
            #     # print("_________________________________________________________")
            # else:
            #     # 如果最大维度不同，保持原来的操作
            #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)







            # print("x1 is ",end='')
            # print(x1)
            # print("x2 is ",end='')
            # print(x2)
            # print("是否相同 is ",end='')
            # print(same_max_dimension)
            # print("最终结果")
            # print(result)
            # print("_______________________________________________")
            


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)



                result_sur=self.head3(x1.squeeze(1))
                result=result_sur 
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return result,a_result,result_sur


 
    

class DAttentionWithDiffEndSur2(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndSur2, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,2)
        self.head2 = nn.Linear(512,out_dim)
        #self.head3 = nn.Linear(512,out_dim)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        ckpt_path = "/home/huangsheng/shihuazhan/pretrained/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()



    @torch.no_grad()
    def Diffusion_reembed_withInfomin(self,x): #实现思路：min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfomax_min(self,x): #实现思路：max噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector-min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples




    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples



       


    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a=self.Diffusion_reembed_withInfomin(x)
            elif self.ifrand ==1:
                a=self.Diffusion_reembed_withInfomax_min(x)
            else:

                a=self.Diffusion_reembed_withInfoMean(x)
                    
            x = x.squeeze()
            # print(x.shape)
            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result=self.head2(a_embed) #这里具体用哪个head还不是很确定，可以两个都试一下

            b = x

            a_model= a
            if self.a_num == 1:
                cosine_similarity = F.cosine_similarity(a_model.expand_as(b), b, dim=1)
                sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
                sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                k_indices = sorted_indices[:num_elements]
                k_indices_high = sorted_indices_high[:num_elements]

            result = b[k_indices]
            result_high = b[k_indices_high]

            x = result.view(-1, 1024)
            x2=result_high.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N
            
            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x1_f = self.head(x.squeeze(1))
            x2_f = self.head(x2.squeeze(1))
          
            result = x1_f-x2_f
            result_sur1=self.head2(x.squeeze(1))
            result_sur2=self.head2(x2.squeeze(1))
            result_sur=result_sur1-result_sur2
            


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)



                result_sur=self.head2(x1.squeeze(1))-self.head2(x2.squeeze(1))
                result=result_sur 
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return result,a_result,result_sur





class DAttentionWithDiffEndSur2Transmil(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndSur2Transmil, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,2)
        self.head2 = nn.Linear(512,out_dim)
        #self.head3 = nn.Linear(512,out_dim)
        self.Transmodel = TransMIL(n_classes=4,dropout=False,act='relu')
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()



    @torch.no_grad()
    def Diffusion_reembed_withInfomin(self,x): #实现思路：min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfomax_min(self,x): #实现思路：max噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector-min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples




    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples



       


    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a=self.Diffusion_reembed_withInfomin(x)
            elif self.ifrand ==1:
                a=self.Diffusion_reembed_withInfomax_min(x)
            else:

                a=self.Diffusion_reembed_withInfoMean(x)
                    
            x = x.squeeze()
            # print(x.shape)
            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a)

            # a_result=self.head(a_embed)

            b = x

            a_model= a
            if self.a_num == 1:
                cosine_similarity = F.cosine_similarity(a_model.expand_as(b), b, dim=1)
                sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
                sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                k_indices = sorted_indices[:num_elements]
                k_indices_high = sorted_indices_high[:num_elements]
            else:
                print("w")

            result = b[k_indices]
            result_high = b[k_indices_high]
            result_trans=self.Transmodel(result)          
            result = result_trans
            result_sur=result_trans
            a_result=result_trans
            


        else:
            if self.ifTrain == 1:
                result_trans=self.Transmodel(x)
                result_sur=result_trans
                result=result_trans 
                a_result=result_trans

            

        return result,a_result,result_sur





class DAttentionWithDiffEndSur3Transmil(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndSur3Transmil, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.Transmodel = TransMIL(n_classes=4,dropout=False,act='relu')
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()



    @torch.no_grad()
    def Diffusion_reembed_withInfomin(self,x): #实现思路：min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfomax_min(self,x): #实现思路：max噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector-min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples




    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples



       


    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a=self.Diffusion_reembed_withInfomin(x)
            elif self.ifrand ==1:
                a=self.Diffusion_reembed_withInfomax_min(x)
            else:

                a=self.Diffusion_reembed_withInfoMean(x)
                    
            x = x.squeeze()
            # print(x.shape)
            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
 

            b = x

            a_model= a
            if self.a_num == 1:
                cosine_similarity = F.cosine_similarity(a_model.expand_as(b), b, dim=1)
                sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
                sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                k_indices = sorted_indices[:num_elements]
                k_indices_high = sorted_indices_high[:num_elements]
            else:
                print("w")

            result = b[k_indices]
            result_high = b[k_indices_high]
            result_trans=self.Transmodel(result)          
            result = result_trans
            result_sur=result_trans
            a_result=result_trans
            


        else:
            if self.ifTrain == 1:
                result_trans=self.Transmodel(x)
                result_sur=result_trans
                result=result_trans 
                a_result=result_trans

            

        return result,a_result,result_sur











class DAttentionWithDiffEndSur4Transmil(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndSur4Transmil, self).__init__()
        #self.embedding1 = FCLayer()
        #self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.Transmodel2 = TransMIL(n_classes=4,dropout=False,act='relu')
        self.Transmodel1 = TransMIL(n_classes=4,dropout=False,act='relu')
        self.head = nn.Linear(512,out_dim)
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        ckpt_path = "/data3/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()



    @torch.no_grad()
    def Diffusion_reembed_withInfomin(self,x): #实现思路：min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfomax_min(self,x): #实现思路：max噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector-min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples




    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples



       


    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a=self.Diffusion_reembed_withInfomin(x)
            elif self.ifrand ==1:
                a=self.Diffusion_reembed_withInfomax_min(x)
            else:

                a=self.Diffusion_reembed_withInfoMean(x)
                    
            x = x.squeeze()
            # print(x.shape)
            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            a_embed = self.Transmodel2._fc1(a)
            a_result= self.head(a_embed)
            b = x

            a_model= a
            if self.a_num == 1:
                cosine_similarity = F.cosine_similarity(a_model.expand_as(b), b, dim=1)
                sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
                sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                k_indices = sorted_indices[:num_elements]
                k_indices_high = sorted_indices_high[:num_elements]
            else:
                print("w")

            result = b[k_indices]
            result_high = b[k_indices_high]
            result_trans=self.head(self.Transmodel1.forward_feature(result))-self.head(self.Transmodel2.forward_feature(result))
            result = result_trans
            result_sur=result_trans
            #a_result=result_trans
            


        else:
            if self.ifTrain == 1:
                result_trans=self.head(self.Transmodel1.forward_feature(x))-self.head(self.Transmodel2.forward_feature(x))
                result_sur=result_trans
                result=result_trans 
                a_result=result_trans

            

        return result,a_result,result_sur


class DAttentionWithDiffEndSur2T1(nn.Module): #T是指现在sur2用的之前那个不带T的是备份用的,现在跑单分支abmil #
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndSur2T1, self).__init__()
        # self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        #self.head3 = nn.Linear(512,out_dim)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        ckpt_path = "/home/huangsheng/shihuazhan/pretrained/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()



    @torch.no_grad()
    def Diffusion_reembed_withInfomin(self,x): #实现思路：min噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    @torch.no_grad()
    def Diffusion_reembed_withInfomax_min(self,x): #实现思路：max噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector-min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples




    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples



    
    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                a=self.Diffusion_reembed_withInfomin(x)
            elif self.ifrand ==1:
                a=self.Diffusion_reembed_withInfomax_min(x)
            else:

                a=self.Diffusion_reembed_withInfoMean(x)
                    
            x = x.squeeze()
            # print(x.shape)
            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            a_embed=self.embedding2(a)
            a_result=self.head(a_embed)

            b = x
            a_model= a
            if self.a_num == 1:
                cosine_similarity = F.cosine_similarity(a_model.expand_as(b), b, dim=1)
                sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
                sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                k_indices = sorted_indices[:num_elements]
                k_indices_high = sorted_indices_high[:num_elements]

            result = b[k_indices]
            result_high = b[k_indices_high]

            x = result.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            #x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            # A = self.attention(x)
            # A = torch.transpose(A, -1, -2)  # KxN
            # A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N
            
            # x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            # result_sur1=self.head(x.squeeze(1))
            result_sur2=self.head(x2.squeeze(1))
            result_sur=result_sur2
            result=result_sur
            


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                # x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                # A = self.attention(x1)
                # A = torch.transpose(A, -1, -2)  # KxN
                # A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                # x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)



                result_sur=self.head(x2.squeeze(1))
                result=result_sur 
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return result,a_result,result_sur






class fastDAttentionWithDiffEnd(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0):
        super(fastDAttentionWithDiffEnd, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        #self.head2 = nn.Linear(512,out_dim)
        
        # self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        # ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        # state_dict = find_model(ckpt_path)
        # self.Dit.load_state_dict(state_dict)
        # self.Dit.eval()

    @torch.no_grad()
    def Diffusion_reembed(self,x,Dit): #实现思路：随机噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n = x.size(0)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        x_pooled = x_pooled.repeat(1, 4, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        samples = diffusion.p_sample_loop(
        Dit.forward_unconditional_for_wsi2,shape=x_pooled.shape,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
    
    @torch.no_grad()
    def Diffusion_reembed_withInfo(self,x,Dit): #实现思路：max-min当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        min_vector = (torch.min(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        max_vector = (torch.max(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector=max_vector-min_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples

    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x,Dit): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
  

    
    def forward(self, x,Dit):
        if self.training:
            if self.ifrand == 0:
                a = self.Diffusion_reembed(x,Dit)
            elif self.ifrand ==1:
                a=self.Diffusion_reembed_withInfo(x,Dit)
            else:
                a=self.Diffusion_reembed_withInfoMean(x,Dit)
            x = x.squeeze()
            # print(x.shape)

            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result=self.head(a_embed)
            for param in self.embedding2.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
            #tsnex= x.squeeze()
            # """
            # 此处为tsne画图
            # """
            # results = []
            # for _ in range(200):
            #     aa = self.Diffusion_reembed(x)
            #     results.append(aa)
            # b = torch.stack(results, dim=0)
            # nothing=self.drawTsne200(tsnex,b)
            # """
            # 此处为tsne画图
            # """
            b = x
            # cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            
            # mask = torch.zeros_like(cosine_similarity)
            # probabilities = F.softmax(-cosine_similarity, dim=0)
            # k = int(b.size(0) * self.k_ratio)

            # selected_indices = torch.multinomial(probabilities, num_samples=k, replacement=False)
            # mask = torch.zeros_like(b, dtype=torch.bool)
            # mask[selected_indices] = True
            # result = b * mask
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
                        
            # k = self.k_ratio
            # sorted_cosine_similarity, _ = torch.sort(cosine_similarity)
            # threshold_index = int(k * len(cosine_similarity))
            # threshold = sorted_cosine_similarity[threshold_index]
            # mask_high = cosine_similarity > threshold
            # mask_low = cosine_similarity < threshold
            # result_igh = x[mask_high.unsqueeze(1).expand_as(x)].contiguous()
            # result = x[mask_low.unsqueeze(1).expand_as(x)].contiguous()




            # """
            # 此处为tsne画图
            # """
 
            # nothing=self.drawTsne(tsnex,result,result_high,a)
            # """
            # 此处为tsne画图
            # """

            #boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
            x = result.view(-1, 1024)
            x2=result_high.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x1 = self.head(x.squeeze(1))
            x2 = self.head(x2.squeeze(1))
            max_index_x1 = torch.argmax(x1, dim=1)
            max_index_x2 = torch.argmax(x2, dim=1)
            same_max_dimension = max_index_x1 == max_index_x2
            result = torch.where(same_max_dimension, x1 - x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))

            # if same_max_dimension:
            #     # 获取最大维度的索引
            #     max_dim_index = max_index_x1
            #     # print(x1)
            #     # print(x2)
                
            #     # 在最大维度上比较x1和x2的对应值
            #     x1_max_dim_values = x1[:, max_dim_index]
            #     x2_max_dim_values = x2[:, max_dim_index]
            #     #comparison = x1_max_dim_values > x2_max_dim_values
            #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
            #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
            #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
            #                   x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
            #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
            #     # 使用较大的值减去较小的值
            #     # print(result)
            #     # print("_________________________________________________________")
            # else:
            #     # 如果最大维度不同，保持原来的操作
            #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)







            # print("x1 is ",end='')
            # print(x1)
            # print("x2 is ",end='')
            # print(x2)
            # print("是否相同 is ",end='')
            # print(same_max_dimension)
            # print("最终结果")
            # print(result)
            # print("_______________________________________________")
            x=result


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))
                #result = x1
                # if same_max_dimension:
                # # 获取最大维度的索引
                #     max_dim_index = max_index_x1
                #     # print(x1)
                #     # print(x2)
                    
                #     # 在最大维度上比较x1和x2的对应值
                #     x1_max_dim_values = x1[:, max_dim_index]
                #     x2_max_dim_values = x2[:, max_dim_index]
                #     #comparison = x1_max_dim_values > x2_max_dim_values
                #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
                #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
                #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
                #                 x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
                #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
                # # 使用较大的值减去较小的值
                # # print(result)
                # # print("_________________________________________________________")
                # else:
                #     # 如果最大维度不同，保持原来的操作
                #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)



                x=result
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result
    
    def forward_tsne(self, x,b_label,Dit):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x=x.to(device)
        self.to(device)
        a=self.Diffusion_reembed_withInfoMean(x,Dit)
        x = x.squeeze()
        # print(x.shape)
        a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
        b = x

        cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
        sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
        sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
        k = self.k_ratio
        num_elements = int(torch.tensor(b.size(0)) * k)
        k_indices = sorted_indices[:num_elements]
        k_indices_high = sorted_indices_high[:num_elements]


        result = b[k_indices]
        result_high = b[k_indices_high]
        x = result.view(-1, 1024)
        #x2=result_high.view(-1, 1024)
        result_label = [b_label[idx] for idx in k_indices]
        #result_high_label = [b_label[idx] for idx in k_indices_high]

       
        b_new = np.full_like(b_label, 2)
        b_new[k_indices.cpu().numpy()] = b_label[k_indices.cpu().numpy()]

        #

        return x,result_label,b_new


    def forward_test(self, x):
        if self.training:
            return 'error'
        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))
                x=result
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result, A, A2










 
     











if __name__ == "__main__":
    model = DAttention()
    # x=torch.rand(1,2,1024)
    # print(model(x))
    for k,v in model.state_dict().items():
        print(k)


class DAttentionWithDiffEndForConch(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndForConch, self).__init__()
        self.embedding1 = FCLayer(dropout=0.25,act='relu',in_size=512)
        self.embedding2 = FCLayer(dropout=0.25,act='relu',in_size=512)
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024,512,4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        self.head2 = nn.Linear(512,out_dim)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/home/huangsheng/shihuazhan/pretrained/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

        # for conch
        self.emdproj=FCLayer(dropout=0.25,act='relu',in_size=512)



    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        device=x.device
        diffusion = create_diffusion(str(self.t_steps))
        concatenated_vector = torch.randn(1,4,32,32)
        concatenated_vector=concatenated_vector.to(device)
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples

   
    def forward(self, x):
        if self.training:

            a=self.Diffusion_reembed_withInfoMean(x)
            x = x.squeeze()
            # print(x.shape)

            a=self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result=self.head(a_embed)
            for param in self.embedding2.parameters():
                param.requires_grad = True
            for param in self.head.parameters():
                param.requires_grad = True
            #tsnex= x.squeeze()
            # """
            # 此处为tsne画图
            # """
            # results = []
            # for _ in range(200):
            #     aa = self.Diffusion_reembed(x)
            #     results.append(aa)
            # b = torch.stack(results, dim=0)
            # nothing=self.drawTsne200(tsnex,b)
            # """
            # 此处为tsne画图
            # """
            b = x
            # cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            
            # mask = torch.zeros_like(cosine_similarity)
            # probabilities = F.softmax(-cosine_similarity, dim=0)
            # k = int(b.size(0) * self.k_ratio)

            # selected_indices = torch.multinomial(probabilities, num_samples=k, replacement=False)
            # mask = torch.zeros_like(b, dtype=torch.bool)
            # mask[selected_indices] = True
            # result = b * mask
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
                        
            # k = self.k_ratio
            # sorted_cosine_similarity, _ = torch.sort(cosine_similarity)
            # threshold_index = int(k * len(cosine_similarity))
            # threshold = sorted_cosine_similarity[threshold_index]
            # mask_high = cosine_similarity > threshold
            # mask_low = cosine_similarity < threshold
            # result_igh = x[mask_high.unsqueeze(1).expand_as(x)].contiguous()
            # result = x[mask_low.unsqueeze(1).expand_as(x)].contiguous()




            # """
            # 此处为tsne画图
            # """
 
            # nothing=self.drawTsne(tsnex,result,result_high,a)
            # """
            # 此处为tsne画图
            # """

            #boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
            x = result.view(-1, 512)
            x2=result_high.view(-1, 512)
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x1 = self.head(x.squeeze(1))
            x2 = self.head(x2.squeeze(1))
            max_index_x1 = torch.argmax(x1, dim=1)
            max_index_x2 = torch.argmax(x2, dim=1)
            same_max_dimension = max_index_x1 == max_index_x2
            result = torch.where(same_max_dimension, x1 - x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))

            # if same_max_dimension:
            #     # 获取最大维度的索引
            #     max_dim_index = max_index_x1
            #     # print(x1)
            #     # print(x2)
                
            #     # 在最大维度上比较x1和x2的对应值
            #     x1_max_dim_values = x1[:, max_dim_index]
            #     x2_max_dim_values = x2[:, max_dim_index]
            #     #comparison = x1_max_dim_values > x2_max_dim_values
            #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
            #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
            #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
            #                   x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
            #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
            #     # 使用较大的值减去较小的值
            #     # print(result)
            #     # print("_________________________________________________________")
            # else:
            #     # 如果最大维度不同，保持原来的操作
            #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)







            # print("x1 is ",end='')
            # print(x1)
            # print("x2 is ",end='')
            # print(x2)
            # print("是否相同 is ",end='')
            # print(same_max_dimension)
            # print("最终结果")
            # print(result)
            # print("_______________________________________________")
            x=result


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))




                x=result
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result



    def forward_tsne(self, x,b_label):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        x=x.to(device)
        self.to(device)
        a=self.Diffusion_reembed_withInfoMean(x)
        x = x.squeeze()
        # print(x.shape)
        a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
        b = x

        cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
        sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
        sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
        k = self.k_ratio
        num_elements = int(torch.tensor(b.size(0)) * k)
        k_indices = sorted_indices[:num_elements]
        k_indices_high = sorted_indices_high[:num_elements]


        result = b[k_indices]
        result_high = b[k_indices_high]
        x = result.view(-1, 1024)
        #x2=result_high.view(-1, 1024)
        result_label = [b_label[idx] for idx in k_indices]
        #result_high_label = [b_label[idx] for idx in k_indices_high]

       
        b_new = np.full_like(b_label, 2)
        b_new[k_indices.cpu().numpy()] = b_label[k_indices.cpu().numpy()]

        #

        return x,result_label,b_new



    def forward_test(self, x):
        if self.training:
            return 'error'
        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))
                #result = x1
                # if same_max_dimension:
                # # 获取最大维度的索引
                #     max_dim_index = max_index_x1
                #     # print(x1)
                #     # print(x2)
                    
                #     # 在最大维度上比较x1和x2的对应值
                #     x1_max_dim_values = x1[:, max_dim_index]
                #     x2_max_dim_values = x2[:, max_dim_index]
                #     #comparison = x1_max_dim_values > x2_max_dim_values
                #     max_values = torch.where(x1 > x2, x1, x2)[:, max_dim_index]
                #     use_x1 = x1[:, max_dim_index] > x2[:, max_dim_index]
                #     diff_values = torch.where(use_x1, x1[:, 1 - max_dim_index] - x2[:, 1 - max_dim_index],
                #                 x2[:, 1 - max_dim_index] - x1[:, 1 - max_dim_index])
                #     result = torch.stack([max_values if dim == max_dim_index else diff_values for dim in range(2)], dim=1)
                # # 使用较大的值减去较小的值
                # # print(result)
                # # print("_________________________________________________________")
                # else:
                #     # 如果最大维度不同，保持原来的操作
                #     result = torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1)



                x=result
                a_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result, A, A2


class DAttentionWithDiffEndfor3(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndfor3, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )

        self.attention3 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/home/huangsheng/shihuazhan/pretrained/DiT-XL-2-256x256.pt"
        ckpt_path = "/data3/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

        # for conch
        self.emdproj=FCLayer(dropout=0.25,act='relu',in_size=512)






    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples



    
    def forward(self, x):
        if self.training:

            a=self.Diffusion_reembed_withInfoMean(x)
            x = x.squeeze()
            # print(x.shape)

            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result=self.head(a_embed)

            b = x

            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
                        
            x = result.view(-1, 1024)
            x2=result_high.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            x = self.embedding2(x) # 1024->512
            x2 = x
            x3 = x
 
            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            A3 = self.attention3(x3)
            A3 = torch.transpose(A3, -1, -2)  # KxN
            A3 = F.softmax(A3, dim=-1)  # softmax over N

            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)
            x3 = torch.matmul(A3,x3)

            x1 = self.head(x.squeeze(1))
            x2 = self.head(x2.squeeze(1))
            x3 = self.head(x3.squeeze(1))
            result = torch.stack([
    x1[:, 0] - x2[:, 0] - x3[:, 0],  # 第一维
    x2[:, 1] - x1[:, 1] - x3[:, 1],  # 第二维
    x3[:, 2] - x1[:, 2] - x2[:, 2]   # 第三维
], dim=1)
            x=result


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding2(x) # 1024->512
                x2 = x1
                x3 = x1

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                A3 = self.attention3(x3)
                A3 = torch.transpose(A3, -1, -2)  # KxN
                A3 = F.softmax(A3, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)
                x3 = torch.matmul(A3,x3)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))
                x3 = self.head(x3.squeeze(1))
                result=torch.stack([
    x1[:, 0] - x2[:, 0] - x3[:, 0],  # 第一维
    x2[:, 1] - x1[:, 1] - x3[:, 1],  # 第二维
    x3[:, 2] - x1[:, 2] - x2[:, 2]   # 第三维
], dim=1)
                x=result
                a_result=result
            

        return x,a_result


class DAttentionWithDiffEndfor3_noadv(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndfor3_noadv, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )

        self.attention3 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        self.head2048 = nn.Linear(2048,out_dim)
        self.hom=HoMPool()
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/home/huangsheng/shihuazhan/pretrained/DiT-XL-2-256x256.pt"
        ckpt_path = "/data3/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

        # for conch
        self.emdproj=FCLayer(dropout=0.25,act='relu',in_size=512)






    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples



    
    def forward(self, x):
        if self.training:

            a=self.Diffusion_reembed_withInfoMean(x)
            x = x.squeeze()
            # print(x.shape)

            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a) #1,512      1,n,
            # for param in self.head.parameters():
            #     param.requires_grad = False
            

            b = x

            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
                        
            x = result.view(-1, 1024)
            
            # b,p,n = x.size()
            x = self.embedding2(x) # 1024->512
            x_ori= x
 
            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N



            x = torch.matmul(A,x)
            x_hom=torch.cat([x,x_ori],dim=0)
            a_hom=torch.cat([a_embed,x_ori],dim=0)
            x_hom=x_hom.unsqueeze(0)
            a_hom=a_hom.unsqueeze(0)
            x_result=self.hom(x_hom)
            a_result=self.hom(a_hom)
            x1 = self.head2048(x_result)
            a_result=self.head2048(a_result)
 
            x=x1


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding2(x) # 1024->512
                x_ori= x1

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N
                x = torch.matmul(A,x1)
                x_hom=torch.cat([x.squeeze(0),x_ori.squeeze(0)],dim=0)
                x_hom=x_hom.unsqueeze(0)
                x_result=self.hom(x_hom)
                x1 = self.head2048(x_result)
 
                x=x1
                a_result=x1
            

        return x,a_result








class DAttentionWithDiffEndfor3New_(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndfor3New_, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        self.head2 = nn.Linear(512,2)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/data3/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/home/huangsheng/shihuazhan/pretrained/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

       



    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
 
 
       


    
    def forward(self, x):
        if self.training:

            a=self.Diffusion_reembed_withInfoMean(x)
            x = x.squeeze()
            # print(x.shape)

            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result=self.head2(a_embed)
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
            x = result.view(-1, 1024)
            x2=result_high.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x11 = self.head(x.squeeze(1))
            x22 = self.head(x2.squeeze(1))
            x_result=self.head2(x.squeeze(1))
            result = x11-x22
            x=result


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))




                x=result
                a_result=result
                x_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result,x_result


 

class DAttentionWithDiffEndTransfor3New(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndTransfor3New, self).__init__()

        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.Transmodel2 = TransMIL(n_classes=out_dim,dropout=False,act='relu')
        self.Transmodel1 = TransMIL(n_classes=out_dim,dropout=False,act='relu')


        self.head = nn.Linear(512,out_dim)
        self.head2 = nn.Linear(512,2)
        
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/data3/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/home/huangsheng/shihuazhan/pretrained/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

       



    @torch.no_grad()
    def Diffusion_reembed_withInfoMean(self,x): #实现思路：mean噪声当作输入diffusion的特征，然后生成特征当锚点
        output_channels = 4
        # 实例维度转换
        x = x.squeeze() #(n,1024)
        n=x.size(0)
        mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).view(1, 32, 32)
        vector= mean_vector
        vector=vector.unsqueeze(1)
        diffusion = create_diffusion(str(self.t_steps))
        x = x.view(n, 32, 32)
        random_index = torch.randint(0, x.size(0), (1,)) #另一种取法
        x_pooled = x[random_index].unsqueeze(0)    #另一种取法
        
        noise1 = x_pooled.repeat(1, 3, 1, 1)
        # samples = diffusion.p_sample_loop(
        # self.Dit.forward_unconditional_for_wsi2,x_pooled.shape,x_pooled,clip_denoised=False, progress=True,device=x_pooled.device) #随机特征
        img = torch.randn(noise1.shape)
        concatenated_vector = torch.cat([img.cuda(), vector], dim=1)
        
        
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,shape=concatenated_vector.shape,noise=concatenated_vector,clip_denoised=False, progress=True,device=x_pooled.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
 
 
       


    
    def forward(self, x):
        if self.training:

            a=self.Diffusion_reembed_withInfoMean(x)
            x = x.squeeze()
            # print(x.shape)

            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
             
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_embed = self.Transmodel2._fc1(a)
            a_result=self.head2(a_embed)
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
            x = result.view(-1, 1024)
            x2=result_high.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            result=self.head(self.Transmodel1.forward_feature(x))-self.head(self.Transmodel2.forward_feature(x))
            x_result=self.head2(self.Transmodel1.forward_feature(x))
            x=result


        else:
            if self.ifTrain == 1:
                result = self.head(self.Transmodel1.forward_feature(x))-self.head(self.Transmodel2.forward_feature(x))
                x=result
                a_result=result
                x_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result,x_result

   


class DAttentionWithDiffEndfor3New_transform_g(nn.Module):
    def __init__(self,model_t,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0,adapter_ratio=1.0,a_ratio=1.0,a_num=1):
        super(DAttentionWithDiffEndfor3New_transform_g, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.embedding512_1024 = FCLayer512_1024()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        

        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(512,out_dim)
        if out_dim != 2:
            self.head2 = nn.Linear(512,2)
        else:
            self.head2 = self.head
        self.model_t=model_t

       

    def transform_ge(self,x_neg):
        # x = x_neg.squeeze() #(n,1024)
        # n=x.size(0)
        # mean_vector = (torch.mean(x, dim=0, keepdim=True)[0]).unsqueeze(0)
        # # print(mean_vector.shape)
        # aug_params = self.model_t.sample_aug_params(
        # batch_size=1,
        # device=mean_vector.device,
        # mode="wsi_wise"
        # )
        # augmented_embeddings = self.model_t(mean_vector, aug_params)
        # aug=self.embedding512_1024(augmented_embeddings)



        #实现简单的随机抽取
        x = x_neg.squeeze() #(n,1024)
        aug = x[torch.randperm(x.size(0))[:1]]  

        return aug

       


    
    def forward(self, x):
        if self.training:
             
            a=self.transform_ge(x)
            x = x.squeeze()
            #print(a.shape)

            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            # for param in self.embedding2.parameters():
            #     param.requires_grad = False
            a_embed=self.embedding2(a)
            # for param in self.head.parameters():
            #     param.requires_grad = False
            a_result=self.head2(a_embed)
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = b[k_indices]
            result_high = b[k_indices_high]
            x = result.view(-1, 1024)
            x2=result_high.view(-1, 1024)
            x2= x
            # b,p,n = x.size()
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x11 = self.head(x.squeeze(1))
            x22 = self.head(x2.squeeze(1))
            x_result=self.head2(x.squeeze(1))
            result = x11-x22
            x=result


        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))




                x=result
                a_result=result
                x_result=result


            else:
                a = self.Diffusion_reembed(x)
            

        return x,a_result,x_result


 

 # shz rebuttal 0503
import torch
import torch.nn as nn
import torch.nn.functional as F

class DAttentionWithDiffEndfor3New_save(nn.Module):
    def __init__(self, out_dim=2, k_ratio=0.1, t_steps=2, ifTrain=1, ifrand=0, adapter_ratio=1.0, a_ratio=1.0, a_num=1):
        super(DAttentionWithDiffEndfor3New_save, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        self.adapter_ratio = adapter_ratio
        self.a_ratio = a_ratio
        self.a_num=a_num
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )
        
        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )

        self.head = nn.Linear(512,out_dim)
        self.head2 = nn.Linear(512,2)
        
        # =================================================================================
        # ### 修改点 1：彻底删除 self.Dit 以及所有 Diffusion 权重的加载代码 ###
        # 这将使你的模型初始化速度变为毫秒级，保存的 .pt 权重文件缩小 99%
        # =================================================================================


    # =================================================================================
    # ### 修改点 2：彻底删除 Diffusion_reembed_withInfoMean 函数 ###
    # 原型的生成已经由离线脚本做完了，不需要在线计算了
    # =================================================================================


    # ### 修改点 3：修改 forward，接收从 dataloader 传来的 bag = [features, a]
    def forward(self, inputs):
        # inputs 就是 main.txt 里传进来的 bag
        # 由于我们在 dataloader 里改了返回值，这里 inputs 会是一个包含两项的列表：[x, a]
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            x, a = inputs
        else:
            x = inputs
            a = None # 防御性编程，防止某些纯测试情况未传 a

        # 统一实例维度转换
        x = x.squeeze()
        if a is not None:
            a = a.squeeze()

        if self.training:
            # ### 修改点 4：直接使用离线传进来的 a，不再实时计算 ###
            # 此时的 a 已经是包含了 Diffusion 特征的 1024 维向量
            
            a = self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            a_embed = self.embedding2(a)
            a_result = self.head2(a_embed)
            
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            
            result = b[k_indices]
            result_high = b[k_indices_high]
            x = result.view(-1, 1024)
            x2 = result_high.view(-1, 1024)
            x2 = x
            
            x = self.embedding1(x) # 1024->512
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2)  # KxN
            A2 = F.softmax(A2, dim=-1)  # softmax over N

            x = torch.matmul(A,x)
            x2 = torch.matmul(A2,x2)

            x11 = self.head(x.squeeze(1))
            x22 = self.head(x2.squeeze(1))
            x_result=self.head2(x.squeeze(1))
            result = x11-x22
            x = result

        else:
            if self.ifTrain == 1:
                x1  = self.embedding1(x) # 1024->512
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2)  # KxN
                A2 = F.softmax(A2, dim=-1)  # softmax over N

                x1 = torch.matmul(A,x1)
                x2 = torch.matmul(A2,x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                max_index_x1 = torch.argmax(x1, dim=1)
                max_index_x2 = torch.argmax(x2, dim=1)
                same_max_dimension = max_index_x1 == max_index_x2
                result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))

                x = result
                a_result = result
                x_result = result

            else:
                # ### 修改点 5：如果你之前这里也有生成操作，现在直接复用传入的 a ###
                # a = self.Diffusion_reembed(x) -> 删掉
                pass # 你的原代码这里只有 a = self.Diffusion_reembed(x) 却没有后续操作，我就先 pass 了

        return x, a_result, x_result




import torch
import torch.nn as nn
import torch.nn.functional as F

class DAttentionWithDiffEndfor3New(nn.Module):
    def __init__(self, out_dim=2, k_ratio=0.1, t_steps=2, ifTrain=1, ifrand=0, 
                 adapter_ratio=1.0, a_ratio=1.0, a_num=1,
                 ablate_margin=False, adaptive_th=False, adaptive_alpha=0.5): # 新增消融控制参数
        super(DAttentionWithDiffEndfor3New, self).__init__()
        self.embedding1 = FCLayer()
        self.embedding2 = FCLayer()
        self.L = 512
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.adapter = Adapter(1024, 4)
        
        # 通过控制这两个 ratio 可以实现 Adapter 和 Residual 的消融 (R4.6)
        self.adapter_ratio = adapter_ratio 
        self.a_ratio = a_ratio
        self.a_num = a_num
        
        # 新增控制标志
        self.ablate_margin = ablate_margin # 如果为 True，则取消双视角对抗相减，改为独立网络集成 (R4.6)
        self.adaptive_th = adaptive_th     # 如果为 True，则启用基于方差的自适应阈值，废弃固定 k_ratio (R4.4)
        self.adaptive_alpha = adaptive_alpha # 自适应阈值的松弛系数

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D, bias=False),
            nn.Tanh(),
            nn.Linear(self.D, self.K, bias=False)
        )
        
        self.attention2 = nn.Sequential(
            nn.Linear(self.L, self.D, bias=False),
            nn.Tanh(),
            nn.Linear(self.D, self.K, bias=False)
        )

        self.head = nn.Linear(512, out_dim)
        self.head2 = nn.Linear(512, 2)


    def forward(self, inputs):
        # 接收从 dataloader 传来的 bag = [features, a]
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            x, a = inputs
        else:
            x = inputs
            a = x # 防御性编程

        x = x.squeeze()
        if a is not None:
            a = a.squeeze()

        if self.training:
            # 1. 负样本原型精调 (可在此消融 Adapter 和 Residual)
            a = self.a_ratio * a + self.adapter_ratio * self.adapter(a)  
            
            a_embed = self.embedding2(a)
            a_result = self.head2(a_embed)
            
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            
            # 2. 异构实例挖掘策略 (固定阈值 vs 自适应阈值)
            if self.adaptive_th:
                # 【应对 R4.4】：自适应阈值策略
                mean_sim = cosine_similarity.mean()
                std_sim = cosine_similarity.std()
                
                # 挑选极度不像原型的（困难负样本/正样本）
                hard_mask = cosine_similarity < (mean_sim - self.adaptive_alpha * std_sim)
                # 挑选极度像原型的（典型正样本/极端特征）
                easy_mask = cosine_similarity > (mean_sim + self.adaptive_alpha * std_sim)
                
                k_indices = torch.where(hard_mask)[0]
                k_indices_high = torch.where(easy_mask)[0]
                
                # 防御性回退：如果自适应过滤得太狠导致没有样本，回退到固定比例
                if len(k_indices) < 2 or len(k_indices_high) < 2:
                    sorted_indices = torch.argsort(cosine_similarity, dim=0)
                    sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True)
                    num_elements = max(2, int(b.size(0) * self.k_ratio))
                    k_indices = sorted_indices[:num_elements]
                    k_indices_high = sorted_indices_high[:num_elements]
            else:
                # 你的原始逻辑：固定阈值策略
                sorted_indices = torch.argsort(cosine_similarity, dim=0) 
                sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) 
                num_elements = int(torch.tensor(b.size(0)) * self.k_ratio)
                num_elements = max(1, num_elements) # 防止取不到报错
                k_indices = sorted_indices[:num_elements]
                k_indices_high = sorted_indices_high[:num_elements]
            
            result = b[k_indices]
            result_high = b[k_indices_high]
            x = result.view(-1, 1024)
            x2 = result_high.view(-1, 1024)
            x2 = x
            
            x = self.embedding1(x) 
            x2 = self.embedding2(x2)

            A = self.attention(x)
            A = torch.transpose(A, -1, -2) 
            A = F.softmax(A, dim=-1)

            A2 = self.attention2(x2)
            A2 = torch.transpose(A2, -1, -2) 
            A2 = F.softmax(A2, dim=-1) 

            x = torch.matmul(A, x)
            x2 = torch.matmul(A2, x2)

            x11 = self.head(x.squeeze(1))
            x22 = self.head(x2.squeeze(1))
            x_result = self.head2(x.squeeze(1))
            
            # 3. 视角对抗消融 (Margin Maximization vs Independent View)
            if self.ablate_margin:
                # 【应对 R4.6】：移除对立减法，退化为两个独立的网络预测再平均
                result = (x11 + x22) / 2.0
            else:
                # 原始的双视角对立逻辑
                result = x11 - x22
                
            x = result

        else:
            if self.ifTrain == 1:
                x1  = self.embedding1(x) 
                x2 = self.embedding2(x)

                A = self.attention(x1)
                A = torch.transpose(A, -1, -2)
                A = F.softmax(A, dim=-1)

                A2 = self.attention2(x2)
                A2 = torch.transpose(A2, -1, -2) 
                A2 = F.softmax(A2, dim=-1)

                x1 = torch.matmul(A, x1)
                x2 = torch.matmul(A2, x2)

                x1 = self.head(x1.squeeze(1))
                x2 = self.head(x2.squeeze(1))

                if self.ablate_margin:
                    # 推理时同样保持消融逻辑一致
                    result = (x1 + x2) / 2.0
                else:
                    max_index_x1 = torch.argmax(x1, dim=1)
                    max_index_x2 = torch.argmax(x2, dim=1)
                    same_max_dimension = max_index_x1 == max_index_x2
                    result = torch.where(same_max_dimension, x1-x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x1 - x2))

                x = result
                a_result = result
                x_result = result
            else:
                pass 

        return x, a_result, x_result