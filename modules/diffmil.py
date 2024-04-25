import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .swin import SwinEncoder
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


class MeanMIL(nn.Module):
    def __init__(self,n_classes=1,dropout=True,act='relu',init=False,init_pt=None, test=False,input_dim=1024):
        super(MeanMIL, self).__init__()

        head = [nn.Linear(input_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
            
        #head += [SwinEncoder(attn='swin',pool='none')]
        #head += [ASPP(28,512,embed_dim=128)]
        head += [nn.Linear(512,n_classes)]
        
        self.head = nn.Sequential(*head)


        # self.head = nn.Sequential(
        #     nn.Linear(1024,512),
        #     nn.ReLU(),
        #     nn.Dropout(0.25),
        #     nn.Linear(512,n_classes)
        # )

        if test:
            self._test = nn.Linear(1024, 512)
        if init:
            pre_dict = torch.load(init_pt)
            new_state_dict ={}
            target = ['head.0.weight','head.0.bias']
            for k,v in pre_dict.items():
                if k in target:
                    new_state_dict[k.split('.',1)[1]]=v
                    print(k)
            self.head.load_state_dict(new_state_dict,strict=False)
            print('embedding fc Inited')
        else:
            self.apply(initialize_weights)

    def forward(self,x):

        x = self.head(x).mean(axis=1)
        return x



class MaxMIL(nn.Module):
    def __init__(self,n_classes=1,dropout=True,act='relu',init=False,init_pt=None, test=False,input_dim=1024):
        super(MaxMIL, self).__init__()

        #head = [nn.Linear(192,192)]

        head = [nn.Linear(input_dim,512)]

        if act.lower() == 'relu':
            head += [nn.ReLU()]
        elif act.lower() == 'gelu':
            head += [nn.GELU()]

        if dropout:
            head += [nn.Dropout(0.25)]
        #head += [SwinEncoder(attn='swin',pool='none',trans_conv=True)]
        #head += [nn.Linear(192,n_classes)]
        head += [nn.Linear(512,n_classes)]
        self.head = nn.Sequential(*head)

        if test:
            self._test = nn.Linear(1024, 512)
        if init:
            pre_dict = torch.load(init_pt)
            new_state_dict ={}
            target = ['head.0.weight','head.0.bias']
            for k,v in pre_dict.items():
                if k in target:
                    new_state_dict[k.split('.',1)[1]]=v
            self.head.load_state_dict(new_state_dict,strict=False)
            print('embedding fc Inited')
        else:
            self.apply(initialize_weights)

    def forward(self,x):
        x,_ = self.head(x).max(axis=1)
        return x

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

        return x
    

class DAttentionWithDiff(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,n_robust=0,ifTrain=1,ifrand=0):
        super(DAttentionWithDiff, self).__init__()
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
    def Diffusion_reembed(self,x): #实现思路：随机取一个当作输入diffusion的特征，然后生成特征当锚点
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
        plt.scatter(data_tsne[:-1, 0], data_tsne[:-1, 1], c='blue', label='A Data',s=1)
        plt.scatter(data_tsne[-1, 0], data_tsne[-1, 1], c='red', label='B Data', s=1)
        plt.legend()
        plt.title('t-SNE Visualization')
        output_path = '/nas/zhangxiaoxian/output/mil_shz/tsne_random/tsne_visualization_{}.png'.format(random_number)
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
            #random=self.drawTsne(tsnex,a)
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            #sorted_indices = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            
            result = b[k_indices]
            #result = b[highest_k_indices]
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
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,n_robust=0,ifTrain=1,ifrand=0):
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
    


if __name__ == "__main__":
    model = DAttention()
    # x=torch.rand(1,2,1024)
    # print(model(x))
    for k,v in model.state_dict().items():
        print(k)


