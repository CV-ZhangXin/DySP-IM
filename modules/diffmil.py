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
    def __init__(self, c_in, reduction=4):
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
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,ifTrain=1,ifrand=0):
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
        #self.adapter = Adapter(1024, 4)
        
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
            x = x.squeeze()

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
            result = torch.where(same_max_dimension, x1 + x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1))
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
                result = torch.where(same_max_dimension, x1 + x2, torch.where(x1[:, 1] > x2[:, 1], x1 - x2, x2 - x1))
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
    

if __name__ == "__main__":
    model = DAttention()
    # x=torch.rand(1,2,1024)
    # print(model(x))
    for k,v in model.state_dict().items():
        print(k)


