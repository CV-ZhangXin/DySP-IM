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
    
class DAttentionWithDiffUsingNet(nn.Module):
    def __init__(self,out_dim=2,k_ratio=0.1,t_steps=2,n_robust=0,ifTrain=1,ifrand=0,ifEma=0,ifType=1):
        super(DAttentionWithDiffUsingNet, self).__init__()
        self.L = 1024
        self.D = 128
        self.K = 1
        self.k_ratio= k_ratio
        self.t_steps= t_steps
        self.ifTrain= ifTrain
        self.ifrand = ifrand
        self.ifEma = ifEma
        self.ifType= ifType
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )


        self.head = nn.Linear(1024,out_dim)
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        #ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()

    # def updateEma(self,a):
    #     self.attention_ema -= (1 - a) * (self.attention_ema - self.attention.weight.data)

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

        x = x.squeeze() #(n,1024)
        
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        A = torch.matmul(A,x)
        diffusion = create_diffusion(str(self.t_steps))

        A = A.view(1, 32, 32)
        A = A.unsqueeze(1)
        A = A.repeat(1, 4, 1, 1)
        z = torch.randn(1, 4, 32, 32, device=A.device) #shz 4.26
        print(z)
        A = A+z
        # print(A.shape)
        # print("**")
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,A.shape,A,clip_denoised=False, progress=True,device=A.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples
    

    @torch.no_grad()
    def Diffusion_reembed_ChoseScoreMax(self,x): #实现思路：用abmil的head来对特征打分，选择得分最高的特征当锚点
        if self.ifEma==0:
            A = self.attention(x)
        else:
            print("UsingEma,但是呢,我不知道ema怎么实现,这就很糟糕")

        x = x.squeeze() #(n,1024)
        
        A = torch.transpose(A, -1, -2)  # KxN
        attentionScore = F.softmax(A, dim=-1)  # softmax over N
        ascore=attentionScore.squeeze(0)
        ascore=ascore.squeeze(0) # n
        A = torch.matmul(attentionScore,x) 
        scores = self.head(x) # n,2
        A = self.choseInstanceByValue(x,ascore,scores,self.ifType)
        A = A.view(1, 32, 32)
        A=A.unsqueeze(1)
        A=A.repeat(1, 4, 1, 1)
        z = torch.randn(1, 4, 32, 32, device=A.device) #shz 4.26
        A=A+z #shz 4.26
        diffusion = create_diffusion(str(self.t_steps))
        samples = diffusion.p_sample_loop(
        self.Dit.forward_unconditional_for_wsi2,A.shape,A,clip_denoised=False, progress=True,device=A.device)
        new_shape = (1, 4, 1024)
        samples=samples.view(new_shape)
        samples = samples.mean(dim=1) #(1,1024)
        return samples


    def forward(self, x):
        if self.training:
            if self.ifrand == 0:
                #a = self.Diffusion_reembed_shareWeights(x)
                a =self.Diffusion_reembed_ChoseScoreMax(x)
            else:
                print("Something Wrong")
            x = x.squeeze()
            b = x
            cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            # sorted_indices = torch.argsort(cosine_similarity, dim=0, descending=True)  #high
            k = self.k_ratio
            num_elements = int(torch.tensor(b.size(0)) * k)
            lowest_k_indices = sorted_indices[:num_elements]
            result = b[lowest_k_indices]
            # boundary_similarity = cosine_similarity[sorted_indices[num_elements]]
            x = result.view(-1, 1024)
            # b,p,n = x.size()
            A = self.attention(x)
            A = torch.transpose(A, -1, -2)  # KxN
            A = F.softmax(A, dim=-1)  # softmax over N
            x = torch.matmul(A,x)
            x = self.head(x.squeeze(1))
        else:
            if self.ifTrain == 1:
                # b,p,n = x.size()
                A = self.attention(x)
                A = torch.transpose(A, -1, -2)  # KxN
                A = F.softmax(A, dim=-1)  # softmax over N
                x = torch.matmul(A,x)
                x = self.head(x.squeeze(1))
            else:
                if self.ifrand == 0:
                    a = self.Diffusion_reembed_ChoseScoreMax(x)
                else:
                    print("Wrong")       
                x = x.squeeze()
                b = x
                cosine_similarity = F.cosine_similarity(a.expand_as(b), b, dim=1)
                sorted_indices = torch.argsort(cosine_similarity, dim=0)
                k = self.k_ratio
                num_elements = int(torch.tensor(b.size(0)) * k)
                lowest_k_indices = sorted_indices[:num_elements]
                result = b[lowest_k_indices]
                x = result.view(-1, 1024)
                # b,p,n = x.size()
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


