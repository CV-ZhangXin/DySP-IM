import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from .nystrom_attention import NystromAttention
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            # ref from huggingface
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes,dropout,act):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        #self.pos_layer = nn.Identity()
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.25))
        self._fc1 = [nn.Linear(1024, 512)]

        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]

        if dropout:
            self._fc1 += [nn.Dropout(0.25)]

        #self._fc1 += [SwinEncoder(attn='swin',pool='none',n_heads=2,trans_conv=False)]
        
        self._fc1 = nn.Sequential(*self._fc1)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

        self.apply(initialize_weights)

    def forward(self, x):

        h = x.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return logits
    
    def forward_feature(self, x):

        h = x.float() #[B, n, 1024]
        
        h = self._fc1(h) #[B, n, 512]
        if len(h.size()) == 2:
            h = h.unsqueeze(0)
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]
        
        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        # logits = self._fc2(h) #[B, n_classes]
        # Y_hat = torch.argmax(logits, dim=1)
        # Y_prob = F.softmax(logits, dim = 1)
        # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return h

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
    
class TransMILwithDiff(nn.Module):
    def __init__(self, n_classes,dropout,act):
        super(TransMILwithDiff, self).__init__()
        self.pos_layer = PPEG(dim=512)
        #self.pos_layer = nn.Identity()
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.25))
        self._fc1 = [nn.Linear(1024, 512)]

        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]

        if dropout:
            self._fc1 += [nn.Dropout(0.25)]

        #self._fc1 += [SwinEncoder(attn='swin',pool='none',n_heads=2,trans_conv=False)]
        
        self._fc1 = nn.Sequential(*self._fc1)
        self.adapter = Adapter(1024, 4)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.cls_token2, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.layer11 = TransLayer(dim=512)
        self.layer22 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        #self.apply(initialize_weights)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()
        self.t_steps=2
        self.k_ratio=0.6 #0.4-call
        self.a_ratio=1.0
        self.adapter_ratio=0.1

        

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
            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a) 
            a_embed=self._fc1(a)
            
            logits3 = self._fc2(a_embed)
            x = x.squeeze()
            cosine_similarity = F.cosine_similarity(a.expand_as(x), x, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(x.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = x[k_indices]
            h = result.float() #[B, n, 1024]
            h = self._fc1(h) #[B, n, 512]
            if len(h.size()) == 2:
                h = h.unsqueeze(0)
            #---->pad
            H = h.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512] 
            #---->cls_token
            B = h.shape[0]


            cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
            h1 = torch.cat((cls_tokens, h), dim=1)
            
            cls_tokens2 = self.cls_token2.expand(B, -1, -1).to(h.device)
            h2 = torch.cat((cls_tokens2, h), dim=1)

            #---->Translayer x1
            h1 = self.layer1(h1) #[B, N, 512]
            h2 = self.layer11(h2) #[B, N, 512]

            #---->PPEG
            h1 = self.pos_layer(h1, _H, _W) #[B, N, 512]
            h2 = self.pos_layer(h2, _H, _W) #[B, N, 512]
            #---->Translayer x2
            h1 = self.layer2(h1) #[B, N, 512]
            h2 = self.layer22(h2) #[B, N, 512]

            #---->cls_token
            h1 = self.norm(h1)[:,0]
            h2 = self.norm2(h2)[:,0]

            #---->predict
            logits = self._fc2(h1) #[B, n_classes]
            logits2 = self._fc2(h2) #[B, n_classes]
            # Y_hat = torch.argmax(logits, dim=1)
            # Y_prob = F.softmax(logits, dim = 1)
            # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
            final=logits - logits2






        else:
            h = x.float() #[B, n, 1024]
            h = self._fc1(h) #[B, n, 512]
            if len(h.size()) == 2:
                h = h.unsqueeze(0)
            #---->pad
            H = h.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
            #---->cls_token
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
            h = torch.cat((cls_tokens, h), dim=1)
            #---->Translayer x1
            h = self.layer1(h) #[B, N, 512]
            #---->PPEG
            h = self.pos_layer(h, _H, _W) #[B, N, 512]
            #---->Translayer x2
            h = self.layer2(h) #[B, N, 512]
            #---->cls_token
            h = self.norm(h)[:,0]
            #---->predict
            logits = self._fc2(h) #[B, n_classes]
            # Y_hat = torch.argmax(logits, dim=1)
            # Y_prob = F.softmax(logits, dim = 1)
            # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
            final= logits
            logits3= logits
        return final,logits3
    



class TransMILwithDiff2(nn.Module):
    def __init__(self, n_classes,dropout,act):
        super(TransMILwithDiff2, self).__init__()
        self.pos_layer = PPEG(dim=512)
        #self.pos_layer = nn.Identity()
        # self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),nn.Dropout(0.25))
        self._fc1 = [nn.Linear(1024, 512)]

        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]

        if dropout:
            self._fc1 += [nn.Dropout(0.25)]

        #self._fc1 += [SwinEncoder(attn='swin',pool='none',n_heads=2,trans_conv=False)]
        
        self._fc1 = nn.Sequential(*self._fc1)
        self.adapter = Adapter(1024, 4)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, 512))
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.cls_token2, std=1e-6)
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.layer11 = TransLayer(dim=512)
        self.layer22 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)
        self.Dit= DiT_models["DiT-XL/2"](input_size=32)
        #self.apply(initialize_weights)
        ckpt_path = "/home/shihuazhan/DiT/DiT/pretrained_models/DiT-XL-2-256x256.pt"
        # ckpt_path = "/data/zhangxiaoxian/output/pretrained_models/DiT-XL-2-256x256.pt"
        state_dict = find_model(ckpt_path)
        self.Dit.load_state_dict(state_dict)
        self.Dit.eval()
        self.t_steps=2
        self.k_ratio=0.6 #0.4-call
        self.a_ratio=1.0
        self.adapter_ratio=1.0

        

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
            a= self.a_ratio * a + self.adapter_ratio * self.adapter(a) 
            for param in self._fc1.parameters():
                param.requires_grad = True
            for param in self._fc2.parameters():
                param.requires_grad = False
            a_embed=self._fc1(a)
            logits3 = self._fc2(a_embed)
            for param in self._fc1.parameters():
                param.requires_grad = True
            for param in self._fc2.parameters():
                param.requires_grad = True












            x = x.squeeze()
            x = self._fc1(x)

            cosine_similarity = F.cosine_similarity(a_embed.expand_as(x), x, dim=1)
            sorted_indices = torch.argsort(cosine_similarity, dim=0) #low
            sorted_indices_high = torch.argsort(cosine_similarity, dim=0, descending=True) #high
            k = self.k_ratio
            num_elements = int(torch.tensor(x.size(0)) * k)
            k_indices = sorted_indices[:num_elements]
            k_indices_high = sorted_indices_high[:num_elements]
            result = x[k_indices]
            h = result.float() #[B, n, 1024]
           
            if len(h.size()) == 2:
                h = h.unsqueeze(0)
            #---->pad
            H = h.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512] 
            #---->cls_token
            B = h.shape[0]


            cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
            h1 = torch.cat((cls_tokens, h), dim=1)
            
            cls_tokens2 = self.cls_token2.expand(B, -1, -1).to(h.device)
            h2 = torch.cat((cls_tokens2, h), dim=1)

            #---->Translayer x1
            h1 = self.layer1(h1) #[B, N, 512]
            h2 = self.layer11(h2) #[B, N, 512]

            #---->PPEG
            h1 = self.pos_layer(h1, _H, _W) #[B, N, 512]
            h2 = self.pos_layer(h2, _H, _W) #[B, N, 512]
            #---->Translayer x2
            h1 = self.layer2(h1) #[B, N, 512]
            h2 = self.layer22(h2) #[B, N, 512]

            #---->cls_token
            h1 = self.norm(h1)[:,0]
            h2 = self.norm2(h2)[:,0]

            #---->predict
            logits = self._fc2(h1) #[B, n_classes]
            logits2 = self._fc2(h2) #[B, n_classes]
            # Y_hat = torch.argmax(logits, dim=1)
            # Y_prob = F.softmax(logits, dim = 1)
            # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
            final=logits - logits2






        else:
            h = x.float() #[B, n, 1024]
            h = self._fc1(h) #[B, n, 512]
            if len(h.size()) == 2:
                h = h.unsqueeze(0)
            #---->pad
            H = h.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]
            #---->cls_token
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
            h = torch.cat((cls_tokens, h), dim=1)
            #---->Translayer x1
            h = self.layer1(h) #[B, N, 512]
            #---->PPEG
            h = self.pos_layer(h, _H, _W) #[B, N, 512]
            #---->Translayer x2
            h = self.layer2(h) #[B, N, 512]
            #---->cls_token
            h = self.norm(h)[:,0]
            #---->predict
            logits = self._fc2(h) #[B, n_classes]
            # Y_hat = torch.argmax(logits, dim=1)
            # Y_prob = F.softmax(logits, dim = 1)
            # results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
            final= logits
            logits3= logits
        return final,logits3

if __name__ == "__main__":
    data = torch.randn((1, 6000, 1024))
    model = TransMIL(n_classes=2,dropout=False,act='relu')
    for k, v in model.state_dict().items():
        print(k)
    # print(model.eval())
    # results_dict = model(data = data)
    # print(results_dict)