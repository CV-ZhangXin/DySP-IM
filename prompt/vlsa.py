"""
VLSA's implementation compatitable with 
- OpenAI CLIP (github.com/openai/CLIP)
- HuggingFace CLIP (github.com/huggingface/transformers)
- mahmoodlab/CONCH (github.com/mahmoodlab/CONCH)
"""
import os.path as osp
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from prompt.prompt_learners import load_prompt_learner
from prompt.prompt_learners import load_prompt_adapter
from prompt.prompt_encoder import get_prompt_encoder
from prompt.utils_vl import load_vl_model_to_cpu
from prompt.utils_vl import Tokenizer
from prompt.deepmil import logit_pooling
# from prompt.zft_util.py import

from prompt.zft_util import initialize_weights
from prompt.zft_util import NystromAttention
from prompt.zft_util import BilinearFusion
from prompt.zft_util import SNN_Block
from prompt.zft_util import MultiheadAttention
import numpy as np
from modules.transmil import *
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

class FCLayer(nn.Module):
    def __init__(self, dropout=0.25,act='relu',in_size=512):
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



def contrastive_loss(embedding_positive, embedding_negative, margin=100000000000000000000000000):
    """
    计算对比损失。

    :param embedding_positive: 正文本嵌入，形状为 [1, 512]
    :param embedding_negative: 负文本嵌入，形状为 [1, 512]
    :param margin: 边距参数，用于定义正负嵌入之间的最小距离
    :return: 对比损失值
    """

    distance = F.pairwise_distance(embedding_positive, embedding_negative, p=2)
    # 损失为 max(0, margin - distance) 的平方
    loss = torch.mean(F.relu(margin - distance)**2)

    return loss.mean()



class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512):
        super(Transformer_G, self).__init__()
        # Encoder
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1).cuda()
        h = torch.cat((cls_tokens, features), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        h = self.norm(h)
        return h[:, 0], h[:, 1:]




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









class VLSA(nn.Module):
    def __init__(
        self,
        # text_encoder_cfg,
        # image_encoder_cfg,
        prompt_learner_cfg,
        pretrained_prompt_learner_cfg=None,
        info_prefix='VLSA-UNI',
        **kwargs,
    ) -> None:
        super().__init__()

        self.kwargs = kwargs
        print(f"[{info_prefix}] Found additional kwargs: {self.kwargs}.")
        assert 'vlsa_api' in kwargs, "Please specify `vlsa_api` in arguments."
        assert 'path_clip_model' in kwargs, "Please specify `path_clip_model` in arguments."

        self.text_tokenizer = Tokenizer(
            # root=kwargs['path_clip_model'], 
            # name=text_encoder_cfg['name'],
            api='CONCH'
        )

        vl_model = load_vl_model_to_cpu(
            # text_encoder_cfg,
            # image_encoder_cfg,
            # root=kwargs['path_clip_model'],
            api='CONCH'
        )
        
        # Language-end
        self.pmt_learner_name = prompt_learner_cfg['name']
        self.prompt_encoder = get_prompt_encoder(vl_model, api=kwargs['vlsa_api'])
        if self.pmt_learner_name == 'CoOp':
            
            self.prompt_learner, pretrained_text_features = self._build_prompt_learner(
                prompt_learner_cfg, pretrained_prompt_learner_cfg
            )
            
            if pretrained_text_features is not None:
                pretrained_text_features = pretrained_text_features.detach().clone()
                self.register_buffer("pretrained_text_features", pretrained_text_features, persistent=False)
                print("[VLSA] warning: skip CoOp-based prompt learner and use pretrained text features.")
        
        elif self.pmt_learner_name == 'Adapter':
            self.prompt_adapter = self._build_prompt_adapter(prompt_learner_cfg, pretrained_prompt_learner_cfg)
        
        else:
            raise ValueError(f"{self.pmt_learner_name} is not a valid name of prompt learner.")

        # Vision-end
        if hasattr(vl_model, 'vision_model'):
            assert kwargs['vlsa_api'] == 'HF'
            self.mil_encoder = vl_model.vision_model
        elif hasattr(vl_model, 'visual'):
            assert kwargs['vlsa_api'] in ['CLIP', 'CONCH']
            self.mil_encoder = vl_model.visual
        else:
            raise ValueError(f"[{info_prefix}] `vision_model` or `visual` is not found in {vl_model}.")

        
        
        # self.text_encoder_cfg = text_encoder_cfg
        # self.image_encoder_cfg = image_encoder_cfg
        self.prompt_learner_cfg = prompt_learner_cfg

        self.logit_scale = vl_model.logit_scale
        self.embeding_layer1=FCLayer()
        self.head = nn.Linear(512,2)

        self.G_in_P_Att = MultiheadAttention(embed_dim=512, num_heads=1)
        self.L = 512
        self.D = 128
        self.K = 1
        self.norm = nn.LayerNorm(512)
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D,bias=False),
            # nn.GELU(),
            nn.Tanh(),
            nn.Linear(self.D, self.K,bias=False)
        )

        self.dropout1 = nn.Dropout(p=0.2)
        self.maskPlan=self.kwargs.get('maskPlan')
        self.loss_total=self.kwargs.get('loss_total')
        self.maskTh=self.kwargs.get('maskTh')
        self.headClass = self.kwargs.get('headClass')
        self.logit_buffer = {'global': [], 'refined': []}
        self.vis_count = 0  # 计数器，防止覆盖图片

    def _build_prompt_learner(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
        _prompt_learner_cfg = prompt_learner_cfg.copy()
        _prompt_learner_cfg.update(dict(
            tokenizer = self.text_tokenizer, 
            text_config = self.prompt_encoder.text_config,
            token_embedding = self.prompt_encoder.token_embedding
        ))
        
        prompt_learner = load_prompt_learner(_prompt_learner_cfg['method'], _prompt_learner_cfg)
        
        # if use pretrained text prompts
        pretrained_text_features = None
        if _prompt_learner_cfg['pretrained']:
            assert pretrained_prompt_learner_cfg is not None, "Please specify `config` for `pretrained_prompt_learner`."
            prompt_learner.load_pretrained_parameters(pretrained_prompt_learner_cfg['ckpt'])
            
            # if there is no trainable parameter, pre-compute the fixed text features
            if _prompt_learner_cfg['frozen_context_embeds'] and _prompt_learner_cfg['frozen_rank_embeds']:
                with torch.no_grad():
                    pretrained_text_features = self.compute_text_features_with_coop(prompt_learner)

        return prompt_learner, pretrained_text_features

    def _build_prompt_adapter(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
        _prompt_learner_cfg = prompt_learner_cfg.copy()
        _pretrained_prompt_learner_cfg = pretrained_prompt_learner_cfg.copy()

        # if use CoOp-pretrained text prompts for Adapter
        pretrained_text_features = None
        if _prompt_learner_cfg['pretrained']:
            _pretrained_prompt_learner_cfg['pretrained'] = True
            _, pretrained_text_features = self._build_prompt_learner(
                _pretrained_prompt_learner_cfg, {'ckpt': _pretrained_prompt_learner_cfg['ckpt']}
            )
            assert pretrained_text_features is not None, "Found empty `pretrained_text_features`."
            pretrained_text_features = pretrained_text_features.detach().clone()

        _prompt_learner_cfg.update(dict(
            tokenizer = self.text_tokenizer,
            num_prompts = _prompt_learner_cfg['num_ranks'],
            pretrained_prompt_features = pretrained_text_features,
        ))
        prompt_adapter = load_prompt_adapter(self.prompt_encoder, _prompt_learner_cfg)

        return prompt_adapter

    def compute_text_features_with_coop(self, prompt_learner):
        sentence_embeds = prompt_learner()
        pseudo_sentence_tokens = prompt_learner.pseudo_sentence_tokens
        text_features = self.prompt_encoder(
            prompts_embedding=sentence_embeds, 
            prompts_pseudo_tokens=pseudo_sentence_tokens
        )
        return text_features

    def forward_text_only(self):
        # use pretrained_text_features if exists

        if self.pmt_learner_name == 'CoOp':
            text_features = self.compute_text_features_with_coop(self.prompt_learner)

        elif self.pmt_learner_name == 'Adapter':
            text_features = self.prompt_adapter()

        else:
            text_features = None
            pass

        return text_features

    def encode_instances(self, X):
        return self.mil_encoder(X)

    def get_logit_scale(self):
        return self.logit_scale.exp()

    def mask_draw(self, x_embed_o, Att, index, threshold):
        """
        修改说明：除了返回 masked 特征，还返回 boolean_mask (1代表被掩盖/丢弃，0代表保留)
        注意：根据你的逻辑，mask=1 是要丢弃的(low attention)，mask=0 是保留的。
        """
        A1 = (Att.squeeze(0)).squeeze(0)[0].unsqueeze(0)
        A2 = (Att.squeeze(0)).squeeze(0)[1].unsqueeze(0)
        
        A1_sorted, _ = torch.sort(A1, descending=True)
        A2_sorted, _ = torch.sort(A2, descending=True)

        # 动态阈值计算
        high_threshold_A1 = A1_sorted[0, int(threshold * A1.size(1)) - 1]
        high_threshold_A2 = A2_sorted[0, int(threshold * A2.size(1)) - 1]
        
        high_A1 = A1 >= high_threshold_A1 
        high_A2 = A2 >= high_threshold_A2 
        low_A1 = ~high_A1 
        low_A2 = ~high_A2 

        # 逻辑组合
        mask1 = (high_A1 & high_A2).float() 
        mask2 = (low_A1 & low_A2).float() # 你的默认方案：mask2=1 代表既不是A1高也不是A2高
        mask3 = (high_A1 & low_A2).float()
        mask4 = (low_A1 & high_A2).float()
        
        # 选择方案
        if index == 1:
            chosen_mask = mask1
        elif index == 2:
            chosen_mask = mask2 # 默认方案
        elif index == 3:
            chosen_mask = mask3
        else:
            chosen_mask = mask4
            
        # 应用掩码 (保留特征)
        # 注意：chosen_mask为1的地方被置零
        a_masked = x_embed_o * (1 - chosen_mask.squeeze()).unsqueeze(1)
        
        # 【新增返回】：返回这个 mask 本身，用于可视化
        # chosen_mask shape: [1, n], 1=dropped, 0=kept
        return a_masked, chosen_mask

    def mask(self,x_embed_o,Att,index,threshold):
        A1=(Att.squeeze(0)).squeeze(0)[0].unsqueeze(0)
        A2=(Att.squeeze(0)).squeeze(0)[1].unsqueeze(0)
        
        A1_sorted, A1_indices = torch.sort(A1, descending=True)
        A2_sorted, A2_indices = torch.sort(A2, descending=True)

        high_threshold_A1 = A1_sorted[0, int(threshold * A1.size(1)) - 1]
        high_threshold_A2 = A2_sorted[0, int(threshold * A2.size(1)) - 1]
        high_A1 = A1 >= high_threshold_A1  # 高注意力区域
        high_A2 = A2 >= high_threshold_A2  # 高注意力区域

        low_A1 = ~high_A1  # 低注意力区域
        low_A2 = ~high_A2  # 低注意力区域


        mask1 = (high_A1 & high_A2).float()  # 转为浮点型掩码
        mask2 = (low_A1 & low_A2).float()
        mask3 = (high_A1 & low_A2).float()
        mask4 = (low_A1 & high_A2).float()
        a_masked1 = x_embed_o * (1 - mask1.squeeze()).unsqueeze(1)  # 反转掩码，并扩展维度
        a_masked2 = x_embed_o * (1 - mask2.squeeze()).unsqueeze(1)
        a_masked3 = x_embed_o * (1 - mask3.squeeze()).unsqueeze(1)
        a_masked4 = x_embed_o * (1 - mask4.squeeze()).unsqueeze(1)

        if index==1:
            a_masked=a_masked1
        elif index==2:
            a_masked=a_masked2
        elif index==3:
            a_masked=a_masked3
        else:
            a_masked=a_masked4
        return a_masked

    def stackk(self,x1,x2):
        tensor1 = x1
        tensor2 = x2
        

        # 使用unsqueeze增加两个维度，使它们的形状变为torch.Size([1, 1, 13987])
        tensor1_expanded = tensor1.unsqueeze(0).unsqueeze(0)
        tensor2_expanded = tensor2.unsqueeze(0).unsqueeze(0)
        

        # 使用stack在新的维度上拼接张量，得到形状torch.Size([2, 1, 13987])
        stacked_tensors = torch.stack((tensor1_expanded, tensor2_expanded))

        # 最后，我们需要交换维度，使形状变为torch.Size([1, 1, 2, 13987])
        final_tensor = stacked_tensors.permute(1, 2, 0, 3)
        return final_tensor

    def calSimLoss(self,text_features,image_features):
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        text_features_conv = text_features.unsqueeze(-1)  # [num_classes, feature_dim, 1] 

        image_features_conv = image_features.unsqueeze(-1)  # [batch_size, feature_dim, 1] 


         # 直接使用卷积计算相似度  
        output = 20 * F.conv1d(image_features_conv, text_features_conv) 

        # 移除复杂的权重计算，直接在最后一维上求平均或求和  
        output = output.mean(-1)  # 或者 output.sum(-1)  

         # 返回 logits  
        logits = output  # 直接使用输出作为logits  

        return logits 


    def knowledge_distillation_loss(self,student_scores, teacher_scores, temperature):
        student_probs = F.softmax(student_scores / temperature, dim=1)
       
        teacher_probs = F.softmax(teacher_scores / temperature, dim=1)
        loss = F.kl_div(student_probs.log(), teacher_probs, reduction='sum')
        return loss

    def sim_loss(self,p, z):
        z = z.detach()
        p = nn.functional.normalize(p, dim=1)
        z = nn.functional.normalize(z, dim=1)
        return -(p * z).sum(dim=1).mean()



    def forward(self,X):
        if isinstance(X,(list,tuple)):
            X = X[0]
        text_features = self.forward_text_only()
        x_embed_o=self.embeding_layer1(X.squeeze(0)) # 512, n
        t_p,Att=self.G_in_P_Att(
        text_features.unsqueeze(0).transpose(1, 0),
        x_embed_o.unsqueeze(0).transpose(1, 0),
        x_embed_o.unsqueeze(0).transpose(1, 0),
                                )   # Att shape [1, 1, 2, 8144]
        text_query = self.norm(t_p.squeeze(1) + text_features)
        
        final_feature = ((t_p.squeeze(1)).mean(0)).unsqueeze(0)

        

 
        S_Posi = torch.matmul(x_embed_o, text_query[0])  # 得到形状为 [n, 1] 的正得分图
        S_Negi = torch.matmul(x_embed_o, text_query[1])  # 得到形状为 [n, 1] 的负得分图
         
        
        delta_S_Posi = S_Posi  # 得到形状为 [n, 1] 的加权正得分
        delta_S_Negi = S_Negi  # 得到形状为 [n, 1] 的加权负得分
        final_score=self.stackk(delta_S_Posi,delta_S_Negi)
        # print(final_score.shape)
        if self.training:
            x_p=self.mask(x_embed_o,final_score,self.maskPlan,self.maskTh) #**************************************************8
            # print("22")
        else:
            #x_p = x_embed_o
            x_p=self.mask(x_embed_o,final_score,self.maskPlan,self.maskTh)
            # print("hello")
        #x_T=self.mask(x_embed_o,Att,4,0.6)
        # 计算每一对行向量之间的余弦相似度
        # cos_sim = F.cosine_similarity(x_p, x_embed_o, dim=1)

        # # 计算平均相似度
        # mean_cos_sim = cos_sim.mean().item()
        A = self.attention(x_p.squeeze(0))
        A = torch.transpose(A, -1, -2)  # KxN
        A = F.softmax(A, dim=-1)  # softmax over N
        # print(mean_cos_sim)
        A2 = self.attention(x_embed_o.squeeze(0))
        A2 = torch.transpose(A2, -1, -2)  # KxN
        A2 = F.softmax(A2, dim=-1)  # softmax over N
        
        x_p = torch.matmul(A,x_p.squeeze(0))
        x_T = torch.matmul(A2,x_embed_o.squeeze(0))
        


        if self.headClass == 1:
            final_feature = final_feature
        elif self.headClass == 2:
            final_feature = x_p
        elif self.headClass == 3:
            final_feature = x_T
        logits_1 = self.head(x_T)
        logits_2 = self.head(final_feature)
        logits_3 = x_p @ text_query.T

        loss = self.sim_loss(logits_3, logits_1) / 2 + self.sim_loss(x_T, x_p) / 2 #这是现在的
        return logits_2,self.loss_total*loss
    

     
