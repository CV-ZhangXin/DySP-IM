import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 假设这些引用在你环境中是有效的
from prompt.prompt_learners import load_prompt_learner
from prompt.prompt_learners import load_prompt_adapter
from prompt.prompt_encoder import get_prompt_encoder
from prompt.utils_vl import load_vl_model_to_cpu
from prompt.utils_vl import Tokenizer

def pairwise_cosine_distance_safe(x, y):
    # 增加 eps 防止除以 0
    x = F.normalize(x, dim=-1, eps=1e-6)
    y = F.normalize(y, dim=-1, eps=1e-6)
    return 1.0 - torch.matmul(x, y.t())

def sinkhorn_ot(mu, nu, cost, epsilon=0.05, n_iters=20):
    B, N, K1 = mu.shape
    _, _, K2 = nu.shape

    cost = cost.unsqueeze(0).unsqueeze(0).expand(B, N, K1, K2)
    K_mat = torch.exp(-cost / epsilon)

    u = torch.ones_like(mu) / K1
    v = torch.ones_like(nu) / K2

    for _ in range(n_iters):
        u = mu / (torch.einsum("bnij,bnj->bni", K_mat, v) + 1e-8)
        v = nu / (torch.einsum("bnij,bni->bnj", K_mat, u) + 1e-8)

    T = K_mat * u.unsqueeze(-1) * v.unsqueeze(-2)
    return T

# class VLSA_a(nn.Module):
#     def __init__(
#         self,
#         # text_encoder_cfg,
#         # image_encoder_cfg,
#         prompt_learner_cfg,
#         # [修改 1]: 移除了 T_struct_llm 和 T_bag_llm 参数
#         num_struct_prompts=2, # 你的 JSON 里有 2 类，这里建议默认设为 2
#         pretrained_prompt_learner_cfg=None,
#         info_prefix='VLSA-UNI',
#         num_vis_prototypes=6,
#         dim=512,
#         num_classes=2, 
#         pooling_type="attention", 
#         use_proj=True, 
#         ot_epsilon=0.05, 
#         ot_iter=20, 
#         loss_weights=None, 
#         ablation_setting=None,
#         num_heads=8,
#         **kwargs,
#     ) -> None:
#         super().__init__()

#         self.kwargs = kwargs
#         print(f"[{info_prefix}] Found additional kwargs: {self.kwargs}.")
#         assert 'vlsa_api' in kwargs, "Please specify `vlsa_api` in arguments."
#         assert 'path_clip_model' in kwargs, "Please specify `path_clip_model` in arguments."

#         self.text_tokenizer = Tokenizer(api='CONCH')

#         vl_model = load_vl_model_to_cpu(api='CONCH')
        
#         # Language-end
#         self.pmt_learner_name = prompt_learner_cfg['name']
#         self.prompt_encoder = get_prompt_encoder(vl_model, api=kwargs['vlsa_api'])
        
#         # [保留]: Prompt Learner 构建逻辑，这是你的黑箱来源
#         if self.pmt_learner_name == 'CoOp':
#             self.prompt_learner, pretrained_text_features = self._build_prompt_learner(
#                 prompt_learner_cfg, pretrained_prompt_learner_cfg
#             )
#             if pretrained_text_features is not None:
#                 pretrained_text_features = pretrained_text_features.detach().clone()
#                 self.register_buffer("pretrained_text_features", pretrained_text_features, persistent=False)
#                 print("[VLSA] warning: skip CoOp-based prompt learner and use pretrained text features.")
#         elif self.pmt_learner_name == 'Adapter':
#             self.prompt_adapter = self._build_prompt_adapter(prompt_learner_cfg, pretrained_prompt_learner_cfg)
#         else:
#             raise ValueError(f"{self.pmt_learner_name} is not a valid name of prompt learner.")

#         # Vision-end
#         if hasattr(vl_model, 'vision_model'):
#             assert kwargs['vlsa_api'] == 'HF'
#             self.mil_encoder = vl_model.vision_model
#         elif hasattr(vl_model, 'visual'):
#             assert kwargs['vlsa_api'] in ['CLIP', 'CONCH']
#             self.mil_encoder = vl_model.visual
#         else:
#             raise ValueError(f"[{info_prefix}] `vision_model` or `visual` is not found in {vl_model}.")

#         self.prompt_learner_cfg = prompt_learner_cfg
#         self.logit_scale = vl_model.logit_scale
#         self.dim = dim
#         self.K1 = num_struct_prompts
#         self.K2 = num_vis_prototypes
#         self.C = num_classes
#         self.use_proj = use_proj
#         self.ot_epsilon = ot_epsilon
#         self.ot_iter = ot_iter

#         self.ablation_setting = ablation_setting

#         # [修改 2]: 移除了 T_struct_llm/T_bag_llm 的 register_buffer
#         # [修改 3]: 移除了 prompt_inst/prompt_bag 的 nn.Parameter 初始化，改为只初始化视觉原型
#         self.proto_vis = nn.Parameter(torch.randn(self.K2, dim))
        
#         # 注意：这里我们不再有 self.prompt_inst 和 self.prompt_bag，它们将在 forward 中生成

#         if use_proj:
#             self.proj_v = nn.Sequential(
#                 nn.Linear(dim, dim),
#                 nn.LayerNorm(dim),
#                 nn.GELU(),
#             )
#             self.proj_llm = nn.Sequential(
#                 nn.Linear(dim, dim),
#                 nn.LayerNorm(dim),
#             )
#         else:
#             self.proj_v = nn.Identity()
#             self.proj_llm = nn.Identity()

#         self.temp_struct = nn.Parameter(torch.tensor(1.0))
#         self.temp_vis = nn.Parameter(torch.tensor(1.0))

#         # Cross Attention 参数
#         self.q_proj = nn.Linear(dim, dim)  
#         self.k_proj = nn.Linear(dim, dim)  
#         self.v_proj = nn.Linear(dim, dim)  
#         self.o_proj = nn.Linear(dim, dim)  
#         self.num_heads = num_heads
#         self.head_dim = dim // self.num_heads  
#         self.feature_dim = dim  

#         self.classification_head = nn.Sequential(
#             nn.LayerNorm(self.feature_dim),
#             nn.Linear(self.feature_dim, self.feature_dim),
#             nn.ReLU(),
#             nn.Linear(self.feature_dim, 1)  
#         )
       
#     def _build_prompt_learner(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
#         # ... (保持原样) ...
#         _prompt_learner_cfg = prompt_learner_cfg.copy()
#         _prompt_learner_cfg.update(dict(
#             tokenizer = self.text_tokenizer, 
#             text_config = self.prompt_encoder.text_config,
#             token_embedding = self.prompt_encoder.token_embedding
#         ))
#         prompt_learner = load_prompt_learner(_prompt_learner_cfg['method'], _prompt_learner_cfg)
#         pretrained_text_features = None
#         if _prompt_learner_cfg['pretrained']:
#             assert pretrained_prompt_learner_cfg is not None, "Please specify `config` for `pretrained_prompt_learner`."
#             prompt_learner.load_pretrained_parameters(pretrained_prompt_learner_cfg['ckpt'])
#             if _prompt_learner_cfg['frozen_context_embeds'] and _prompt_learner_cfg['frozen_rank_embeds']:
#                 with torch.no_grad():
#                     pretrained_text_features = self.compute_text_features_with_coop(prompt_learner)
#         return prompt_learner, pretrained_text_features

#     def _build_prompt_adapter(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
#         # ... (保持原样) ...
#         _prompt_learner_cfg = prompt_learner_cfg.copy()
#         _pretrained_prompt_learner_cfg = pretrained_prompt_learner_cfg.copy()
#         pretrained_text_features = None
#         if _prompt_learner_cfg['pretrained']:
#             _pretrained_prompt_learner_cfg['pretrained'] = True
#             _, pretrained_text_features = self._build_prompt_learner(
#                 _pretrained_prompt_learner_cfg, {'ckpt': _pretrained_prompt_learner_cfg['ckpt']}
#             )
#             assert pretrained_text_features is not None, "Found empty `pretrained_text_features`."
#             pretrained_text_features = pretrained_text_features.detach().clone()
#         _prompt_learner_cfg.update(dict(
#             tokenizer = self.text_tokenizer,
#             num_prompts = _prompt_learner_cfg['num_ranks'],
#             pretrained_prompt_features = pretrained_text_features,
#         ))
#         prompt_adapter = load_prompt_adapter(self.prompt_encoder, _prompt_learner_cfg)
#         return prompt_adapter

#     def compute_text_features_with_coop(self, prompt_learner):
#         sentence_embeds = prompt_learner()
#         pseudo_sentence_tokens = prompt_learner.pseudo_sentence_tokens
#         text_features = self.prompt_encoder(
#             prompts_embedding=sentence_embeds, 
#             prompts_pseudo_tokens=pseudo_sentence_tokens
#         )
#         return text_features

#     def forward_text_only(self):
#         if self.pmt_learner_name == 'CoOp':
#             text_features = self.compute_text_features_with_coop(self.prompt_learner)
#         elif self.pmt_learner_name == 'Adapter':
#             text_features = self.prompt_adapter()
#         else:
#             text_features = None
#         return text_features

#     def encode_instances(self, X):
#         return self.mil_encoder(X)

#     def get_logit_scale(self):
#         return self.logit_scale.exp()

#     def cross_attention(self, queries, keys, values, attention_mask=None):
#         bsz, q_len, _ = queries.size()
#         _, kv_len, _ = keys.size()
 
#         query_states = self.q_proj(queries)
#         key_states = self.k_proj(keys)
#         value_states = self.v_proj(values)
        
#         query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
#         key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
#         value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
#         attn_output = F.scaled_dot_product_attention(
#             query_states, key_states, value_states,
#             attn_mask=attention_mask
#         )
        
#         attn_output = attn_output.transpose(1, 2).contiguous()
#         attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)
#         attn_output = self.o_proj(attn_output)
        
#         return attn_output

#     def compute_loss(self, logits, labels):
#         loss_cls = F.cross_entropy(logits, labels)

#         return loss_cls

#     def forward(self, V_patch, labels=None):
#         B, N, D = V_patch.shape

#         # [修改 5]: 动态生成文本特征
#         # text_features shape: [2, D] (对应 class 0 和 class 1)
#         raw_text_feats = self.forward_text_only()

#         # [修改 6]: 分配特征
#         # 1. Structure Prompts: 用于 OT 对齐，我们将 0类 和 1类 的描述都作为结构化提示
#         # 这样 OT 就能计算 patch 与 "良性" 和 "恶性" 描述的距离
#         prompt_struct = self.proj_llm(raw_text_feats) # [K1=2, D]

#         # 2. Bag Prompt: 用于最终查询，我们选取 "Class 1" (通常是癌症/正类) 的特征作为查询向量
#         # 形状需要变为 [1, D]
#         prompt_bag_raw = raw_text_feats[1].unsqueeze(0) 
#         prompt_bag = self.proj_llm(prompt_bag_raw) # [1, D]

#         # 视觉投影
#         V_proj = self.proj_v(V_patch)
#         P_vis = self.proj_v(self.proto_vis)

#         # OT: 计算 Patch 与 Structure Prompts (0类和1类文本) 的相似度
#         attn_struct = torch.einsum("bnd,kd->bnk", V_proj, prompt_struct) / self.temp_struct.exp()
#         attn_struct = F.softmax(attn_struct, dim=-1) 

#         # OT: 计算 Patch 与 Visual Prototypes 的相似度
#         attn_vis = torch.einsum("bnd,kd->bnk", V_proj, P_vis) / self.temp_vis.exp()
#         attn_vis = F.softmax(attn_vis, dim=-1)  

#         # 运行 Sinkhorn
#         cost_matrix = pairwise_cosine_distance(prompt_struct, P_vis)
#         T = sinkhorn_ot(attn_struct, attn_vis, cost_matrix, epsilon=self.ot_epsilon, n_iters=self.ot_iter)

#         # 融合特征
#         attn_fused = torch.einsum("bnij->bnj", T)
#         attn_fused = F.softmax(attn_fused, dim=-1)
#         patch_fused = torch.einsum("bnk,bnd->bnd", attn_fused, V_proj)

#         # Cross Attention 聚合
#         # Query 需要扩展 Batch 维度: [1, D] -> [B, 1, D]
#         prompt_bag = prompt_bag.unsqueeze(0).expand(B, -1, -1)
#         cross_attn_output = self.cross_attention(queries=prompt_bag, keys=patch_fused, values=patch_fused)
        
#         logits = self.classification_head(cross_attn_output).squeeze(-1).squeeze(-1) # -> [B]

#         output = {
#             'logits': logits,
#         }

#         if labels is not None:
#             loss = self.compute_loss(logits, labels)
#             output['loss'] = loss
#         print(output['logits'])
#         return output['logits'],0.
    


class VLSA_a(nn.Module):
    def __init__(
        self,
        prompt_learner_cfg,
        num_struct_prompts=2, 
        pretrained_prompt_learner_cfg=None,
        info_prefix='VLSA-UNI',
        num_vis_prototypes=10,
        dim=512,
        num_classes=2,  # 确保这里是 2
        pooling_type="attention", 
        use_proj=True, 
        ot_epsilon=0.05, 
        ot_iter=10, 
        loss_weights=None, 
        ablation_setting=None,
        num_heads=8,
        **kwargs,
    ) -> None:
        super().__init__()

        self.kwargs = kwargs
        # ... (Tokenizer 和 VL Model 加载部分保持不变) ...
        assert 'vlsa_api' in kwargs, "Please specify `vlsa_api` in arguments."
        assert 'path_clip_model' in kwargs, "Please specify `path_clip_model` in arguments."
        self.text_tokenizer = Tokenizer(api='CONCH')
        vl_model = load_vl_model_to_cpu(api='CONCH')
        
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
        elif self.pmt_learner_name == 'Adapter':
            self.prompt_adapter = self._build_prompt_adapter(prompt_learner_cfg, pretrained_prompt_learner_cfg)
        else:
            raise ValueError(f"{self.pmt_learner_name} is not a valid name of prompt learner.")

        # Vision-end
        if hasattr(vl_model, 'vision_model'):
            self.mil_encoder = vl_model.vision_model
        elif hasattr(vl_model, 'visual'):
            self.mil_encoder = vl_model.visual
        
        self.dim = dim
        self.K1 = num_struct_prompts
        self.K2 = num_vis_prototypes
        self.num_classes = num_classes # 修正变量名一致性
        self.ot_epsilon = ot_epsilon
        self.ot_iter = ot_iter
        self.logit_scale = vl_model.logit_scale

        # 视觉原型
        self.proto_vis = nn.Parameter(torch.randn(self.K2, dim))

        if use_proj:
            self.proj_v = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
            self.proj_llm = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
        else:
            self.proj_v = nn.Identity()
            self.proj_llm = nn.Identity()

        self.temp_struct = nn.Parameter(torch.tensor(1.0))
        self.temp_vis = nn.Parameter(torch.tensor(1.0))

        # Cross Attention
        self.q_proj = nn.Linear(dim, dim)  
        self.k_proj = nn.Linear(dim, dim)  
        self.v_proj = nn.Linear(dim, dim)  
        self.o_proj = nn.Linear(dim, dim)  
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads  
        self.feature_dim = dim  

        # [修复1]: 输出维度改为 num_classes (2)，适配 main.py 的 argmax
        self.classification_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, num_classes) 
        )

    # ... (_build_prompt_learner, _build_prompt_adapter, compute_text_features_with_coop, forward_text_only, encode_instances 保持不变) ...
    # 为了节省篇幅，这里省略中间未修改的辅助函数，请保留你原有的代码
    def _build_prompt_learner(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
        _prompt_learner_cfg = prompt_learner_cfg.copy()
        _prompt_learner_cfg.update(dict(
            tokenizer = self.text_tokenizer, 
            text_config = self.prompt_encoder.text_config,
            token_embedding = self.prompt_encoder.token_embedding
        ))
        prompt_learner = load_prompt_learner(_prompt_learner_cfg['method'], _prompt_learner_cfg)
        pretrained_text_features = None
        if _prompt_learner_cfg['pretrained']:
            prompt_learner.load_pretrained_parameters(pretrained_prompt_learner_cfg['ckpt'])
            if _prompt_learner_cfg['frozen_context_embeds'] and _prompt_learner_cfg['frozen_rank_embeds']:
                with torch.no_grad():
                    pretrained_text_features = self.compute_text_features_with_coop(prompt_learner)
        return prompt_learner, pretrained_text_features

    def _build_prompt_adapter(self, prompt_learner_cfg, pretrained_prompt_learner_cfg):
        _prompt_learner_cfg = prompt_learner_cfg.copy()
        _pretrained_prompt_learner_cfg = pretrained_prompt_learner_cfg.copy()
        pretrained_text_features = None
        if _prompt_learner_cfg['pretrained']:
            _pretrained_prompt_learner_cfg['pretrained'] = True
            _, pretrained_text_features = self._build_prompt_learner(
                _pretrained_prompt_learner_cfg, {'ckpt': _pretrained_prompt_learner_cfg['ckpt']}
            )
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
        if self.pmt_learner_name == 'CoOp':
            text_features = self.compute_text_features_with_coop(self.prompt_learner)
        elif self.pmt_learner_name == 'Adapter':
            text_features = self.prompt_adapter()
        else:
            text_features = None
        return text_features

    def encode_instances(self, X):
        return self.mil_encoder(X)

    def get_logit_scale(self):
        return self.logit_scale.exp()

    def cross_attention(self, queries, keys, values, attention_mask=None):
        bsz, q_len, _ = queries.size()
        _, kv_len, _ = keys.size()
 
        query_states = self.q_proj(queries)
        key_states = self.k_proj(keys)
        value_states = self.v_proj(values)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=attention_mask
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.feature_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output

    def compute_loss(self, logits, labels):
        # [修复2]: 使用 Cross Entropy，它对整数标签更健壮，且适配 2 类输出
        loss_cls = F.cross_entropy(logits, labels)
        return loss_cls

    def forward(self, V_patch, labels=None):
        if isinstance(V_patch,(list,tuple)):
            V_patch = V_patch[0]
        B, N, D = V_patch.shape
        raw_text_feats = self.forward_text_only() # [2, D]

        # 构造 Prompts
        prompt_struct = self.proj_llm(raw_text_feats) # [K1=2, D]
        prompt_bag_raw = raw_text_feats[1].unsqueeze(0) # [1, D] 使用正类(索引1)作为Query
        prompt_bag = self.proj_llm(prompt_bag_raw) 

        V_proj = self.proj_v(V_patch)
        P_vis = self.proj_v(self.proto_vis)

        # [修复3]: 数值稳定性保护 (clamp)
        # 防止 temp 过小导致 exp 后除零
        temp_struct = torch.clamp(self.temp_struct.exp(), min=1e-4, max=100)
        temp_vis = torch.clamp(self.temp_vis.exp(), min=1e-4, max=100)

        attn_struct = torch.einsum("bnd,kd->bnk", V_proj, prompt_struct) / temp_struct
        attn_struct = F.softmax(attn_struct, dim=-1) 

        attn_vis = torch.einsum("bnd,kd->bnk", V_proj, P_vis) / temp_vis
        attn_vis = F.softmax(attn_vis, dim=-1)  

        # [修复4]: OT 距离计算加 eps，防止 normalize 除零
        cost_matrix = pairwise_cosine_distance_safe(prompt_struct, P_vis) 
        
        # 运行 Sinkhorn
        T = sinkhorn_ot(attn_struct, attn_vis, cost_matrix, epsilon=self.ot_epsilon, n_iters=self.ot_iter)

        attn_fused = torch.einsum("bnij->bnj", T)
        attn_fused = F.softmax(attn_fused, dim=-1)
        patch_fused = torch.einsum("bnk,bnd->bnd", attn_fused, V_proj)

        # Cross Attention
        prompt_bag = prompt_bag.unsqueeze(0).expand(B, -1, -1)
        cross_attn_output = self.cross_attention(queries=prompt_bag, keys=patch_fused, values=patch_fused)
        
        # [修复5]: 移除 squeeze，保持 (B, 2)
        logits = self.classification_head(cross_attn_output).squeeze(1) # [B, 1, 2] -> [B, 2]

        output = {'logits': logits}
        if labels is not None:
            loss = self.compute_loss(logits, labels)
            output['loss'] = loss

        return output['logits'],0.

# [修复6]: 请务必替换文件顶部的 distance 函数为这个安全版本
# def pairwise_cosine_distance_safe(x, y):
#     # 增加 eps 防止除以 0
#     x = F.normalize(x, dim=-1, eps=1e-6)
#     y = F.normalize(y, dim=-1, eps=1e-6)
#     return 1.0 - torch.matmul(x, y.t())