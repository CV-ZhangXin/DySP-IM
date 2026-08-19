import time
import torch
import wandb
import numpy as np
from copy import deepcopy
import torch.nn as nn
from dataloader import *
from torch.utils.data import DataLoader, RandomSampler
import argparse, os
# from modules import attmil,clam,mhim,dsmil,transmil,mean_max,diffmil,wikg
from modules import attmil,clam,mhim,dsmil,transmil,mean_max,diffmil,rrt,vis
from modules import diffusionnet as diffusionnet
from torch.nn.functional import one_hot
from torch.cuda.amp import GradScaler
from contextlib import suppress
import time

import os 
import h5py


from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict
import yaml
from utils import *
from prompt import vlsa,vlsa_a
import logging
from utils import parse_str_dims, fetch_kws, freeze_param, rename_keys



def draw_wsi_vision(model,V_patch):

    model_weight = torch.load('/data3/shihuazhan/output_wsi/mil_shz/libramil_call_5/fold_1_model_best_auc.pt')
    model.load_state_dict(model_weight['model'], strict=True)
      
    if isinstance(V_patch,(list,tuple)):
        V_patch = V_patch[0]

    tif_dir ='/data2/zhangxiaoxian/c16/slide'
    json_dir='/data2/zhangxiaoxian/c16/json'
    h5_dir ='/data3/Public/CAMELYON_ALL/camelyon_all_512_mag10_conch_feature'

    id_num = [4,13,14,15,16,20,21,34,42,46,52,55,62,64,71,72,73,76,82,83,84,85,89,90,91,95,101,104,105]


    for item in id_num:
        idx  = str(item)
        if len(idx) == 1:
            idx = '00' + idx
        elif len(idx) ==2:
            idx = '0' + idx

        f_id = 'tumor_'+str(idx)

        h5_path = os.path.join( h5_dir,f_id+".h5")
        scale_l = h5py.File(h5_path, 'r')

        with torch.no_grad():
            features = torch.from_numpy(np.array(scale_l['features'])).to(V_patch.device).unsqueeze(0)
            classifier_list,score_list,feature=model(features)
            feature = feature.squeeze(0)

        # 这个是wsi可视化
            classifier_list = classifier_list.to('cpu')
            vis.darw_original_img(f_id,tif_dir,json_dir,h5_dir)
            score = score_list[:,:,0,:].to('cpu')
            score = score.mean(1)
            vis.draw_attention(f_id,tif_dir,json_dir,h5_dir,classifier_list,score,'deep',feature.to('cpu'))

        classifier_list.to(V_patch.device)
    exit()


def main(args,cfg):
    # output_file = open(os.path.join(args.model_path, 'training_output.txt'), 'w')
    # sys.stdout = output_file
    log_file_path = os.path.join(args.model_path, 'app.log')
    # logging.basicConfig(level=logging.INFO,
    #                 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    #                 filename=log_file_path)  
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_path = log_file_path

    fh = logging.FileHandler(log_path, mode='w') 
    fh.setLevel(logging.DEBUG)  
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)



    # set seed
    seed_torch(args.seed)

    train_p, train_l = [], []
    val_p, val_l = [], []
    test_p, test_l = [], []
    num_folds_to_run = 0
    
    if args.use_split_files:
        label_path=os.path.join(args.dataset_root,'label.csv')
        if not os.path.exists(label_path):
            label_path = '/data3/Public/CAMELYON_ALL/label.csv'
        print(f"INFO: Running in pre-defined split mode from directory: {args.split_dir}")
        logging.info(f"Running in pre-defined split mode from directory: {args.split_dir}")
        try:
            all_patients, all_labels = get_patient_label(label_path)
            label_map = dict(zip(all_patients, all_labels))
        except Exception as e:
            print(f"ERROR: Failed to load or parse the main label CSV file with get_patient_label. Error: {e}")
            return
        for k in range(args.num_splits):
            split_file_path = os.path.join(args.split_dir, f'splits_{k}.csv')
            if not os.path.exists(split_file_path):
                print(f"WARNING: Split file not found, skipping: {split_file_path}")
                continue
            split_df = pd.read_csv(split_file_path, index_col=0)
            p_train_k = split_df['train'].dropna().tolist()
            p_val_k = split_df['val'].dropna().tolist()
            p_test_k = split_df['test'].dropna().tolist()
            try:
                l_train_k = [next((label for short_id, label in label_map.items() if short_id in name), None) for name in p_train_k]
                l_val_k   = [next((label for short_id, label in label_map.items() if short_id in name), None) for name in p_val_k]
                l_test_k  = [next((label for short_id, label in label_map.items() if short_id in name), None) for name in p_test_k]
                # l_train_k = [label_map[name] for name in p_train_k]
                # l_val_k = [label_map[name] for name in p_val_k]
                # l_test_k = [label_map[name] for name in p_test_k]
            except KeyError as e:
                print(f"ERROR: Slide ID {e} from {split_file_path} was not found in the label map created from {label_path}.")
                return
            train_p.append(p_train_k)
            train_l.append(l_train_k)
            val_p.append(p_val_k)
            val_l.append(l_val_k)
            test_p.append(p_test_k)
            test_l.append(l_test_k)
            
        num_folds_to_run = args.num_splits
        train_p, train_l = np.array(train_p, dtype=object), np.array(train_l, dtype=object)
        val_p, val_l = np.array(val_p, dtype=object), np.array(val_l, dtype=object)
        test_p, test_l = np.array(test_p, dtype=object), np.array(test_l, dtype=object)
    # --->get dataset
    else:
        if args.datasets.lower() == 'camelyon16':
            label_path=os.path.join(args.dataset_root,'label.csv')
            if not os.path.exists(label_path):
                label_path = '/data3/Public/CAMELYON_ALL/label.csv'
            p, l = get_patient_label(label_path)
            index = [i for i in range(len(p))]
            random.shuffle(index)
            p = p[index]
            l = l[index]

        elif args.datasets.lower() == 'tcga':
            label_path=os.path.join(args.dataset_root,'label.csv')
            p, l = get_patient_label(label_path)
            index = [i for i in range(len(p))]
            random.shuffle(index)
            p = p[index]
            l = l[index]

        elif args.datasets.lower() == 'bracs':
            label_path=os.path.join(args.dataset_root,'label.csv')
            if not os.path.exists(label_path):
                label_path=os.path.join(args.dataset_root,'labels.csv')
            p, l, d = get_patient_label_bracs(label_path)
            index = [i for i in range(len(p))]
            random.shuffle(index)
            p = p[index]
            l = l[index]
            d = d[index]
            if args.cv_fold == 1:
                train_p,train_l,test_p,test_l,val_p,val_l = [],[],[],[],[],[]
                for i in range(len(p)):
                    if 'Testing' in d[i]:
                        test_p.extend([p[i]])
                        test_l.extend([l[i]])
                    elif 'Validation' in d[i]:
                        val_p.extend([p[i]])
                        val_l.extend([l[i]])
                    else:
                        #print(p[i])
                        train_p.extend([p[i]])
                        train_l.extend([l[i]])
                train_p,train_l,test_p,test_l,val_p,val_l = np.array(train_p).reshape(1,-1),np.array(train_l).reshape(1,-1),np.array(test_p).reshape(1,-1),np.array(test_l).reshape(1,-1),np.array(val_p).reshape(1,-1),np.array(val_l).reshape(1,-1)
        if args.cv_fold > 1:
            train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)
        num_folds_to_run = args.cv_fold
    acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]

    if not args.no_log:
        print('Dataset: ' + args.datasets)
        logging.info('Dataset: ' + args.datasets)

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        args.fold_start = ckp['k']
        if len(ckp['ckc_metric']) == 6:
            acs, pre, rec,fs,auc,te_auc = ckp['ckc_metric']
        elif len(ckp['ckc_metric']) == 7:
            acs, pre, rec,fs,auc,te_auc,te_fs = ckp['ckc_metric']
        else:
            acs, pre, rec,fs,auc = ckp['ckc_metric']

    # for k in range(args.fold_start, args.cv_fold):
    #     if not args.no_log:
    #         print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
    #         logging.info('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
    #     ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l,cfg)
    for k in range(args.fold_start, num_folds_to_run):
        if not args.no_log:
            if args.use_split_files:
                log_msg = f'Start experiment using pre-defined split file: splits_{k}.csv'
            else:
                log_msg = f'Start {num_folds_to_run}-fold cross validation: fold {k}'
            print(log_msg)
            logging.info(log_msg)
        
        ckc_metric = one_fold(args, k, ckc_metric, train_p, train_l, test_p, test_l, val_p, val_l, cfg)

    if args.always_test:
        if args.wandb:
            wandb.log({
                "cross_val/te_auc_mean":np.mean(np.array(te_auc)),
                "cross_val/te_auc_std":np.std(np.array(te_auc)),
                "cross_val/te_f1_mean":np.mean(np.array(te_fs)),
                "cross_val/te_f1_std":np.std(np.array(te_fs)),
            })

    if args.wandb:
        wandb.log({
            "cross_val/acc_mean":np.mean(np.array(acs)),
            "cross_val/auc_mean":np.mean(np.array(auc)),
            "cross_val/f1_mean":np.mean(np.array(fs)),
            "cross_val/pre_mean":np.mean(np.array(pre)),
            "cross_val/recall_mean":np.mean(np.array(rec)),
            "cross_val/acc_std":np.std(np.array(acs)),
            "cross_val/auc_std":np.std(np.array(auc)),
            "cross_val/f1_std":np.std(np.array(fs)),
            "cross_val/pre_std":np.std(np.array(pre)),
            "cross_val/recall_std":np.std(np.array(rec)),
        })
    if not args.no_log:
        print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))
        print("*****************************************************************************")
        print('Cross validation accuracy mean: %.4f, std %.4f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        print('Cross validation auc mean: %.4f, std %.4f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        print('Cross validation precision mean: %.4f, std %.4f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        print('Cross validation recall mean: %.4f, std %.4f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        print('Cross validation fscore mean: %.4f, std %.4f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

        logging.info('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        logging.info('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        logging.info('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        logging.info('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        logging.info('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))
        logging.info("*****************************************************************************")
        logging.info('Cross validation accuracy mean: %.4f, std %.4f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        logging.info('Cross validation auc mean: %.4f, std %.4f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        logging.info('Cross validation precision mean: %.4f, std %.4f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        logging.info('Cross validation recall mean: %.4f, std %.4f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        logging.info('Cross validation fscore mean: %.4f, std %.4f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

    # sys.stdout = sys.__stdout__
    # output_file.close()





def func_load_model(cfg):
    arch = cfg['arch']
    key_vlsa_api = f'{arch.lower()}_api'
    assert key_vlsa_api in cfg, "Please specify the API for VLSA models."

    # prompt learner config
    pmt_learner_name = cfg['vlsa_pmt_learner_name']
    prompt_learner_cfg = fetch_kws(cfg, prefix=arch.lower() + '_pmt_learner_' + pmt_learner_name.lower())
    prompt_learner_cfg.update({"name": pmt_learner_name})

    # if use the text prompts pretrained by CoOp
    pmt_learner_pretrained = cfg['vlsa_pmt_learner_pretrained'] if 'vlsa_pmt_learner_pretrained' in cfg else False
    prompt_learner_cfg.update({"pretrained": pmt_learner_pretrained})
    if pmt_learner_pretrained:
        pretrained_prompt_learner_cfg = fetch_kws(cfg, prefix='vlsa_pmt_learner_coop')
        assert 'ckpt' in pretrained_prompt_learner_cfg and pretrained_prompt_learner_cfg['ckpt'] is not None, "Found null ckpt path."
        pretrained_prompt_learner_cfg['ckpt'] = pretrained_prompt_learner_cfg['ckpt'].format(cfg['data_split_seed'], pretrained_prompt_learner_cfg['method'])
        # assert pretrained_prompt_learner_cfg['frozen_context_embeds'], "Frozen context_embeds by default if pretrained."
        # assert pretrained_prompt_learner_cfg['frozen_rank_embeds'], "Frozen rank_embeds by default if pretrained."
    else:
        pretrained_prompt_learner_cfg = None

    text_encoder_cfg  = fetch_kws(cfg, prefix=arch.lower() + '_txt_encoder')
    image_encoder_cfg = fetch_kws(cfg, prefix=arch.lower() + '_img_encoder')
    arch_cfg = {
        'vlsa_api': cfg[key_vlsa_api],
        'text_encoder_cfg':   text_encoder_cfg,
        'image_encoder_cfg':  image_encoder_cfg,
        'prompt_learner_cfg': prompt_learner_cfg,
        'pretrained_prompt_learner_cfg': pretrained_prompt_learner_cfg,
        'path_clip_model': cfg['path_clip_model'],
        'k_ratio':args.k_ratio,
        'maskTh':args.maskTh,
        'maskPlan':args.maskPlan,
        'loss_total':args.loss_total,
        'headClass':args.headClass
    }
    model = vlsa.VLSA(**arch_cfg)

    if pmt_learner_name == 'CoOp':
        cfg_frozen_parameter = [
            ('prompt_learner.context_embeds', model.prompt_learner.context_embeds, prompt_learner_cfg['frozen_context_embeds']),
            ('prompt_learner.rank_embeds', model.prompt_learner.rank_embeds, prompt_learner_cfg['frozen_rank_embeds']),
            ('mil_encoder', model.mil_encoder, image_encoder_cfg['frozen']),
            ('text_encoder', model.prompt_encoder if hasattr(model, 'prompt_encoder') else model.text_encoder, text_encoder_cfg['frozen']),
            ('logit_scale', model.logit_scale, cfg[arch.lower() + '_frozen_logit_scale']),
        ]
    elif pmt_learner_name == 'Adapter':
        cfg_frozen_parameter = [
            ('mil_encoder', model.mil_encoder, image_encoder_cfg['frozen']),
            ('text_encoder', model.prompt_encoder if hasattr(model, 'prompt_encoder') else model.text_encoder, text_encoder_cfg['frozen']),
            ('logit_scale', model.logit_scale, cfg[arch.lower() + '_frozen_logit_scale']),
        ]
    else:
        cfg_frozen_parameter = []

    for name, module, frozen_it in cfg_frozen_parameter:
        if frozen_it:
            print(f"[setup] VLSA with prompt_learner ({pmt_learner_name}): freezing {name}.")
            try:
                freeze_param(module)
            except AttributeError:
                pass

    return model




def func_load_model_a(cfg):
    arch = cfg['arch']
    key_vlsa_api = f'{arch.lower()}_api'
    assert key_vlsa_api in cfg, "Please specify the API for VLSA models."

    # prompt learner config
    pmt_learner_name = cfg['vlsa_pmt_learner_name']
    prompt_learner_cfg = fetch_kws(cfg, prefix=arch.lower() + '_pmt_learner_' + pmt_learner_name.lower())
    prompt_learner_cfg.update({"name": pmt_learner_name})

    # if use the text prompts pretrained by CoOp
    pmt_learner_pretrained = cfg['vlsa_pmt_learner_pretrained'] if 'vlsa_pmt_learner_pretrained' in cfg else False
    prompt_learner_cfg.update({"pretrained": pmt_learner_pretrained})
    if pmt_learner_pretrained:
        pretrained_prompt_learner_cfg = fetch_kws(cfg, prefix='vlsa_pmt_learner_coop')
        assert 'ckpt' in pretrained_prompt_learner_cfg and pretrained_prompt_learner_cfg['ckpt'] is not None, "Found null ckpt path."
        pretrained_prompt_learner_cfg['ckpt'] = pretrained_prompt_learner_cfg['ckpt'].format(cfg['data_split_seed'], pretrained_prompt_learner_cfg['method'])
        # assert pretrained_prompt_learner_cfg['frozen_context_embeds'], "Frozen context_embeds by default if pretrained."
        # assert pretrained_prompt_learner_cfg['frozen_rank_embeds'], "Frozen rank_embeds by default if pretrained."
    else:
        pretrained_prompt_learner_cfg = None

    text_encoder_cfg  = fetch_kws(cfg, prefix=arch.lower() + '_txt_encoder')
    image_encoder_cfg = fetch_kws(cfg, prefix=arch.lower() + '_img_encoder')
    arch_cfg = {
        'vlsa_api': cfg[key_vlsa_api],
        'text_encoder_cfg':   text_encoder_cfg,
        'image_encoder_cfg':  image_encoder_cfg,
        'prompt_learner_cfg': prompt_learner_cfg,
        'pretrained_prompt_learner_cfg': pretrained_prompt_learner_cfg,
        'path_clip_model': cfg['path_clip_model'],
        'k_ratio':args.k_ratio,
        'maskTh':args.maskTh,
        'maskPlan':args.maskPlan,
        'loss_total':args.loss_total,
        'loss_text':args.loss_text,
        'loss_visual':args.loss_visual,
        'headClass':args.headClass
    }
    model = vlsa_a.VLSA_a(**arch_cfg)

    if pmt_learner_name == 'CoOp':
        cfg_frozen_parameter = [
            ('prompt_learner.context_embeds', model.prompt_learner.context_embeds, prompt_learner_cfg['frozen_context_embeds']),
            ('prompt_learner.rank_embeds', model.prompt_learner.rank_embeds, prompt_learner_cfg['frozen_rank_embeds']),
            ('mil_encoder', model.mil_encoder, image_encoder_cfg['frozen']),
            ('text_encoder', model.prompt_encoder if hasattr(model, 'prompt_encoder') else model.text_encoder, text_encoder_cfg['frozen']),
            ('logit_scale', model.logit_scale, cfg[arch.lower() + '_frozen_logit_scale']),
        ]
    elif pmt_learner_name == 'Adapter':
        cfg_frozen_parameter = [
            ('mil_encoder', model.mil_encoder, image_encoder_cfg['frozen']),
            ('text_encoder', model.prompt_encoder if hasattr(model, 'prompt_encoder') else model.text_encoder, text_encoder_cfg['frozen']),
            ('logit_scale', model.logit_scale, cfg[arch.lower() + '_frozen_logit_scale']),
        ]
    else:
        cfg_frozen_parameter = []

    for name, module, frozen_it in cfg_frozen_parameter:
        if frozen_it:
            print(f"[setup] VLSA with prompt_learner ({pmt_learner_name}): freezing {name}.")
            try:
                freeze_param(module)
            except AttributeError:
                pass

    return model



def count_parameters(model):
    # 兼容 DataParallel / DistributedDataParallel
    model = model.module if hasattr(model, "module") else model

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    frozen_params = total_params - trainable_params

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters:    {frozen_params:,}")
    print(f"Trainable ratio:      {trainable_params / total_params * 100:.2f}%")

    print(f"\nTotal parameters:     {total_params / 1e6:.3f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.3f} M")
    print(f"Frozen parameters:    {frozen_params / 1e6:.3f} M")

    return total_params, trainable_params



def one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l,cfg):
    # --->initiation
    seed_torch(args.seed)
    loss_scaler = GradScaler() if args.amp else None
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric

    # --->load data
    if args.datasets.lower() == 'camelyon16':

        train_set = C16Dataset(train_p[k],train_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
        test_set = C16Dataset(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        if args.use_split_files or args.val_ratio != 0.:
            val_set = C16Dataset(val_p[k],val_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        else:
            val_set = test_set

    elif args.datasets.lower() == 'tcga':
        
        train_set = TCGADataset(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True,_type=args.tcga_sub)
        test_set = TCGADataset(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
        if args.use_split_files or args.val_ratio != 0.:
            val_set = TCGADataset(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
        else:
            val_set = test_set

    elif args.datasets.lower() == 'bracs':  
        train_set = BRACSDataset(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,n_class=args.n_classes)
        test_set = BRACSDataset(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,n_class=args.n_classes)
        if args.use_split_files or args.val_ratio != 0.:
            val_set = BRACSDataset(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,n_class=args.n_classes)
        else:
            val_set = test_set


    if args.kshot < 1000:
        assert hasattr(args, 'kshot') and args.kshot > 0, "Please specify --k for small experiment"
        
        # Step 1: 获取原始训练集的所有标签
        original_labels = []
        for i in range(len(train_set)):
            # 注意：这里假设 __getitem__ 返回 (feature, label)，且 label 是 int
            # 如果 dataset 存储了 labels 属性（如 self.slide_label），可直接用，避免遍历
            if hasattr(train_set, 'slide_label'):
                original_labels = train_set.slide_label
                break
            elif hasattr(train_set, 'patient_label'):
                # 对于某些 dataset 可能是 patient-level，但 slide-level 才是实际样本
                # 所以最好还是通过 __getitem__ 或内部属性获取样本级标签
                pass
            else:
                _, label = train_set[i]
                original_labels.append(label)
        if not isinstance(original_labels, list):
            original_labels = list(original_labels)

        original_labels = np.array(original_labels)
        num_classes = len(np.unique(original_labels))

        # Step 2: 按类别采样 k 个样本索引
        selected_indices = []
        for cls in range(num_classes):
            cls_indices = np.where(original_labels == cls)[0]
            if len(cls_indices) < args.kshot:
                print(f"Warning: class {cls} has only {len(cls_indices)} samples, but k={args.kshot}. Using all.")
                selected_indices.extend(cls_indices.tolist())
            else:
                selected_indices.extend(np.random.choice(cls_indices, args.kshot, replace=False).tolist())

        selected_indices = sorted(selected_indices)  # 保持顺序（非必须）

        # Step 3: 创建子集
        from torch.utils.data import Subset
        small_train_set = Subset(train_set, selected_indices)
        print(f"[Small Exp] Selected {len(selected_indices)} samples ({args.kshot} per class)")

        # 替换 train_set 为 small_train_set
        train_set = small_train_set

    if args.fix_loader_random:
        # generated by int(torch.empty((), dtype=torch.int64).random_().item())
        big_seed_list = 7784414403328510413
        generator = torch.Generator()
        generator.manual_seed(big_seed_list)  
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    mm_sche = None
    if not args.teacher_init.endswith('.pt'):
        _str = 'fold_{fold}_model_best_auc.pt'.format(fold=k)
        _teacher_init = os.path.join(args.teacher_init,_str)
    else:
        _teacher_init =args.teacher_init

    # --->bulid networks
    if args.model == 'mhim':
        if args.mrh_sche:
            mrh_sche = cosine_scheduler(args.mask_ratio_h,0.,epochs=args.num_epoch,niter_per_ep=len(train_loader))
        else:
            mrh_sche = None

        model_params = {
            'baseline': args.baseline,
            'dropout': args.dropout,
            'mask_ratio' : args.mask_ratio,
            'n_classes': args.n_classes,
            'temp_t': args.temp_t,
            'act': args.act,
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion,
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr,
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche,
            'da_act': args.da_act,
            'attn_layer': args.attn_layer,
        }
        
        if args.mm_sche:
            mm_sche = cosine_scheduler(args.mm,args.mm_final,epochs=args.num_epoch,niter_per_ep=len(train_loader),start_warmup_value=1.)

        model = mhim.MHIM(**model_params).to(device)
    elif args.model == 'conch':
        model = func_load_model(cfg).to(device)  
    elif args.model == 'conch_a':
        model = func_load_model_a(cfg).to(device)  
    elif args.model == 'diff':
        model = diffmil.TransmilWithMining(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)
    elif args.model == 'dualTrans':
        model = diffmil.DualTransmilforExperiment(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)
    # elif args.model == 'wikg':
    #     model = wikg.WiKG(dim_in=1024, dim_hidden=512, topk=args.wikg_topk, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn').to(device)
    elif args.model == 'difftune':
        model = diffmil.DAttentionWithDiffTune(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)
    elif args.model == 'diffSim':
        model = diffmil.DAttentionWithDiffEndTransmil(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain,a_ratio=args.a_ratio,adapter_ratio=args.adapter_ratio,a_num=args.a_num).to(device)
    elif args.model == 'diff2End':
        model = diffmil.DAttentionWithDiffTwoEnd(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)
    elif args.model == 'diffCon':
        model = diffmil.DAttentionWithDiffContrast(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)
    elif args.model == 'diffusionnet':
        model = diffusionnet.DiffusionNet(out_dim=args.n_classes,t=args.t_steps).to(device) 
    elif args.model == 'random':
        model = diffmil.DAttentionWithRandomAbandon(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)           
    elif args.model == 'chose':
        model = diffmil. DAttentionWithDiffchose(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain,ifType=args.ifType,ifClose=args.ifClose).to(device)           
    elif args.model == 'pure':
        model = mhim.MHIM(select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline).to(device)
    elif args.model == 'attmil':
        model = attmil.DAttention(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'gattmil':
        model = attmil.AttentionGated(dropout=args.dropout).to(device)
    elif args.model == 'frmil':
        model = frmil.FRMIL(num_class = args.n_classes)
    elif args.model == 'rrtmil':
        model_params = {
            'input_dim': args.input_dim,
            'n_classes': args.n_classes,
            'dropout': args.dropout,
            'act': args.act,
            'region_num': args.region_num,
            'pos': args.pos,
            'pos_pos': args.pos_pos,
            'pool': args.pool,
            'peg_k': args.peg_k,
            'drop_path': args.drop_path,
            'n_layers': args.n_trans_layers,
            'n_heads': args.n_heads,
            'attn': args.attn,
            'da_act': args.da_act,
            'trans_dropout': args.trans_drop_out,
            'ffn': args.ffn,
            'mlp_ratio': args.mlp_ratio,
            'trans_dim': args.trans_dim,
            'epeg': args.epeg,
            'min_region_num': args.min_region_num,
            'qkv_bias': args.qkv_bias,
            'epeg_k': args.epeg_k,
            'epeg_2d': args.epeg_2d,
            'epeg_bias': args.epeg_bias,
            'epeg_type': args.epeg_type,
            'region_attn': args.region_attn,
            'peg_1d': args.peg_1d,
            'cr_msa': args.cr_msa,
            'crmsa_k': args.crmsa_k,
            'all_shortcut': args.all_shortcut,
            'crmsa_mlp':args.crmsa_mlp,
            'crmsa_heads':args.crmsa_heads,
         }
        model = rrt.RRTMIL(**model_params).to(device)
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model == 'clam_sb':
        model = clam.CLAM_SB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'clam_mb':
        model = clam.CLAM_MB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'transmil':
        model = transmil.TransMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'transmilDiff':
        model = transmil.TransMILwithDiff(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'dsmil':
        model = dsmil.MILNet(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5
        state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
        info = model.load_state_dict(state_dict_weights, strict=False)
        if not args.no_log:
            print(info)
    elif args.model == 'meanmil':
        model = mean_max.MeanMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'maxmil':
        model = mean_max.MaxMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)

    if args.init_stu_type != 'none':
        if not args.no_log:
            print('######### Model Initializing.....')
            logging.info('######### Model Initializing.....')
        pre_dict = torch.load(_teacher_init)
        new_state_dict ={}
        if args.init_stu_type == 'fc':
        # only patch_to_emb
            for _k,v in pre_dict.items():
                _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
                new_state_dict[_k]=v
            info = model.patch_to_emb.load_state_dict(new_state_dict,strict=False)
        else:
        # init all
            info = model.load_state_dict(pre_dict,strict=False)
        if not args.no_log:
            print(info)
            logging.info(info)

    # teacher model
    if args.model == 'mhim':
        model_tea = deepcopy(model)
        if not args.no_tea_init and args.tea_type != 'same':
            if not args.no_log:
                print('######### Teacher Initializing.....')
                logging.info('######### Teacher Initializing.....')
            try:
                pre_dict = torch.load(_teacher_init)
                info = model_tea.load_state_dict(pre_dict,strict=False)
                if not args.no_log:
                    print(info)
                    logging.info(info)
            except:
                if not args.no_log:
                    print('########## Init Error')
                    logging.info('########## Init Error')
        if args.tea_type == 'same':
            model_tea = model
    else:
        model_tea = None

    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()

    # optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_thr,opt_epoch = 0, 0, 0, 0,0,0,0
    opt_te_auc,opt_tea_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs  = 0., 0., 0., 0., 0.
    epoch_start = 0

    if args.fix_train_random:
        seed_torch(args.seed)

    # resume
    if args.auto_resume and not args.no_log:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        epoch_start = ckp['epoch']
        model.load_state_dict(ckp['model'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['lr_sche'])
        early_stopping.load_state_dict(ckp['early_stop'])
        optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = ckp['val_best_metric']
        opt_te_auc = ckp['te_best_metric'][0]
        if len(ckp['te_best_metric']) > 1:
            opt_te_fs = ckp['te_best_metric'][1]
        opt_te_tea_auc,opt_te_tea_fs = ckp['te_best_metric'][2:4]
        np.random.set_state(ckp['random']['np'])
        torch.random.set_rng_state(ckp['random']['torch'])
        random.setstate(ckp['random']['py'])
        if args.fix_loader_random:
            train_loader.sampler.generator.set_state(ckp['random']['loader'])
        args.auto_resume = False

    train_time_meter = AverageMeter()
    
    count_parameters(model)

    for epoch in range(epoch_start, args.num_epoch):
        import time
        # time1 = time.time()
        train_loss,start,end = train_loop(args,model,model_tea,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch)
        train_time_meter.update(end-start)
        stop,accuracy, auc_value, precision, recall, fscore, test_loss, threshold_optimal = val_loop(args,model,val_loader,device,criterion,early_stopping,epoch,model_tea)
        # time2 = time.time()
        # print(time2-time1)
        if model_tea is not None:
            _,accuracy_tea, auc_value_tea, precision_tea, recall_tea, fscore_tea, test_loss_tea = val_loop(args,model_tea,val_loader,device,criterion,None,epoch,model_tea)
            if args.wandb:
                rowd = OrderedDict([
                    ("val_acc_tea",accuracy_tea),
                    ("val_precision_tea",precision_tea),
                    ("val_recall_tea",recall_tea),
                    ("val_fscore_tea",fscore_tea),
                    ("val_auc_tea",auc_value_tea),
                    ("val_loss_tea",test_loss_tea),
                ])

                rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)

            if auc_value_tea > opt_tea_auc:
                opt_tea_auc = auc_value_tea
                if args.wandb:
                    rowd = OrderedDict([
                        ("best_tea_auc",opt_tea_auc)
                    ])
                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)

        if args.always_test:

            _te_accuracy, _te_auc_value, _te_precision, _te_recall, _te_fscore,_te_test_loss_log = test(args,model,test_loader,device,criterion,model_tea)
            
            if args.wandb:
                rowd = OrderedDict([
                    ("te_acc",_te_accuracy),
                    ("te_precision",_te_precision),
                    ("te_recall",_te_recall),
                    ("te_fscore",_te_fscore),
                    ("te_auc",_te_auc_value),
                    ("te_loss",_te_test_loss_log),
                ])

                rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                wandb.log(rowd)

            if _te_auc_value > opt_te_auc:
                opt_te_auc = _te_auc_value
                opt_te_fs = _te_fscore
                if args.wandb:
                    rowd = OrderedDict([
                        ("best_te_auc",opt_te_auc),
                        ("best_te_f1",_te_fscore)
                    ])
                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)
            
            if model_tea is not None:
                _te_tea_accuracy, _te_tea_auc_value, _te_tea_precision, _te_tea_recall, _te_tea_fscore,_te_tea_test_loss_log = test(args,model_tea,test_loader,device,criterion,model_tea)
            
                if args.wandb:
                    rowd = OrderedDict([
                        ("te_tea_acc",_te_tea_accuracy),
                        ("te_tea_precision",_te_tea_precision),
                        ("te_tea_recall",_te_tea_recall),
                        ("te_tea_fscore",_te_tea_fscore),
                        ("te_tea_auc",_te_tea_auc_value),
                        ("te_tea_loss",_te_tea_test_loss_log),
                    ])

                    rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                    wandb.log(rowd)

                if _te_tea_auc_value > opt_te_tea_auc:
                    opt_te_tea_auc = _te_tea_auc_value
                    opt_te_tea_fs = _te_tea_fscore
                    if args.wandb:
                        rowd = OrderedDict([
                            ("best_te_tea_auc",opt_te_tea_auc),
                            ("best_te_tea_f1",_te_fscore)
                        ])
                        rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
                        wandb.log(rowd)
        if not args.no_log:
            print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
        (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))
            logging.info('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
        (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))

        if args.wandb:
            rowd = OrderedDict([
                ("val_acc",accuracy),
                ("val_precision",precision),
                ("val_recall",recall),
                ("val_fscore",fscore),
                ("val_auc",auc_value),
                ("val_loss",test_loss),
                ("epoch",epoch),
            ])

            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            wandb.log(rowd)

        if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.num_epoch:
            optimal_ac = accuracy
            opt_pre = precision
            opt_re = recall
            opt_fs = fscore
            opt_auc = auc_value
            opt_thr = threshold_optimal
            opt_epoch = epoch

            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            if not args.no_log:
                best_pt = {
                    'model': model.state_dict(),
                    'teacher': model_tea.state_dict() if model_tea is not None else None,
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
        if args.wandb:
            rowd = OrderedDict([
                ("val_best_acc",optimal_ac),
                ("val_best_precesion",opt_pre),
                ("val_best_recall",opt_re),
                ("val_best_fscore",opt_fs),
                ("val_best_auc",opt_auc),
                ("val_best_epoch",opt_epoch),
            ])

            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            wandb.log(rowd)
        
        # save checkpoint
        random_state = {
            'np': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
            'py': random.getstate(),
            'loader': train_loader.sampler.generator.get_state() if args.fix_loader_random else '',
        }
        ckp = {
            'model': model.state_dict(),
            'lr_sche': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'k': k,
            'early_stop': early_stopping.state_dict(),
            'random': random_state,
            'ckc_metric': [acs,pre,rec,fs,auc,te_auc,te_fs],
            'val_best_metric': [optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch],
            'te_best_metric': [opt_te_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs],
            'wandb_id': wandb.run.id if args.wandb else '',
        }
        if not args.no_log:
            torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))

        if stop:
            break
    
    # test
    if not args.no_log:
        best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
        info = model.load_state_dict(best_std['model'])
        print(info)
        logging.info(info)
        if model_tea is not None and best_std['teacher'] is not None:
            info = model_tea.load_state_dict(best_std['teacher'])
            print(info)
            logging.info(info)

    accuracy, auc_value, precision, recall, fscore,test_loss_log = test(args,model,test_loader,device,criterion,model_tea,opt_thr)
    
    if args.wandb:
        wandb.log({
            "test_acc":accuracy,
            "test_precesion":precision,
            "test_recall":recall,
            "test_fscore":fscore,
            "test_auc":auc_value,
            "test_loss":test_loss_log,
        })
    if not args.no_log:
        print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
        logging.info('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)

    if args.always_test:
        te_auc.append(opt_te_auc)
        te_fs.append(opt_te_fs)
        
    return [acs,pre,rec,fs,auc,te_auc,te_fs]

def train_loop(args,model,model_tea,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch):
    start = time.time()
    loss_cls_meter = AverageMeter()
    loss_cl_meter = AverageMeter()
    loss_cl_p_meter = AverageMeter()
    loss_cl_n_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    keep_num_meter = AverageMeter()
    mm_meter = AverageMeter()
    train_loss_log = 0.
    model.train()
    if model_tea is not None:
        model_tea.train()

    for i, data in enumerate(loader):
        optimizer.zero_grad()

        if isinstance(data[0],(list,tuple)):
            for i in range(len(data[0])):
                data[0][i] = data[0][i].to(device)
            bag=data[0]
            batch_size=data[0][0].size(0)
        else:
            bag=data[0].to(device)  # b*n*1024
            batch_size=bag.size(0)
            
        label=data[1].to(device)
        label_negative = torch.zeros_like(label)
        label_positive = torch.ones_like(label)

        with amp_autocast():
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)

            if args.model == 'mhim':
                if model_tea is not None:
                    cls_tea,attn = model_tea.forward_teacher(bag,return_attn=True)
                else:
                    attn,cls_tea = None,None

                cls_tea = None if args.cl_alpha == 0. else cls_tea

                train_logits, cls_loss,patch_num,keep_num = model(bag,attn,cls_tea,i=epoch*len(loader)+i)

            elif args.model == 'pure':
                train_logits, cls_loss,patch_num,keep_num = model.pure(bag)
            elif args.model in ('clam_sb','clam_mb','dsmil'):
                train_logits,cls_loss,patch_num = model(bag,label,criterion)
                keep_num = patch_num
            elif args.model== 'diffusionnet': #shz
                train_logits = model(bag)
                cls_loss,patch_num,keep_num = 0.,0.,0.
            # elif args.model== 'diff2End': #shz diff2End
            #     train_logits,cls_logits_p,cls_logits_n = model(bag)
            #     if args.loss == 'ce':
            #         cls_loss_n = criterion(cls_logits_n.view(batch_size,-1),label_negative)
            #         cls_loss_p = criterion(cls_logits_p.view(batch_size,-1),label_positive)
            #     elif args.loss == 'bce':
            #         cls_loss_n = criterion(cls_logits_n.view(batch_size,-1),one_hot(label_negative.view(batch_size,-1).float(),num_classes=2))
            #         cls_loss_p = criterion(cls_logits_p.view(batch_size,-1),one_hot(label_positive.view(batch_size,-1).float(),num_classes=2))
            #     patch_num,keep_num,cls_loss = 0.,0.,0.
            elif args.model== 'diffSim' or args.model=='transmilDiff': 
            
                train_logits,cls_logits = model(bag)
                if args.loss == 'ce':
                    cls_loss = criterion(cls_logits.view(batch_size,-1),label_negative)
                elif args.loss == 'bce':
                    cls_loss = criterion(cls_logits.view(batch_size,-1),one_hot(label_negative.view(batch_size,-1).float(),num_classes=2))
                patch_num,keep_num = 0.,0.
            
            elif args.model== 'conch' or args.model== 'conch_a':
        
                train_logits,cls_loss = model(bag) #shz 这个和下面这个2选1 


                # #可视化的
                # draw_wsi_vision(model,bag)



                # train_logits,cls_logits = model(bag)
                # if args.loss == 'ce':
                #     cls_loss = criterion(cls_logits.view(batch_size,-1),label)
                # elif args.loss == 'bce':
                #     cls_loss = criterion(cls_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))
                patch_num,keep_num = 0.,0.
    
            else:
                train_logits = model(bag)
                cls_loss,patch_num,keep_num = 0.,0.,0.

            if args.loss == 'ce':
                logit_loss = criterion(train_logits.view(batch_size,-1),label)
                
            elif args.loss == 'bce':
                logit_loss = criterion(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))

        #train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha

        #train_loss = args.cls_alpha * logit_loss +  cls_loss * 1   #shz
        if args.model== 'diff2End':
            train_loss = args.cls_alpha * logit_loss +  cls_loss_p * 1 + cls_loss_n * 1  #shz
        else:
            train_loss = args.cls_alpha * logit_loss +  cls_loss * args.cl_alpha   #shz
        train_loss = train_loss / args.accumulation_steps
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        if (i+1) % args.accumulation_steps == 0:
            train_loss.backward()
            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()
            if args.model == 'mhim':
                if mm_sche is not None:
                    mm = mm_sche[epoch*len(loader)+i]
                else:
                    mm = args.mm
                if model_tea is not None:
                    if args.tea_type == 'same':
                        pass
                    else:
                        ema_update(model,model_tea,mm)
            else:
                mm = 0.
        # # for name, parms in model.named_parameters():
        # #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
        # for name, parms in model.named_parameters():
        #     if parms.grad is None:
        #         #print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
        #         a=1
        #     else:
        #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', "Yes")

        # # print(model)
        # print("__________________________*****___________________________________*******__________________") #shz
        loss_cls_meter.update(logit_loss,1)
        loss_cl_meter.update(cls_loss,1)
        #loss_cl_p_meter.update(cls_loss_p,1)
        #loss_cl_n_meter.update(cls_loss_n,1)
        patch_num_meter.update(patch_num,1)
        keep_num_meter.update(keep_num,1)
        mm_meter.update(mm,1)

        if i % args.log_iter == 0 or i == len(loader)-1:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            rowd = OrderedDict([
                ('cls_loss',loss_cls_meter.avg),
                ('lr',lr),
                ('cl_loss',loss_cl_meter.avg),
                ('cl_loss_p',loss_cl_p_meter.avg),
                ('cl_loss_n',loss_cl_n_meter.avg),
                ('patch_num',patch_num_meter.avg),
                ('keep_num',keep_num_meter.avg),
                ('mm',mm_meter.avg),
            ])
            if not args.no_log:
                print('[{}/{}] logit_loss:{}, cls_loss:{},patch_num:{}, keep_num:{} '.format(i,len(loader)-1,loss_cls_meter.avg,loss_cl_meter.avg,patch_num_meter.avg, keep_num_meter.avg))
                logging.info('[{}/{}] logit_loss:{}, cls_loss:{},patch_num:{}, keep_num:{} '.format(i,len(loader)-1,loss_cls_meter.avg,loss_cl_meter.avg,patch_num_meter.avg, keep_num_meter.avg))
            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            if args.wandb:
                wandb.log(rowd)

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    train_loss_log = train_loss_log/len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log,start,end

def val_loop(args,model,loader,device,criterion,early_stopping,epoch,model_tea=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    loss_cls_meter = AverageMeter()
    bag_logit, bag_labels=[], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
            elif args.model == 'diffSim' or args.model=='transmilDiff':
            #elif args.model in ('diffSim','diff2End'):
                test_logits,_ = model(bag)
            elif args.model == 'diff2End':
            #elif args.model in ('diffSim','diff2End'):
                test_logits,_,_ = model(bag)
            elif args.model== 'conch' or args.model== 'conch_a':
                test_logits,_= model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if args.n_classes<=2:
                        if batch_size > 1:
                            bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                        else:
                            bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        if batch_size > 1:
                            bag_logit.extend(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
                        else:
                            bag_logit.append(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
                    
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label.view(batch_size,-1).float())
                    
                    bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            loss_cls_meter.update(test_loss,1)

    # save the log file
    # accuracy, auc_value, precision, recall, fscore, threshold_optimal = five_scores(bag_labels, bag_logit) #five_scores_new
    accuracy, auc_value, precision, recall, fscore, threshold_optimal = five_scores_new(bag_labels, bag_logit,threshold_optimal=None,n_classes=args.n_classes) #shz
    # early stop
    if early_stopping is not None:
        early_stopping(epoch,-auc_value,model)
        stop = early_stopping.early_stop
    else:
        stop = False
    return stop,accuracy, auc_value, precision, recall, fscore,loss_cls_meter.avg, threshold_optimal

def test(args,model,loader,device,criterion,model_tea=None,opt_thr=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    test_loss_log = 0.
    bag_logit, bag_labels=[], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())
                
            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
            elif args.model == 'diffSim' or args.model=='transmilDiff':
            # elif args.model in ('diffSim','diff2End'):
                test_logits,_ = model(bag)
            elif args.model == 'diff2End':
            # elif args.model in ('diffSim','diff2End'):
                test_logits,_,_ = model(bag)
            elif args.model== 'conch' or args.model== 'conch_a':
                test_logits,_= model(bag)
            else:
                test_logits = model(bag)

            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                    if args.n_classes<=2:
                        if batch_size > 1:
                            bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                        else:
                            bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
                    else:
                        if batch_size > 1:
                            bag_logit.extend(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
                        else:
                            bag_logit.append(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    test_loss = criterion(test_logits[0].view(batch_size,-1),label)
                    bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    test_loss = criterion(test_logits.view(batch_size,-1),label.view(1,-1).float())
                bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

            test_loss_log = test_loss_log + test_loss.item()
    
    # save the log file
    # cal the best thr with val set
    opt_thr = opt_thr if args.best_thr_val else None
    #ccuracy, auc_value, precision, recall, fscore, _ = five_scores(bag_labels, bag_logit,threshold_optimal=opt_thr)
    accuracy, auc_value, precision, recall, fscore, threshold_optimal = five_scores_new(bag_labels, bag_logit,threshold_optimal=None,n_classes=args.n_classes) #shz
    test_loss_log = test_loss_log/len(loader)

    return accuracy, auc_value, precision, recall, fscore,test_loss_log

def get_config(config_path):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')

    #conch
    parser.add_argument('--config', '-f', required=False, type=str, help='Path to the config file.')

    # Dataset 
    parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga]')
    parser.add_argument('--dataset_root', default='/data/xxx/TCGA', type=str, help='Dataset root path')
    parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
    parser.add_argument('--tcga_sub', default='nsclc', type=str, help='[nsclc,brca]')

    # Train
    parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--model', default='mhim', type=str, help='Model name')
    parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
    parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')
    parser.add_argument('--best_thr_val', action='store_true', help='Cal the best thr with val set in the test phase. Thanks Weiyi Wu!')
    parser.add_argument('--input_dim', default=1024, type=int, help='dim of input features. PLIP features should be [512]')

    # Model
    # diffusion shz
    parser.add_argument('--ifType', default=1, type=int, help='to chose the method of chosing arch')
    parser.add_argument('--k_ratio', default=0.1, type=float, help='Number of total k ratio')
    parser.add_argument('--t_steps', default=2, type=int, help='t in diffusion model')
    parser.add_argument('--ifTrain', default=1, type=int, help='if 0 means train and test use the same method,1 means different')
    parser.add_argument('--ifrand', default=0, type=int, help='if 0 means using diff,1 means using rand')
    parser.add_argument('--temp_nums', default=100, type=int, help='the numbers of templates made by diffusion model')
    parser.add_argument('--ifEma', default=0, type=int, help='if 0 means no Ema in the sharing weights')
    parser.add_argument('--ifClose', default=0, type=int, help='if 0 means far, if 1 means near')
    parser.add_argument('--adapter_ratio', default=1.0, type=float, help='adapter ratio')
    parser.add_argument('--a_ratio', default=1.0, type=float, help='a ratio')
    parser.add_argument('--a_num', default=1, type=int, help='a number')
    # wikg wikg_topk
    parser.add_argument('--wikg_topk', default=6, type=int, help='no')



    # Our
    parser.add_argument('--baseline', default='selfattn', type=str, help='Baselin model [attn,selfattn]')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # MHIM
    # Mask ratio
    parser.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio')
    parser.add_argument('--mask_ratio_l', default=0., type=float, help='Low attention mask ratio')
    parser.add_argument('--mask_ratio_h', default=0., type=float, help='High attention mask ratio')
    parser.add_argument('--mask_ratio_hr', default=1., type=float, help='Randomly high attention mask ratio')
    parser.add_argument('--mrh_sche', action='store_true', help='Decay of HAM')
    parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote]')
    parser.add_argument('--attn_layer', default=0, type=int)
    
    # Siamese framework
    parser.add_argument('--cl_alpha', default=1., type=float, help='Auxiliary loss alpha')
    parser.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
    parser.add_argument('--teacher_init', default='none', type=str, help='Path to initial teacher model')
    parser.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization')
    parser.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none,fc,all]')
    parser.add_argument('--tea_type', default='none', type=str, help='[none,same]')
    parser.add_argument('--mm', default=0.9999, type=float, help='Ema decay [0.9997]')
    parser.add_argument('--mm_final', default=1., type=float, help='Final ema decay [1.]')
    parser.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay')

    # Misc
    parser.add_argument('--title', default='default', type=str, help='Title of exp')
    parser.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--model_path', type=str, help='Output path')

    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--only_rrt_enc',action='store_true', help='RRT+other MIL models [dsmil,clam,]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    # Transformer
    parser.add_argument('--attn', default='rmsa', type=str, help='Inner attention')
    parser.add_argument('--pool', default='attn', type=str, help='Classification poolinp. use abmil.')
    parser.add_argument('--ffn', action='store_true', help='Feed-forward network. only for ablation')
    parser.add_argument('--n_trans_layers', default=2, type=int, help='Number of layer in the transformer')
    parser.add_argument('--mlp_ratio', default=4., type=int, help='Ratio of MLP in the FFN')
    parser.add_argument('--qkv_bias', action='store_false')
    parser.add_argument('--all_shortcut', action='store_true', help='x = x + rrt(x)')
    # R-MSA
    parser.add_argument('--region_attn', default='native', type=str, help='only for ablation')
    parser.add_argument('--min_region_num', default=0, type=int, help='only for ablation')
    parser.add_argument('--region_num', default=8, type=int, help='Number of the region. [8,12,16,...]')
    parser.add_argument('--trans_dim', default=64, type=int, help='only for ablation')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the R-MSA')
    parser.add_argument('--trans_drop_out', default=0.1, type=float, help='Dropout in the R-MSA')
    parser.add_argument('--drop_path', default=0., type=float, help='Droppath in the R-MSA')
    # PEG or PPEG. only for alation
    parser.add_argument('--pos', default='none', type=str, help='Position embedding, enable PEG or PPEG')
    parser.add_argument('--pos_pos', default=0, type=int, help='Position of pos embed [-1,0]')
    parser.add_argument('--peg_k', default=7, type=int, help='K of the PEG and PPEG')
    parser.add_argument('--peg_1d', action='store_true', help='1-D PEG and PPEG')
    # EPEG
    parser.add_argument('--epeg', action='store_false', help='enable epeg')
    parser.add_argument('--epeg_bias', action='store_false', help='enable conv bias')
    parser.add_argument('--epeg_2d', action='store_true', help='enable 2d conv. only for ablation')
    parser.add_argument('--epeg_k', default=15, type=int, help='K of the EPEG. [9,15,21,...]')
    parser.add_argument('--epeg_type', default='attn', type=str, help='only for ablation')
    # CR-MSA
    parser.add_argument('--cr_msa', action='store_false', help='enable CR-MSA')
    parser.add_argument('--crmsa_k', default=3, type=int, help='K of the CR-MSA. [1,3,5]')
    parser.add_argument('--crmsa_heads', default=8, type=int, help='head of CR-MSA. [1,8,...]')
    parser.add_argument('--crmsa_mlp', action='store_true', help='mlp phi of CR-MSA?')

    # conch
    parser.add_argument('--maskTh', default=0.5, type=float, help='enable mask')
    parser.add_argument('--maskPlan', default=3, type=int, help='enable mask')
    parser.add_argument('--kshot', default=2000, type=int, help='few—shot')
    parser.add_argument('--headClass', default=1, type=int, help='enable mask')
    parser.add_argument('--loss_total', default=0.1, type=float, help='loss_total')
    parser.add_argument('--loss_text', default=0.5, type=float, help='loss_text')
    parser.add_argument('--loss_visual', default=0.5, type=float, help='loss_visual')

    #few-shot
    parser.add_argument('--use_split_files', action='store_true', help='Use pre-defined split CSV files instead of internal k-fold')
    parser.add_argument('--split_dir', type=str, default=None, help='Directory containing the split CSV files (e.g., splits_0.csv)')
    parser.add_argument('--num_splits', type=int, default=10, help='Number of splits to run when using split files')
    # 注意：这个参数很重要，需要指向你的全局 label.csv 文件
    # parser.add_argument('--label_csv_path', type=str, default='path/to/your/label.csv', help='Path to the global label CSV file')


    args = parser.parse_args()
    config_conch = get_config(args.config)

    if not os.path.exists(os.path.join(args.model_path,args.project)):
        os.mkdir(os.path.join(args.model_path,args.project))
    args.model_path = os.path.join(args.model_path,args.project,args.title)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if args.model == 'pure':
        args.cl_alpha=0.
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model == 'clam_sb':
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'clam_mb':
        args.cls_alpha= .7
        args.cl_alpha = .3
    elif args.model == 'dsmil':
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5

    if args.datasets == 'camelyon16':
        args.fix_loader_random = True
        args.fix_train_random = True

    if args.datasets == 'tcga':
        args.num_workers = 0
        args.always_test = True

    if args.wandb:
        if args.auto_resume:
            ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
            wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path),id=ckp['wandb_id'],resume='must')
        else:
            wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path))
        
    print(args)
    logging.info(args)

    localtime = time.asctime( time.localtime(time.time()) )
    print(localtime)
    logging.info(localtime)
    main(args=args,cfg=config_conch)




# import time
# import torch
# import wandb
# import numpy as np
# from copy import deepcopy
# import torch.nn as nn
# from dataloader import *
# from torch.utils.data import DataLoader, RandomSampler
# import argparse, os
# # from modules import attmil,clam,mhim,dsmil,transmil,mean_max,diffmil,wikg
# from modules import attmil,clam,mhim,dsmil,transmil,mean_max,diffmil
# from modules import diffusionnet as diffusionnet
# from torch.nn.functional import one_hot
# from torch.cuda.amp import GradScaler
# from contextlib import suppress
# import time

# from timm.utils import AverageMeter,dispatch_clip_grad
# from timm.models import  model_parameters
# from collections import OrderedDict

# from utils import *

# def main(args):
#     # set seed
#     seed_torch(args.seed)

#     # --->get dataset
#     if args.datasets.lower() == 'camelyon16':
#         label_path=os.path.join(args.dataset_root,'label.csv')
#         p, l = get_patient_label(label_path)
#         index = [i for i in range(len(p))]
#         random.shuffle(index)
#         p = p[index]
#         l = l[index]

#     elif args.datasets.lower() == 'tcga':
#         label_path=os.path.join(args.dataset_root,'label.csv')
#         p, l = get_patient_label(label_path)
#         index = [i for i in range(len(p))]
#         random.shuffle(index)
#         p = p[index]
#         l = l[index]

#     elif args.datasets.lower() == 'bracs':
#         label_path=os.path.join(args.dataset_root,'label.csv')
#         if not os.path.exists(label_path):
#             label_path=os.path.join(args.dataset_root,'labels.csv')
#         p, l, d = get_patient_label_bracs(label_path)
#         index = [i for i in range(len(p))]
#         random.shuffle(index)
#         p = p[index]
#         l = l[index]
#         d = d[index]
#         if args.cv_fold == 1:
#             train_p,train_l,test_p,test_l,val_p,val_l = [],[],[],[],[],[]
#             for i in range(len(p)):
#                 if 'Testing' in d[i]:
#                     test_p.extend([p[i]])
#                     test_l.extend([l[i]])
#                 elif 'Validation' in d[i]:
#                     val_p.extend([p[i]])
#                     val_l.extend([l[i]])
#                 else:
#                     #print(p[i])
#                     train_p.extend([p[i]])
#                     train_l.extend([l[i]])
#             train_p,train_l,test_p,test_l,val_p,val_l = np.array(train_p).reshape(1,-1),np.array(train_l).reshape(1,-1),np.array(test_p).reshape(1,-1),np.array(test_l).reshape(1,-1),np.array(val_p).reshape(1,-1),np.array(val_l).reshape(1,-1)
#     if args.cv_fold > 1:
#         train_p, train_l, test_p, test_l,val_p,val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)

#     acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
#     ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]

#     if not args.no_log:
#         print('Dataset: ' + args.datasets)

#     # resume
#     if args.auto_resume and not args.no_log:
#         ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
#         args.fold_start = ckp['k']
#         if len(ckp['ckc_metric']) == 6:
#             acs, pre, rec,fs,auc,te_auc = ckp['ckc_metric']
#         elif len(ckp['ckc_metric']) == 7:
#             acs, pre, rec,fs,auc,te_auc,te_fs = ckp['ckc_metric']
#         else:
#             acs, pre, rec,fs,auc = ckp['ckc_metric']

#     for k in range(args.fold_start, args.cv_fold):
#         if not args.no_log:
#             print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
#         ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)

#     if args.always_test:
#         if args.wandb:
#             wandb.log({
#                 "cross_val/te_auc_mean":np.mean(np.array(te_auc)),
#                 "cross_val/te_auc_std":np.std(np.array(te_auc)),
#                 "cross_val/te_f1_mean":np.mean(np.array(te_fs)),
#                 "cross_val/te_f1_std":np.std(np.array(te_fs)),
#             })

#     if args.wandb:
#         wandb.log({
#             "cross_val/acc_mean":np.mean(np.array(acs)),
#             "cross_val/auc_mean":np.mean(np.array(auc)),
#             "cross_val/f1_mean":np.mean(np.array(fs)),
#             "cross_val/pre_mean":np.mean(np.array(pre)),
#             "cross_val/recall_mean":np.mean(np.array(rec)),
#             "cross_val/acc_std":np.std(np.array(acs)),
#             "cross_val/auc_std":np.std(np.array(auc)),
#             "cross_val/f1_std":np.std(np.array(fs)),
#             "cross_val/pre_std":np.std(np.array(pre)),
#             "cross_val/recall_std":np.std(np.array(rec)),
#         })
#     if not args.no_log:
#         print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
#         print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
#         print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
#         print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
#         print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))

# def one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l):
#     # --->initiation
#     seed_torch(args.seed)
#     loss_scaler = GradScaler() if args.amp else None
#     amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric

#     # --->load data
#     if args.datasets.lower() == 'camelyon16':

#         train_set = C16Dataset(train_p[k],train_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
#         test_set = C16Dataset(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
#         if args.val_ratio != 0.:
#             val_set = C16Dataset(val_p[k],val_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
#         else:
#             val_set = test_set

#     elif args.datasets.lower() == 'tcga':
        
#         train_set = TCGADataset(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True,_type=args.tcga_sub)
#         test_set = TCGADataset(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
#         if args.val_ratio != 0.:
#             val_set = TCGADataset(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,_type=args.tcga_sub)
#         else:
#             val_set = test_set

#     elif args.datasets.lower() == 'bracs':  
#         train_set = BRACSDataset(train_p[k],train_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,n_class=args.n_classes)
#         test_set = BRACSDataset(test_p[k],test_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,n_class=args.n_classes)
#         if args.val_ratio != 0.:
#             val_set = BRACSDataset(val_p[k],val_l[k],args.tcga_max_patch,args.dataset_root,persistence=args.persistence,n_class=args.n_classes)
#         else:
#             val_set = test_set

#     if args.fix_loader_random:
#         # generated by int(torch.empty((), dtype=torch.int64).random_().item())
#         big_seed_list = 7784414403328510413
#         generator = torch.Generator()
#         generator.manual_seed(big_seed_list)  
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
#     else:
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)

#     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
#     test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

#     mm_sche = None
#     if not args.teacher_init.endswith('.pt'):
#         _str = 'fold_{fold}_model_best_auc.pt'.format(fold=k)
#         _teacher_init = os.path.join(args.teacher_init,_str)
#     else:
#         _teacher_init =args.teacher_init

#     # --->bulid networks
#     if args.model == 'mhim':
#         if args.mrh_sche:
#             mrh_sche = cosine_scheduler(args.mask_ratio_h,0.,epochs=args.num_epoch,niter_per_ep=len(train_loader))
#         else:
#             mrh_sche = None

#         model_params = {
#             'baseline': args.baseline,
#             'dropout': args.dropout,
#             'mask_ratio' : args.mask_ratio,
#             'n_classes': args.n_classes,
#             'temp_t': args.temp_t,
#             'act': args.act,
#             'head': args.n_heads,
#             'msa_fusion': args.msa_fusion,
#             'mask_ratio_h': args.mask_ratio_h,
#             'mask_ratio_hr': args.mask_ratio_hr,
#             'mask_ratio_l': args.mask_ratio_l,
#             'mrh_sche': mrh_sche,
#             'da_act': args.da_act,
#             'attn_layer': args.attn_layer,
#         }
        
#         if args.mm_sche:
#             mm_sche = cosine_scheduler(args.mm,args.mm_final,epochs=args.num_epoch,niter_per_ep=len(train_loader),start_warmup_value=1.)

#         model = mhim.MHIM(**model_params).to(device)
#     elif args.model == 'diff':
#         model = diffmil.DAttentionWithDiff(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)
#     # elif args.model == 'wikg':
#     #     model = wikg.WiKG(dim_in=1024, dim_hidden=512, topk=args.wikg_topk, n_classes=2, agg_type='bi-interaction', dropout=0.3, pool='attn').to(device)
#     elif args.model == 'difftune':
#         model = diffmil.DAttentionWithDiffTune(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)
#     elif args.model == 'diffSim':
#         model = diffmil.DAttentionWithDiffEnd(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain,a_ratio=args.a_ratio,adapter_ratio=args.adapter_ratio).to(device)
#     elif args.model == 'diff2End':
#         model = diffmil.DAttentionWithDiffTwoEnd(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)
#     elif args.model == 'diffCon':
#         model = diffmil.DAttentionWithDiffContrast(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)
#     elif args.model == 'diffusionnet':
#         model = diffusionnet.DiffusionNet(out_dim=args.n_classes,t=args.t_steps).to(device) 
#     elif args.model == 'random':
#         model = diffmil.DAttentionWithRandomAbandon(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain).to(device)           
#     elif args.model == 'chose':
#         model = diffmil. DAttentionWithDiffchose(out_dim=args.n_classes,k_ratio=args.k_ratio,t_steps=args.t_steps,ifrand=args.ifrand,ifTrain=args.ifTrain,ifType=args.ifType,ifClose=args.ifClose).to(device)           
#     elif args.model == 'pure':
#         model = mhim.MHIM(select_mask=False,n_classes=args.n_classes,act=args.act,head=args.n_heads,da_act=args.da_act,baseline=args.baseline).to(device)
#     elif args.model == 'attmil':
#         model = attmil.DAttention(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
#     elif args.model == 'gattmil':
#         model = attmil.AttentionGated(dropout=args.dropout).to(device)
#     # follow the official code
#     # ref: https://github.com/mahmoodlab/CLAM
#     elif args.model == 'clam_sb':
#         model = clam.CLAM_SB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
#     elif args.model == 'clam_mb':
#         model = clam.CLAM_MB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
#     elif args.model == 'transmil':
#         model = transmil.TransMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
#     elif args.model == 'transmilDiff':
#         model = transmil.TransMILwithDiff(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
#     elif args.model == 'dsmil':
#         model = dsmil.MILNet(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
#         args.cls_alpha = 0.5
#         args.cl_alpha = 0.5
#         state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
#         info = model.load_state_dict(state_dict_weights, strict=False)
#         if not args.no_log:
#             print(info)
#     elif args.model == 'meanmil':
#         model = mean_max.MeanMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
#     elif args.model == 'maxmil':
#         model = mean_max.MaxMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)

#     if args.init_stu_type != 'none':
#         if not args.no_log:
#             print('######### Model Initializing.....')
#         pre_dict = torch.load(_teacher_init)
#         new_state_dict ={}
#         if args.init_stu_type == 'fc':
#         # only patch_to_emb
#             for _k,v in pre_dict.items():
#                 _k = _k.replace('patch_to_emb.','') if 'patch_to_emb' in _k else _k
#                 new_state_dict[_k]=v
#             info = model.patch_to_emb.load_state_dict(new_state_dict,strict=False)
#         else:
#         # init all
#             info = model.load_state_dict(pre_dict,strict=False)
#         if not args.no_log:
#             print(info)

#     # teacher model
#     if args.model == 'mhim':
#         model_tea = deepcopy(model)
#         if not args.no_tea_init and args.tea_type != 'same':
#             if not args.no_log:
#                 print('######### Teacher Initializing.....')
#             try:
#                 pre_dict = torch.load(_teacher_init)
#                 info = model_tea.load_state_dict(pre_dict,strict=False)
#                 if not args.no_log:
#                     print(info)
#             except:
#                 if not args.no_log:
#                     print('########## Init Error')
#         if args.tea_type == 'same':
#             model_tea = model
#     else:
#         model_tea = None

#     if args.loss == 'bce':
#         criterion = nn.BCEWithLogitsLoss()
#     elif args.loss == 'ce':
#         criterion = nn.CrossEntropyLoss()

#     # optimizer
#     if args.opt == 'adamw':
#         optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
#     elif args.opt == 'adam':
#         optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

#     if args.lr_sche == 'cosine':
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
#     elif args.lr_sche == 'step':
#         assert not args.lr_supi
#         # follow the DTFD-MIL
#         # ref:https://github.com/hrzhang1123/DTFD-MIL
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
#     elif args.lr_sche == 'const':
#         scheduler = None

#     if args.early_stopping:
#         early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
#     else:
#         early_stopping = None

#     optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_thr,opt_epoch = 0, 0, 0, 0,0,0,0
#     opt_te_auc,opt_tea_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs  = 0., 0., 0., 0., 0.
#     epoch_start = 0

#     if args.fix_train_random:
#         seed_torch(args.seed)

#     # resume
#     if args.auto_resume and not args.no_log:
#         ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
#         epoch_start = ckp['epoch']
#         model.load_state_dict(ckp['model'])
#         optimizer.load_state_dict(ckp['optimizer'])
#         scheduler.load_state_dict(ckp['lr_sche'])
#         early_stopping.load_state_dict(ckp['early_stop'])
#         optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = ckp['val_best_metric']
#         opt_te_auc = ckp['te_best_metric'][0]
#         if len(ckp['te_best_metric']) > 1:
#             opt_te_fs = ckp['te_best_metric'][1]
#         opt_te_tea_auc,opt_te_tea_fs = ckp['te_best_metric'][2:4]
#         np.random.set_state(ckp['random']['np'])
#         torch.random.set_rng_state(ckp['random']['torch'])
#         random.setstate(ckp['random']['py'])
#         if args.fix_loader_random:
#             train_loader.sampler.generator.set_state(ckp['random']['loader'])
#         args.auto_resume = False

#     train_time_meter = AverageMeter()

#     for epoch in range(epoch_start, args.num_epoch):
#         train_loss,start,end = train_loop(args,model,model_tea,train_loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch)
#         train_time_meter.update(end-start)
#         stop,accuracy, auc_value, precision, recall, fscore, test_loss, threshold_optimal = val_loop(args,model,val_loader,device,criterion,early_stopping,epoch,model_tea)

#         if model_tea is not None:
#             _,accuracy_tea, auc_value_tea, precision_tea, recall_tea, fscore_tea, test_loss_tea = val_loop(args,model_tea,val_loader,device,criterion,None,epoch,model_tea)
#             if args.wandb:
#                 rowd = OrderedDict([
#                     ("val_acc_tea",accuracy_tea),
#                     ("val_precision_tea",precision_tea),
#                     ("val_recall_tea",recall_tea),
#                     ("val_fscore_tea",fscore_tea),
#                     ("val_auc_tea",auc_value_tea),
#                     ("val_loss_tea",test_loss_tea),
#                 ])

#                 rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
#                 wandb.log(rowd)

#             if auc_value_tea > opt_tea_auc:
#                 opt_tea_auc = auc_value_tea
#                 if args.wandb:
#                     rowd = OrderedDict([
#                         ("best_tea_auc",opt_tea_auc)
#                     ])
#                     rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
#                     wandb.log(rowd)

#         if args.always_test:

#             _te_accuracy, _te_auc_value, _te_precision, _te_recall, _te_fscore,_te_test_loss_log = test(args,model,test_loader,device,criterion,model_tea)
            
#             if args.wandb:
#                 rowd = OrderedDict([
#                     ("te_acc",_te_accuracy),
#                     ("te_precision",_te_precision),
#                     ("te_recall",_te_recall),
#                     ("te_fscore",_te_fscore),
#                     ("te_auc",_te_auc_value),
#                     ("te_loss",_te_test_loss_log),
#                 ])

#                 rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
#                 wandb.log(rowd)

#             if _te_auc_value > opt_te_auc:
#                 opt_te_auc = _te_auc_value
#                 opt_te_fs = _te_fscore
#                 if args.wandb:
#                     rowd = OrderedDict([
#                         ("best_te_auc",opt_te_auc),
#                         ("best_te_f1",_te_fscore)
#                     ])
#                     rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
#                     wandb.log(rowd)
            
#             if model_tea is not None:
#                 _te_tea_accuracy, _te_tea_auc_value, _te_tea_precision, _te_tea_recall, _te_tea_fscore,_te_tea_test_loss_log = test(args,model_tea,test_loader,device,criterion,model_tea)
            
#                 if args.wandb:
#                     rowd = OrderedDict([
#                         ("te_tea_acc",_te_tea_accuracy),
#                         ("te_tea_precision",_te_tea_precision),
#                         ("te_tea_recall",_te_tea_recall),
#                         ("te_tea_fscore",_te_tea_fscore),
#                         ("te_tea_auc",_te_tea_auc_value),
#                         ("te_tea_loss",_te_tea_test_loss_log),
#                     ])

#                     rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
#                     wandb.log(rowd)

#                 if _te_tea_auc_value > opt_te_tea_auc:
#                     opt_te_tea_auc = _te_tea_auc_value
#                     opt_te_tea_fs = _te_tea_fscore
#                     if args.wandb:
#                         rowd = OrderedDict([
#                             ("best_te_tea_auc",opt_te_tea_auc),
#                             ("best_te_tea_f1",_te_fscore)
#                         ])
#                         rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
#                         wandb.log(rowd)
#         if not args.no_log:
#             print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f , time: %.3f(%.3f)' % 
#         (epoch+1, args.num_epoch, train_loss, test_loss, accuracy, auc_value, precision, recall, fscore, train_time_meter.val,train_time_meter.avg))

#         if args.wandb:
#             rowd = OrderedDict([
#                 ("val_acc",accuracy),
#                 ("val_precision",precision),
#                 ("val_recall",recall),
#                 ("val_fscore",fscore),
#                 ("val_auc",auc_value),
#                 ("val_loss",test_loss),
#                 ("epoch",epoch),
#             ])

#             rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
#             wandb.log(rowd)

#         if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.num_epoch:
#             optimal_ac = accuracy
#             opt_pre = precision
#             opt_re = recall
#             opt_fs = fscore
#             opt_auc = auc_value
#             opt_thr = threshold_optimal
#             opt_epoch = epoch

#             if not os.path.exists(args.model_path):
#                 os.mkdir(args.model_path)
#             if not args.no_log:
#                 best_pt = {
#                     'model': model.state_dict(),
#                     'teacher': model_tea.state_dict() if model_tea is not None else None,
#                 }
#                 torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
#         if args.wandb:
#             rowd = OrderedDict([
#                 ("val_best_acc",optimal_ac),
#                 ("val_best_precesion",opt_pre),
#                 ("val_best_recall",opt_re),
#                 ("val_best_fscore",opt_fs),
#                 ("val_best_auc",opt_auc),
#                 ("val_best_epoch",opt_epoch),
#             ])

#             rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
#             wandb.log(rowd)
        
#         # save checkpoint
#         random_state = {
#             'np': np.random.get_state(),
#             'torch': torch.random.get_rng_state(),
#             'py': random.getstate(),
#             'loader': train_loader.sampler.generator.get_state() if args.fix_loader_random else '',
#         }
#         ckp = {
#             'model': model.state_dict(),
#             'lr_sche': scheduler.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'epoch': epoch+1,
#             'k': k,
#             'early_stop': early_stopping.state_dict(),
#             'random': random_state,
#             'ckc_metric': [acs,pre,rec,fs,auc,te_auc,te_fs],
#             'val_best_metric': [optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch],
#             'te_best_metric': [opt_te_auc,opt_te_fs,opt_te_tea_auc,opt_te_tea_fs],
#             'wandb_id': wandb.run.id if args.wandb else '',
#         }
#         if not args.no_log:
#             torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))

#         if stop:
#             break
    
#     # test
#     if not args.no_log:
#         best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
#         info = model.load_state_dict(best_std['model'])
#         print(info)
#         if model_tea is not None and best_std['teacher'] is not None:
#             info = model_tea.load_state_dict(best_std['teacher'])
#             print(info)

#     accuracy, auc_value, precision, recall, fscore,test_loss_log = test(args,model,test_loader,device,criterion,model_tea,opt_thr)
    
#     if args.wandb:
#         wandb.log({
#             "test_acc":accuracy,
#             "test_precesion":precision,
#             "test_recall":recall,
#             "test_fscore":fscore,
#             "test_auc":auc_value,
#             "test_loss":test_loss_log,
#         })
#     if not args.no_log:
#         print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
#     acs.append(accuracy)
#     pre.append(precision)
#     rec.append(recall)
#     fs.append(fscore)
#     auc.append(auc_value)

#     if args.always_test:
#         te_auc.append(opt_te_auc)
#         te_fs.append(opt_te_fs)
        
#     return [acs,pre,rec,fs,auc,te_auc,te_fs]

# def train_loop(args,model,model_tea,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch):
#     start = time.time()
#     loss_cls_meter = AverageMeter()
#     loss_cl_meter = AverageMeter()
#     loss_cl_p_meter = AverageMeter()
#     loss_cl_n_meter = AverageMeter()
#     patch_num_meter = AverageMeter()
#     keep_num_meter = AverageMeter()
#     mm_meter = AverageMeter()
#     train_loss_log = 0.
#     model.train()
#     if model_tea is not None:
#         model_tea.train()

#     for i, data in enumerate(loader):
#         optimizer.zero_grad()

#         if isinstance(data[0],(list,tuple)):
#             for i in range(len(data[0])):
#                 data[0][i] = data[0][i].to(device)
#             bag=data[0]
#             batch_size=data[0][0].size(0)
#         else:
#             bag=data[0].to(device)  # b*n*1024
#             batch_size=bag.size(0)
            
#         label=data[1].to(device)
#         label_negative = torch.zeros_like(label)
#         label_positive = torch.ones_like(label)

#         with amp_autocast():
#             if args.patch_shuffle:
#                 bag = patch_shuffle(bag,args.shuffle_group)
#             elif args.group_shuffle:
#                 bag = group_shuffle(bag,args.shuffle_group)

#             if args.model == 'mhim':
#                 if model_tea is not None:
#                     cls_tea,attn = model_tea.forward_teacher(bag,return_attn=True)
#                 else:
#                     attn,cls_tea = None,None

#                 cls_tea = None if args.cl_alpha == 0. else cls_tea

#                 train_logits, cls_loss,patch_num,keep_num = model(bag,attn,cls_tea,i=epoch*len(loader)+i)

#             elif args.model == 'pure':
#                 train_logits, cls_loss,patch_num,keep_num = model.pure(bag)
#             elif args.model in ('clam_sb','clam_mb','dsmil'):
#                 train_logits,cls_loss,patch_num = model(bag,label,criterion)
#                 keep_num = patch_num
#             elif args.model== 'diffusionnet': #shz
#                 train_logits = model(bag)
#                 cls_loss,patch_num,keep_num = 0.,0.,0.
#             # elif args.model== 'diff2End': #shz diff2End
#             #     train_logits,cls_logits_p,cls_logits_n = model(bag)
#             #     if args.loss == 'ce':
#             #         cls_loss_n = criterion(cls_logits_n.view(batch_size,-1),label_negative)
#             #         cls_loss_p = criterion(cls_logits_p.view(batch_size,-1),label_positive)
#             #     elif args.loss == 'bce':
#             #         cls_loss_n = criterion(cls_logits_n.view(batch_size,-1),one_hot(label_negative.view(batch_size,-1).float(),num_classes=2))
#             #         cls_loss_p = criterion(cls_logits_p.view(batch_size,-1),one_hot(label_positive.view(batch_size,-1).float(),num_classes=2))
#             #     patch_num,keep_num,cls_loss = 0.,0.,0.
#             elif args.model== 'diffSim' or args.model=='transmilDiff': 
            
#                 train_logits,cls_logits = model(bag)
#                 if args.loss == 'ce':
#                     cls_loss = criterion(cls_logits.view(batch_size,-1),label_negative)
#                 elif args.loss == 'bce':
#                     cls_loss = criterion(cls_logits.view(batch_size,-1),one_hot(label_negative.view(batch_size,-1).float(),num_classes=2))
#                 patch_num,keep_num = 0.,0.
                
                

#             else:
#                 train_logits = model(bag)
#                 cls_loss,patch_num,keep_num = 0.,0.,0.

#             if args.loss == 'ce':
#                 logit_loss = criterion(train_logits.view(batch_size,-1),label)
                
#             elif args.loss == 'bce':
#                 logit_loss = criterion(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1).float(),num_classes=2))

#         #train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha

#         #train_loss = args.cls_alpha * logit_loss +  cls_loss * 1   #shz
#         if args.model== 'diff2End':
#             train_loss = args.cls_alpha * logit_loss +  cls_loss_p * 1 + cls_loss_n * 1  #shz
#         else:
#             train_loss = args.cls_alpha * logit_loss +  cls_loss * 1   #shz
#         train_loss = train_loss / args.accumulation_steps
#         if args.clip_grad > 0.:
#             dispatch_clip_grad(
#                 model_parameters(model),
#                 value=args.clip_grad, mode='norm')

#         if (i+1) % args.accumulation_steps == 0:
#             train_loss.backward()
#             optimizer.step()
#             if args.lr_supi and scheduler is not None:
#                 scheduler.step()
#             if args.model == 'mhim':
#                 if mm_sche is not None:
#                     mm = mm_sche[epoch*len(loader)+i]
#                 else:
#                     mm = args.mm
#                 if model_tea is not None:
#                     if args.tea_type == 'same':
#                         pass
#                     else:
#                         ema_update(model,model_tea,mm)
#             else:
#                 mm = 0.
#         # # for name, parms in model.named_parameters():
#         # #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
#         # for name, parms in model.named_parameters():
#         #     if parms.grad is not None:
#         #         #print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
#         #         a=1
#         #     else:
#         #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', "No Gradient")

#         # # print(model)
#         # print("__________________________*****___________________________________*******__________________") #shz
#         loss_cls_meter.update(logit_loss,1)
#         loss_cl_meter.update(cls_loss,1)
#         #loss_cl_p_meter.update(cls_loss_p,1)
#         #loss_cl_n_meter.update(cls_loss_n,1)
#         patch_num_meter.update(patch_num,1)
#         keep_num_meter.update(keep_num,1)
#         mm_meter.update(mm,1)

#         if i % args.log_iter == 0 or i == len(loader)-1:
#             lrl = [param_group['lr'] for param_group in optimizer.param_groups]
#             lr = sum(lrl) / len(lrl)
#             rowd = OrderedDict([
#                 ('cls_loss',loss_cls_meter.avg),
#                 ('lr',lr),
#                 ('cl_loss',loss_cl_meter.avg),
#                 ('cl_loss_p',loss_cl_p_meter.avg),
#                 ('cl_loss_n',loss_cl_n_meter.avg),
#                 ('patch_num',patch_num_meter.avg),
#                 ('keep_num',keep_num_meter.avg),
#                 ('mm',mm_meter.avg),
#             ])
#             if not args.no_log:
#                 print('[{}/{}] logit_loss:{}, cls_loss:{},patch_num:{}, keep_num:{} '.format(i,len(loader)-1,loss_cls_meter.avg,loss_cl_meter.avg,patch_num_meter.avg, keep_num_meter.avg))
#             rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
#             if args.wandb:
#                 wandb.log(rowd)

#         train_loss_log = train_loss_log + train_loss.item()

#     end = time.time()
#     train_loss_log = train_loss_log/len(loader)
#     if not args.lr_supi and scheduler is not None:
#         scheduler.step()
    
#     return train_loss_log,start,end

# def val_loop(args,model,loader,device,criterion,early_stopping,epoch,model_tea=None):
#     if model_tea is not None:
#         model_tea.eval()
#     model.eval()
#     loss_cls_meter = AverageMeter()
#     bag_logit, bag_labels=[], []

#     with torch.no_grad():
#         for i, data in enumerate(loader):
#             if len(data[1]) > 1:
#                 bag_labels.extend(data[1].tolist())
#             else:
#                 bag_labels.append(data[1].item())

#             if isinstance(data[0],(list,tuple)):
#                 for i in range(len(data[0])):
#                     data[0][i] = data[0][i].to(device)
#                 bag=data[0]
#                 batch_size=data[0][0].size(0)
#             else:
#                 bag=data[0].to(device)  # b*n*1024
#                 batch_size=bag.size(0)

#             label=data[1].to(device)
#             if args.model in ('mhim','pure'):
#                 test_logits = model.forward_test(bag)
#             elif args.model == 'dsmil':
#                 test_logits,_ = model(bag)
#             elif args.model == 'diffSim' or args.model=='transmilDiff':
#             #elif args.model in ('diffSim','diff2End'):
#                 test_logits,_ = model(bag)
#             elif args.model == 'diff2End':
#             #elif args.model in ('diffSim','diff2End'):
#                 test_logits,_,_ = model(bag)
#             else:
#                 test_logits = model(bag)

#             if args.loss == 'ce':
#                 if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
#                     test_loss = criterion(test_logits[0].view(batch_size,-1),label)
#                     bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
#                 else:
#                     test_loss = criterion(test_logits.view(batch_size,-1),label)
#                     if args.n_classes<=2:
#                         if batch_size > 1:
#                             bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
#                         else:
#                             bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
#                     else:
#                         if batch_size > 1:
#                             bag_logit.extend(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
#                         else:
#                             bag_logit.append(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
                    
#             elif args.loss == 'bce':
#                 if args.model == 'dsmil' and args.ds_average:
#                     test_loss = criterion(test_logits.view(batch_size,-1),label)
#                     bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
#                 else:
#                     test_loss = criterion(test_logits[0].view(batch_size,-1),label.view(batch_size,-1).float())
                    
#                     bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

#             loss_cls_meter.update(test_loss,1)

#     # save the log file
#     # accuracy, auc_value, precision, recall, fscore, threshold_optimal = five_scores(bag_labels, bag_logit) #five_scores_new
#     accuracy, auc_value, precision, recall, fscore, threshold_optimal = five_scores_new(bag_labels, bag_logit,threshold_optimal=None,n_classes=args.n_classes) #shz
#     # early stop
#     if early_stopping is not None:
#         early_stopping(epoch,-auc_value,model)
#         stop = early_stopping.early_stop
#     else:
#         stop = False
#     return stop,accuracy, auc_value, precision, recall, fscore,loss_cls_meter.avg, threshold_optimal

# def test(args,model,loader,device,criterion,model_tea=None,opt_thr=None):
#     if model_tea is not None:
#         model_tea.eval()
#     model.eval()
#     test_loss_log = 0.
#     bag_logit, bag_labels=[], []

#     with torch.no_grad():
#         for i, data in enumerate(loader):
#             if len(data[1]) > 1:
#                 bag_labels.extend(data[1].tolist())
#             else:
#                 bag_labels.append(data[1].item())
                
#             if isinstance(data[0],(list,tuple)):
#                 for i in range(len(data[0])):
#                     data[0][i] = data[0][i].to(device)
#                 bag=data[0]
#                 batch_size=data[0][0].size(0)
#             else:
#                 bag=data[0].to(device)  # b*n*1024
#                 batch_size=bag.size(0)

#             label=data[1].to(device)
#             if args.model in ('mhim','pure'):
#                 test_logits = model.forward_test(bag)
#             elif args.model == 'dsmil':
#                 test_logits,_ = model(bag)
#             elif args.model == 'diffSim' or args.model=='transmilDiff':
#             # elif args.model in ('diffSim','diff2End'):
#                 test_logits,_ = model(bag)
#             elif args.model == 'diff2End':
#             # elif args.model in ('diffSim','diff2End'):
#                 test_logits,_,_ = model(bag)
#             else:
#                 test_logits = model(bag)

#             if args.loss == 'ce':
#                 if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
#                     test_loss = criterion(test_logits[0].view(batch_size,-1),label)
#                     bag_logit.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
#                 else:
#                     test_loss = criterion(test_logits.view(batch_size,-1),label)
#                     if args.n_classes<=2:
#                         if batch_size > 1:
#                             bag_logit.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
#                         else:
#                             bag_logit.append(torch.softmax(test_logits,dim=-1)[:,1].cpu().squeeze().numpy())
#                     else:
#                         if batch_size > 1:
#                             bag_logit.extend(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
#                         else:
#                             bag_logit.append(torch.softmax(test_logits,dim=-1).cpu().squeeze().numpy())
#             elif args.loss == 'bce':
#                 if args.model == 'dsmil' and args.ds_average:
#                     test_loss = criterion(test_logits[0].view(batch_size,-1),label)
#                     bag_logit.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
#                 else:
#                     test_loss = criterion(test_logits.view(batch_size,-1),label.view(1,-1).float())
#                 bag_logit.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

#             test_loss_log = test_loss_log + test_loss.item()
    
#     # save the log file
#     # cal the best thr with val set
#     opt_thr = opt_thr if args.best_thr_val else None
#     #ccuracy, auc_value, precision, recall, fscore, _ = five_scores(bag_labels, bag_logit,threshold_optimal=opt_thr)
#     accuracy, auc_value, precision, recall, fscore, threshold_optimal = five_scores_new(bag_labels, bag_logit,threshold_optimal=None,n_classes=args.n_classes) #shz
#     test_loss_log = test_loss_log/len(loader)

#     return accuracy, auc_value, precision, recall, fscore,test_loss_log

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='MIL Training Script')

#     # Dataset 
#     parser.add_argument('--datasets', default='camelyon16', type=str, help='[camelyon16, tcga]')
#     parser.add_argument('--dataset_root', default='/data/xxx/TCGA', type=str, help='Dataset root path')
#     parser.add_argument('--tcga_max_patch', default=-1, type=int, help='Max Number of patch in TCGA [-1]')
#     parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
#     parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
#     parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
#     parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
#     parser.add_argument('--cv_fold', default=3, type=int, help='Number of cross validation fold [3]')
#     parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
#     parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
#     parser.add_argument('--tcga_sub', default='nsclc', type=str, help='[nsclc,brca]')

#     # Train
#     parser.add_argument('--cls_alpha', default=1.0, type=float, help='Main loss alpha')
#     parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
#     parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
#     parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
#     parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
#     parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
#     parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
#     parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
#     parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
#     parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
#     parser.add_argument('--model', default='mhim', type=str, help='Model name')
#     parser.add_argument('--seed', default=2021, type=int, help='random number [2021]' )
#     parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate [0.0002]')
#     parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
#     parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
#     parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
#     parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
#     parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')
#     parser.add_argument('--always_test', action='store_true', help='Test model in the training phase')
#     parser.add_argument('--best_thr_val', action='store_true', help='Cal the best thr with val set in the test phase. Thanks Weiyi Wu!')

#     # Model
#     # diffusion shz
#     parser.add_argument('--ifType', default=1, type=int, help='to chose the method of chosing arch')
#     parser.add_argument('--k_ratio', default=0.1, type=float, help='Number of total k ratio')
#     parser.add_argument('--t_steps', default=2, type=int, help='t in diffusion model')
#     parser.add_argument('--ifTrain', default=1, type=int, help='if 0 means train and test use the same method,1 means different')
#     parser.add_argument('--ifrand', default=0, type=int, help='if 0 means using diff,1 means using rand')
#     parser.add_argument('--temp_nums', default=100, type=int, help='the numbers of templates made by diffusion model')
#     parser.add_argument('--ifEma', default=0, type=int, help='if 0 means no Ema in the sharing weights')
#     parser.add_argument('--ifClose', default=0, type=int, help='if 0 means far, if 1 means near')
#     parser.add_argument('--adapter_ratio', default=1.0, type=float, help='adapter ratio')
#     parser.add_argument('--a_ratio', default=1.0, type=float, help='a ratio')
#     # wikg wikg_topk
#     parser.add_argument('--wikg_topk', default=6, type=int, help='no')

#     # Other models
#     parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
#     # Our
#     parser.add_argument('--baseline', default='selfattn', type=str, help='Baselin model [attn,selfattn]')
#     parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
#     parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
#     parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the MSA')
#     parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')

#     # Shuffle
#     parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
#     parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
#     parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

#     # MHIM
#     # Mask ratio
#     parser.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio')
#     parser.add_argument('--mask_ratio_l', default=0., type=float, help='Low attention mask ratio')
#     parser.add_argument('--mask_ratio_h', default=0., type=float, help='High attention mask ratio')
#     parser.add_argument('--mask_ratio_hr', default=1., type=float, help='Randomly high attention mask ratio')
#     parser.add_argument('--mrh_sche', action='store_true', help='Decay of HAM')
#     parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote]')
#     parser.add_argument('--attn_layer', default=0, type=int)
    
#     # Siamese framework
#     parser.add_argument('--cl_alpha', default=0., type=float, help='Auxiliary loss alpha')
#     parser.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
#     parser.add_argument('--teacher_init', default='none', type=str, help='Path to initial teacher model')
#     parser.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization')
#     parser.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none,fc,all]')
#     parser.add_argument('--tea_type', default='none', type=str, help='[none,same]')
#     parser.add_argument('--mm', default=0.9999, type=float, help='Ema decay [0.9997]')
#     parser.add_argument('--mm_final', default=1., type=float, help='Final ema decay [1.]')
#     parser.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay')

#     # Misc
#     parser.add_argument('--title', default='default', type=str, help='Title of exp')
#     parser.add_argument('--project', default='mil_new_c16', type=str, help='Project name of exp')
#     parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
#     parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
#     parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
#     parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
#     parser.add_argument('--no_log', action='store_true', help='Without log')
#     parser.add_argument('--model_path', type=str, help='Output path')

#     args = parser.parse_args()
    
#     if not os.path.exists(os.path.join(args.model_path,args.project)):
#         os.mkdir(os.path.join(args.model_path,args.project))
#     args.model_path = os.path.join(args.model_path,args.project,args.title)
#     if not os.path.exists(args.model_path):
#         os.mkdir(args.model_path)

#     if args.model == 'pure':
#         args.cl_alpha=0.
#     # follow the official code
#     # ref: https://github.com/mahmoodlab/CLAM
#     elif args.model == 'clam_sb':
#         args.cls_alpha= .7
#         args.cl_alpha = .3
#     elif args.model == 'clam_mb':
#         args.cls_alpha= .7
#         args.cl_alpha = .3
#     elif args.model == 'dsmil':
#         args.cls_alpha = 0.5
#         args.cl_alpha = 0.5

#     if args.datasets == 'camelyon16':
#         args.fix_loader_random = True
#         args.fix_train_random = True

#     if args.datasets == 'tcga':
#         args.num_workers = 0
#         args.always_test = True

#     if args.wandb:
#         if args.auto_resume:
#             ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
#             wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path),id=ckp['wandb_id'],resume='must')
#         else:
#             wandb.init(project=args.project, entity='dearcat',name=args.title,config=args,dir=os.path.join(args.model_path))
        
#     print(args)

#     localtime = time.asctime( time.localtime(time.time()) )
#     print(localtime)
#     main(args=args)

