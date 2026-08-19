import time
import wandb

from torch.nn.functional import one_hot
from timm.models import  model_parameters
from timm.utils import AverageMeter,dispatch_clip_grad
from collections import OrderedDict
from sksurv.metrics import concordance_index_censored

from utils import *

# 这里只做简单的survival和另外两个任务的区分，各个方法之间还是融合在一起
def build_engine(args):
    if args.datasets.lower().startswith('surv'):
        return surv_train_loop,surv_val_loop,surv_test
    else:
        return 1


############# Survival Prediction ###################
def surv_train_loop(args,model,model_tea,loader,optimizer,device,amp_autocast,criterion,loss_scaler,scheduler,k,mm_sche,epoch,criterion_ce):
    
    start = time.time()
    loss_cls_meter = AverageMeter()
    loss_cl_meter = AverageMeter()
    patch_num_meter = AverageMeter()
    keep_num_meter = AverageMeter()
    mm_meter = AverageMeter()
    train_loss_log = 0.
    model.train()
    if model_tea is not None:
        model_tea.train()

    for i, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(loader):
        optimizer.zero_grad()

        bag = data_WSI.to(device)
        label = data_Label.type(torch.LongTensor).to(device)
        label_0 = torch.zeros_like(label)
        label_1 = torch.ones_like(label)
        label_2 = label_1+label_1   
        label_3 = label_2+label_1
        label_4 = label_3+label_1
        label_5 = label_4+label_1

        data_Censorship = data_Censorship.type(torch.FloatTensor).to(device)

        logit_loss = None
        with amp_autocast():
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)

            if args.model == 'mhim':
                if model_tea is not None:
                    cls_tea,attn = model_tea.forward_teacher(bag)
                else:
                    attn,cls_tea = None,None

                cls_tea = None if args.cl_alpha == 0. else cls_tea
                if args.baseline == 'dsmil':
                    logits, cls_loss,patch_num,keep_num = model(bag,attn,cls_tea,i=epoch*len(loader)+i)
                    logit_loss = 0.5*criterion(hazards=torch.sigmoid(logits[1]), S=torch.cumprod(1 - torch.sigmoid(logits[1]), dim=1), Y=label, c=data_Censorship) + 0.5*criterion(hazards=torch.sigmoid(logits[0]), S=torch.cumprod(1 - torch.sigmoid(logits[0]), dim=1), Y=label, c=data_Censorship)
                else:
                    logits, cls_loss,patch_num,keep_num = model(bag,attn,cls_tea,i=epoch*len(loader)+i)
            elif args.model == 'pure':
                if args.baseline == 'dsmil':
                    logits, cls_loss,patch_num,keep_num = model.pure(bag)
                    logit_loss = 0.5*criterion(hazards=torch.sigmoid(logits[1]), S=torch.cumprod(1 - torch.sigmoid(logits[1]), dim=1), Y=label, c=data_Censorship) + 0.5*criterion(hazards=torch.sigmoid(logits[0]), S=torch.cumprod(1 - torch.sigmoid(logits[0]), dim=1), Y=label, c=data_Censorship)
                else:
                    logits, cls_loss,patch_num,keep_num = model.pure(bag)
            elif args.model in ('clam_sb','clam_mb','dsmil'):
                logits,cls_loss,patch_num = model(bag,label,criterion)
                keep_num = patch_num
            elif args.model in ('diffSur','diffSur2','diffSur2RRT','diffSur3'):
                result,a_result,logits = model(bag)
                patch_num,keep_num = 0.,0.
                #cls_loss = criterion_ce(a_result.view(1,-1),label_0)+criterion_ce(result.view(1,-1),label_1)
                cls_loss = criterion_ce(a_result.view(1,-1),label_0)
                # hazards_cls = torch.sigmoid(a_result)
                # S_cls = torch.cumprod(1 - hazards_cls, dim=1)
                # cls_loss = criterion(hazards=hazards_cls, S=S_cls, Y=label_0, c=data_Censorship)

            elif args.model =='diffSurOut':
                result,a_result,logits = model(bag)
                patch_num,keep_num = 0.,0.
                cls_loss = criterion_ce(a_result.view(1,-1),label_4)+criterion_ce(result.view(1,-1),label_5)
            elif args.model =='diffSimOut':
                logits,logits_2 = model(bag)
                patch_num,keep_num = 0.,0.
                cls_loss = criterion_ce(logits_2.view(1,-1),label_4)
            elif args.model=='acmil':
                sub_preds,logits,attn = model(bag)
                patch_num,keep_num = 0.,0.
                cls_loss = 0
            else:
                logits = model(bag)
                patch_num,keep_num = 0.,0.
                cls_loss = 0
            if logit_loss is None:
                # survival prediction
                hazards = torch.sigmoid(logits)
                S = torch.cumprod(1 - hazards, dim=1)
                logit_loss = criterion(hazards=hazards, S=S, Y=label, c=data_Censorship)
                #cls_loss = criterion(hazards=hazards_2, S=S_2, Y=label_0, c=data_Censorship)
                

        train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha #old
        # train_loss = args.cls_alpha * logit_loss +  cls_loss*args.cl_alpha #shz
         
        train_loss = train_loss / args.accumulation_steps
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        if (i+1) % args.accumulation_steps == 0:
            if loss_scaler is not None:
                loss_scaler.scale(train_loss).backward()
                loss_scaler.step(optimizer)
                loss_scaler.update()
            else:
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

        loss_cls_meter.update(logit_loss,1)
        loss_cl_meter.update(cls_loss,1)
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
                ('patch_num',patch_num_meter.avg),
                ('keep_num',keep_num_meter.avg),
                ('mm',mm_meter.avg),
            ])
            if not args.no_log:
                print('[{}/{}] logit_loss:{}, cls_loss:{},  patch_num:{}, keep_num:{} '.format(i,len(loader)-1,loss_cls_meter.avg,loss_cl_meter.avg,patch_num_meter.avg, keep_num_meter.avg))
            rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
            if args.wandb:
                wandb.log(rowd)

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    train_loss_log = train_loss_log/len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log,start,end

def surv_val_loop(args,model,loader,device,criterion,early_stopping,epoch,model_tea=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    loss_cls_meter = AverageMeter()
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    with torch.no_grad():
        for i, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(loader):
            bag = data_WSI.to(device)
            label = data_Label.type(torch.LongTensor).to(device)
            data_Censorship = data_Censorship.type(torch.FloatTensor).to(device)

            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
                if args.baseline == 'dsmil':
                    #test_logits = 0.5*test_logits[0][0] +  0.5*test_logits[0][1]
                    test_logits = test_logits[0][0]
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
                if args.ds_average:
                    test_logits = 0.5*test_logits[0]+0.5*test_logits[1]
            elif args.model in ('clam_mb','clam_sb'):
                test_logits = model(bag, instance_eval=False)
            elif args.model in ('diffSur','diffSur2','diffSurOut','diffSur2RRT','diffSur3'):
                _,_,test_logits = model(bag)
            elif args.model=='acmil':
                sub_preds,test_logits,attn = model(bag)
            else:
                test_logits = model(bag)

            # survival prediction
            hazards = torch.sigmoid(test_logits)
            S = torch.cumprod(1 - hazards, dim=1)
            test_loss = criterion(hazards=hazards, S=S, Y=label, c=data_Censorship)
            loss_cls_meter.update(test_loss,1)
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[i] = risk
            all_censorships[i] = data_Censorship.item()
            all_event_times[i] = data_Event
    
    # compute the c-index
    cindex = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    
    # early stop
    if early_stopping is not None:
        early_stopping(epoch,-cindex,model)
        stop = early_stopping.early_stop
    else:
        stop = False
    
    rowd = OrderedDict([
                ("cindex",cindex),
                ("loss",loss_cls_meter.avg),
            ])

    return stop,[cindex,cindex],rowd,loss_cls_meter.avg, 0

def surv_test(args,model,loader,device,criterion,model_tea=None,opt_thr=None):
    if model_tea is not None:
        model_tea.eval()
    model.eval()
    loss_cls_meter = AverageMeter()
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    with torch.no_grad():
        for i, (data_ID, data_WSI, data_Event, data_Censorship, data_Label) in enumerate(loader):
            bag = data_WSI.to(device)
            label = data_Label.type(torch.LongTensor).to(device)
            data_Censorship = data_Censorship.type(torch.FloatTensor).to(device)

            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
                if args.baseline == 'dsmil':
                    #test_logits = 0.5*test_logits[0][0] +  0.5*test_logits[0][1]
                    test_logits = test_logits[0][0]
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
                if args.ds_average:
                    test_logits = 0.5*test_logits[0]+0.5*test_logits[1]
            elif args.model in ('clam_mb','clam_sb'):
                test_logits = model(bag, instance_eval=False)
            elif args.model in ('diffSur','diffSur2','diffSurOut','diffSur2RRT','diffSur3'):
                _,_,test_logits = model(bag)
            elif args.model=='acmil':
                sub_preds,test_logits,attn = model(bag)
            else:
                test_logits = model(bag)

            # survival prediction
            hazards = torch.sigmoid(test_logits)
            S = torch.cumprod(1 - hazards, dim=1)
            test_loss = criterion(hazards=hazards, S=S, Y=label, c=data_Censorship)
            loss_cls_meter.update(test_loss,1)
            # results
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
            all_risk_scores[i] = risk
            all_censorships[i] = data_Censorship.item()
            all_event_times[i] = data_Event
    
    # compute the c-index
    cindex = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    rowd = OrderedDict([
                ("cindex",cindex),
                ("loss",loss_cls_meter.avg),
            ])

    return [cindex,cindex],rowd,loss_cls_meter.avg