import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score,balanced_accuracy_score
import torch
import os

def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

@torch.no_grad()
def ema_update(model,targ_model,mm=0.9999):
    r"""Performs a momentum update of the target network's weights.
    Args:
        mm (float): Momentum used in moving average update.
    """
    assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm

    for param_q, param_k in zip(model.parameters(), targ_model.parameters()):
        param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm) # mm*k +(1-mm)*q

def patch_shuffle(x,group=0,g_idx=None,return_g_idx=False):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))

    # padding
    H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
    if group > H or group<= 0:
        return group_shuffle(x,group)
    _n = -H % group
    H, W = H+_n, W+_n
    add_length = H * W - p
    # print(add_length)
    ps = torch.cat([ps,torch.tensor([-1 for i in range(add_length)])])
    # patchify
    ps = ps.reshape(shape=(group,H//group,group,W//group))
    ps = torch.einsum('hpwq->hwpq',ps)
    ps = ps.reshape(shape=(group**2,H//group,W//group))
    # shuffle
    if g_idx is None:
        g_idx = torch.randperm(ps.size(0))
    ps = ps[g_idx]
    # unpatchify
    ps = ps.reshape(shape=(group,group,H//group,W//group))
    ps = torch.einsum('hwpq->hpwq',ps)
    ps = ps.reshape(shape=(H,W))
    idx = ps[ps>=0].view(p)
    
    if return_g_idx:
        return x[:,idx.long()],g_idx
    else:
        return x[:,idx.long()]

def group_shuffle(x,group=0):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))
    if group > 0 and group < p:
        _pad = -p % group
        ps = torch.cat([ps,torch.tensor([-1 for i in range(_pad)])])
        ps = ps.view(group,-1)
        g_idx = torch.randperm(ps.size(0))
        ps = ps[g_idx]
        idx = ps[ps>=0].view(p)
    else:
        idx = torch.randperm(p)
    return x[:,idx.long()]


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

# def five_scores(bag_labels, bag_logits, bag_hat):
#     auc_value = roc_auc_score(bag_labels, bag_logits)
#     precision, recall, f1score, _ = precision_recall_fscore_support(bag_labels, bag_hat, average='binary')
#     accuracy = accuracy_score(bag_labels, bag_hat)
#     return accuracy, auc_value, precision, recall, f1score

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    labels = np.array(dataset.slide_label)
    label_uni = set(dataset.slide_label)
    weight_per_class = [N/len(labels[labels==c]) for c in label_uni]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.slide_label[idx]
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

def five_scores(bag_labels, bag_predictions,threshold_optimal=None):
    if threshold_optimal is None:
        fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = accuracy_score(bag_labels, bag_predictions)
    # accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore,threshold_optimal #shz: 原来是返回前五个 我现在加上了第六个


def five_scores_new(bag_labels, bag_predictions,threshold_optimal=None,n_classes=2):
    
    # threshold_optimal=0.5
    if n_classes<=2:
        if threshold_optimal is None:
            fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
            fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        auc_value = roc_auc_score(bag_labels, bag_predictions)
        this_class_label = np.array(bag_predictions)
        this_class_label[this_class_label>=threshold_optimal] = 1
        this_class_label[this_class_label<threshold_optimal] = 0
        bag_predictions = this_class_label
        precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
        accuracy = accuracy_score(bag_labels, bag_predictions)
    else:
        auc_value = roc_auc_score(bag_labels, bag_predictions,multi_class='ovr')
        #bag_predictions=[  for bag in bag_predictions]
        bag_predictions = np.argmax(bag_predictions, axis=1)
        precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='weighted')
        #accuracy = accuracy_score(bag_labels, bag_predictions)
        accuracy = balanced_accuracy_score(bag_labels, bag_predictions)
    # accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore,threshold_optimal


############# Survival Prediction ###################
def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)  # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float()  # censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1)  # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1)  # S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    # h[y] = h(1)
    # S[1] = S(1)
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )
    censored_loss = -c * torch.log(torch.gather(S_padded, 1, Y + 1).clamp(min=eps))
    neg_l = censored_loss + uncensored_loss
    loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss

class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False,save_best_model_stage=0.):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.save_best_model_stage = save_best_model_stage

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        
        score = -val_loss if epoch >= self.save_best_model_stage else 0.

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def state_dict(self):
        return {
            'patience': self.patience,
            'stop_epoch': self.stop_epoch,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }
    def load_state_dict(self,dict):
        self.patience = dict['patience']
        self.stop_epoch = dict['stop_epoch']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        self.early_stop = dict['early_stop']
        self.val_loss_min = dict['val_loss_min']

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss







from typing import Union
import ast
import sys
import os.path as osp
import yaml
from collections import OrderedDict
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributions as dist
import random
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def check_list_consistency(A, B):
    _A, _B = list(A), list(B)
    assert len(_A) == len(_B), "Consistency check failed."
    for a in _A:
        assert a in _B, "Consistency check failed."
    for b in _B:
        assert b in _A, "Consistency check failed."

def fill_placeholder(target, fill, ind='{}'):
    _target = str(target)
    _fill = str(fill)
    if ind not in _target:
        return target

    new_target = _target.replace(ind, _fill)
    
    if isinstance(target, int):
        return int(new_target)
    elif isinstance(target, float):
        return float(new_target)

    return new_target

def create_output_converter(converter:Union[None,str]=None):
    if converter == 'sigmoid':
        func_converter = torch.sigmoid
    elif converter == 'softmax':
        func_converter = lambda x: F.softmax(x, dim=-1)
    else:
        func_converter = lambda x: x
    print(f"[setup] the converter of network output: {converter}.")
    return func_converter

def random_mask_instance(bag:Tensor, mask_ratio:float, scale=1, mask_way='mask_zero'):
    if mask_ratio <= 0 or mask_ratio > 1:
        return bag

    N = bag.shape[0]
    n_square = scale * scale
    assert N % n_square == 0, 'bag must consist of square instances.'
    N_scaled = N // n_square

    # calculate under the scaled version
    n_keep = max(1, int(N_scaled * (1 - mask_ratio)))
    idxs = np.random.permutation(N_scaled)
    idxs_keep = np.sort(idxs[:n_keep])
    
    # restore to the original scale
    idxs_keep = idxs_keep.reshape(-1, 1) * np.array([n_square] * n_square).reshape(1, -1) + \
        np.array([_ for _ in range(n_square)]).reshape(1, -1)
    idxs_keep = idxs_keep.reshape(-1).tolist()

    if mask_way == 'discard':
        return bag[idxs_keep]
    elif mask_way == 'mask_zero':
        new_bag = torch.zeros_like(bag)
        new_bag[idxs_keep] = bag[idxs_keep]
        return new_bag
    else:
        raise NotImplementedError("Not support for mask_way={}.".format(mask_way))

def add_prefix_to_filename(path, prefix=''):
    dir_name, file_name = osp.split(path)
    file_name = prefix + '_' + file_name
    return osp.join(dir_name, file_name)

def get_kfold_pids(pids, num_fold=5, keep_pids=None, random_state=42):
    kfold_pids = []
    cur_pids = [] if keep_pids is None else keep_pids
    if num_fold <= 1:
        kfold_pids.append(cur_pids + pids)
    else:
        kfold = KFold(n_splits=num_fold, shuffle=True, random_state=random_state)
        X = np.ones((len(pids), 1))
        for _, fold_index in kfold.split(X):
            kfold_pids.append(cur_pids + [pids[_i] for _i in fold_index])
    return kfold_pids

def get_label_mask(t, c, bins):
    n = t.shape[0]
    z = (torch.arange(bins).view(1, -1) * torch.ones((n, 1))).to(t.device)
    label = torch.where(c.to(torch.bool), z > t, z == t).to(torch.float)
    label_mask = (z <= t).to(torch.int) # we ignore the location whose value is greater than t
    return label, label_mask

def get_patient_data(df:pd.DataFrame, at_column='patient_id'):
    df_gps = df.groupby('patient_id').groups
    df_idx = [i[0] for i in df_gps.values()]
    pat_df = df.loc[df_idx, :]
    pat_df = pat_df.reset_index(drop=True)
    return pat_df

def sampling_data(data, num:Union[int,float]):
    total = len(data)
    if isinstance(num, float):
        assert num < 1.0 and num > 0.0
        num = int(total * num)
    assert num < total
    idxs = np.random.permutation(total)
    idxs_sampled = idxs[:num]
    idxs_left = idxs[num:]
    data_sampled = [data[i] for i in idxs_sampled]
    data_left = [data[i] for i in idxs_left]
    return data_sampled, data_left

def rename_keys(d, prefix_name, sep='/'):
    newd = dict()
    for k, v in d.items():
        newd[prefix_name + sep + k] = v
    return newd

def agg_tensor(collector, data):
    for k in data.keys():
        if k not in collector or collector[k] is None:
            collector[k] = data[k]
        else:
            collector[k] = torch.cat([collector[k], data[k]], dim=0)
    return collector

def fetch_kws(d, prefix:str=''):
    if prefix == '':
        return d
    else:
        ret = dict()
        for k in d.keys():
            if k.startswith(prefix):
                new_key = k.split(prefix)[1]
                if len(new_key) < 2:
                    continue
                ret[new_key[1:]] = d[k]
        return ret

def parse_str_dims(s, sep='-', dtype=int):
    if type(s) != str:
        return [s]
    else:
        return [dtype(_) for _ in s.split(sep)]

def set_requires_grad(module, requires_grad):
    if isinstance(module, torch.nn.Module):
        for param in module.parameters():
            param.requires_grad = requires_grad
    elif isinstance(module, torch.nn.parameter.Parameter):
        module.requires_grad = requires_grad
    else:
        raise TypeError(f"The type of the module is wrong: {type(module)}")

    return None

def freeze_param(module):
    if module is None:
        print("model is not instantialized.")
        return
    set_requires_grad(module, False)

def unfreeze_param(module):
    if module is None:
        print("model is not instantialized.")
        return
    set_requires_grad(module, True)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print('[setup] seed: {}'.format(seed))

def setup_device(cuda_id):
    if cuda_id is not None and cuda_id >= 0:
        if torch.cuda.is_available():
            torch.cuda.set_device(cuda_id)
            print(f'[setup] device: CUDA {cuda_id}.')
        else:
            raise RuntimeError(f'[setup] CUDA is not found.')
    else:
        raise RuntimeError(f'[setup] got invalid cuda_id ({cuda_id}).')

# worker_init_fn = seed_worker
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# generator = g
def seed_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def sparse_str(s):
    res = None
    try:
        res = ast.literal_eval(s)
    except Exception as e:
        res = s
    return res

def get_model_cfg(path_run_log):
    full_path_run_log = osp.join(path_run_log, 'config.yaml')
    if osp.exists(full_path_run_log):
        with open(full_path_run_log, "r") as setting:
            cfg = yaml.load(setting, Loader=yaml.FullLoader)
        print("[Model CFG] loaded config from", full_path_run_log)
        return cfg

    full_path_run_log = osp.join(path_run_log, 'print_config.txt')
    if osp.exists(full_path_run_log):
        cfg = dict()
        with open(full_path_run_log, "r") as file:
            for line in file.readlines():
                line = line.strip()
                if '-->' not in line:
                    continue
                k, v = line.split('-->')
                k, v = k.strip(), v.strip()
                cfg[k] = sparse_str(v)
        print("[Model CFG] loaded config from", full_path_run_log)
        return cfg
    
    raise RuntimeError(f"[Model CFG] Model configuration is not found in {path_run_log}.")

def print_config(config, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL CONFIGURATION ****************", file=f)
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val), file=f)
    print("**************** MODEL CONFIGURATION ****************", file=f)
    
    if print_to_path is not None:
        f.close()

def save_config(config, path_to_save):
    with open(path_to_save, "w") as f:
        yaml.dump(config, f)

def print_metrics(metrics, print_to_path=None):
    if print_to_path is not None:
        f = open(print_to_path, 'w')
    else:
        f = sys.stdout
    
    print("**************** MODEL METRICS ****************", file=f)
    for key in sorted(metrics.keys()):
        val = metrics[key]
        for v in val:
            cur_key = key + '/' + v[0]
            keystr  = "{}".format(cur_key) + (" " * (20 - len(cur_key)))
            valstr  = "{}".format(v[1])
            if isinstance(v[1], list):
                valstr = "{}, avg/std = {:.5f}/{:.5f}".format(valstr, np.mean(v[1]), np.std(v[1]))
            print("{} -->   {}".format(keystr, valstr), file=f)
    print("**************** MODEL METRICS ****************", file=f)
    
    if print_to_path is not None:
        f.close()

def args_grid(kwargs):
    """Builds a mesh grid with given keyword arguments for this Config class.
    If the value is not a list, then it is considered fixed"""

    class MncDc:
        """This is because np.meshgrid does not always work properly..."""

        def __init__(self, a):
            self.a = a  # tuple!

        def __call__(self):
            return self.a

    def merge_dicts(*dicts):
        """
        Merges dictionaries recursively. Accepts also `None` and returns always a (possibly empty) dictionary
        """
        from functools import reduce
        def merge_two_dicts(x, y):
            z = x.copy()  # start with x's keys and values
            z.update(y)  # modifies z with y's keys and values & returns None
            return z

        return reduce(lambda a, nd: merge_two_dicts(a, nd if nd else {}), dicts, {})


    sin = OrderedDict({k: v for k, v in kwargs.items() if isinstance(v, list)})
    for k, v in sin.items():
        copy_v = []
        for e in v:
            copy_v.append(MncDc(e) if isinstance(e, tuple) else e)
        sin[k] = copy_v

    grd = np.array(np.meshgrid(*sin.values()), dtype=object).T.reshape(-1, len(sin.values()))
    return [merge_dicts(
        {k: v for k, v in kwargs.items() if not isinstance(v, list)},
        {k: vv[i]() if isinstance(vv[i], MncDc) else vv[i] for i, k in enumerate(sin)}
    ) for vv in grd]




 