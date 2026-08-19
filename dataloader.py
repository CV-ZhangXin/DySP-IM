import os
import csv
import torch
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def get_patient_label(csv_file):
    patients_list=[]
    labels_list=[]
    label_file = readCSV(csv_file)
    for i in range(0, len(label_file)):
        patients_list.append(label_file[i][0])
        labels_list.append(label_file[i][1])
    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)

def get_patient_label_bracs(csv_file):
    patients_list=[]
    labels_list=[]
    datasplit_list=[]
    label_file = readCSV(csv_file)
    for i in range(0, len(label_file)):
        patients_list.append(label_file[i][0])
        labels_list.append(label_file[i][1])
        datasplit_list.append(label_file[i][2])
    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object),np.array(datasplit_list,dtype=object)

def data_split(full_list, ratio, shuffle=True,label=None,label_balance_val=True):
    """
    dataset split: split the full_list randomly into two sublist (val-set and train-set) based on the ratio
    :param full_list: 
    :param ratio:     
    :param shuffle:  
    """
    # select the val-set based on the label ratio
    if label_balance_val and label is not None:
        _label = label[full_list]
        _label_uni = np.unique(_label)
        sublist_1 = []
        sublist_2 = []

        for _l in _label_uni:
            _list = full_list[_label == _l]
            n_total = len(_list)
            offset = int(n_total * ratio)
            if shuffle:
                random.shuffle(_list)
            sublist_1.extend(_list[:offset])
            sublist_2.extend(_list[offset:])
    else:
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        sublist_1 = full_list[:offset]
        sublist_2 = full_list[offset:]

    return sublist_1, sublist_2


def get_kflod(k, patients_array, labels_array,val_ratio=False,label_balance_val=True):
    if k > 1:
        skf = StratifiedKFold(n_splits=k)
    else:
        raise NotImplementedError
    train_patients_list = []
    train_labels_list = []
    test_patients_list = []
    test_labels_list = []
    val_patients_list = []
    val_labels_list = []
    for train_index, test_index in skf.split(patients_array, labels_array):
        if val_ratio != 0.:
            val_index,train_index = data_split(train_index,val_ratio,True,labels_array,label_balance_val)
            x_val, y_val = patients_array[val_index], labels_array[val_index]
        else:
            x_val, y_val = [],[]
        x_train, x_test = patients_array[train_index], patients_array[test_index]
        y_train, y_test = labels_array[train_index], labels_array[test_index]

        train_patients_list.append(x_train)
        train_labels_list.append(y_train)
        test_patients_list.append(x_test)
        test_labels_list.append(y_test)
        val_patients_list.append(x_val)
        val_labels_list.append(y_val)
        
    # print("get_kflod.type:{}".format(type(np.array(train_patients_list))))
    return np.array(train_patients_list,dtype=object), np.array(train_labels_list,dtype=object), np.array(test_patients_list,dtype=object), np.array(test_labels_list,dtype=object),np.array(val_patients_list,dtype=object), np.array(val_labels_list,dtype=object)

def get_tcga_parser(root,cls_name,mini=False):
        x = []
        y = []

        for idx,_cls in enumerate(cls_name):
            _dir = 'mini_pt' if mini else 'pt_files'
            _files = os.listdir(os.path.join(root,_cls,'features',_dir))
            _files = [os.path.join(os.path.join(root,_cls,'features',_dir,_files[i])) for i in range(len(_files))]
            x.extend(_files)
            y.extend([idx for i in range(len(_files))])
            
        return np.array(x).flatten(),np.array(y).flatten()

class TCGADataset(Dataset):
    
    def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=True,keep_same_psize=0,is_train=False,_type='nsclc'):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(TCGADataset, self).__init__()

        self.patient_name = file_name
        self.patient_label = file_label
        self.max_patch = max_patch
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'h5_files')) if keep_same_psize else os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.slide_label = []
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        self.namem=_type

        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    self.slide_name.append(torch.load(os.path.join(self.root,'pt_files',_sides[_idx])))
                else:
                    self.slide_name.append(_sides[_idx])
                self.slide_label.append(self.patient_label[i])
                
        if _type.lower() == 'nsclc':
            self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]
        elif _type.lower() == 'brca':
            self.slide_label = [ 0 if _l == 'IDC' else 1 for _l in self.slide_label]

    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        label = self.slide_label[idx]

        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path),weights_only=True)
        if self.namem.lower() == 'nsclc':
            proto_path = os.path.join('/data3/shihuazhan/DiT', 'tcga', file_path)
            proto_path = os.path.join('/data3/shihuazhan/DiT/generate/tcga', 't_10', file_path)
        elif self.namem.lower() == 'brca':
            proto_path = os.path.join('/data3/shihuazhan/DiT', 'brca', file_path)
            proto_path = os.path.join('/data3/shihuazhan/DiT/generate/brca', 't_100', file_path)
        prototypes_pool = torch.load(proto_path, weights_only=True)
        if self.is_train:
            a = prototypes_pool[torch.randint(0, prototypes_pool.size(0), (1,)).item()]
        else:
            a = prototypes_pool[0]
        return [features, a], int(label)



# class TCGADataset(Dataset):
    
#     def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=True,keep_same_psize=0,is_train=False):
#         """
#         Args
#         :param images: 
#         :param transform: optional transform to be applied on a sample
#         """
#         super(TCGADataset, self).__init__()

#         self.patient_name = file_name
#         self.patient_label = file_label
#         self.max_patch = max_patch
#         self.root = root
#         self.all_pts = os.listdir(os.path.join(self.root,'h5_files')) if keep_same_psize else os.listdir(os.path.join(self.root,'pt_files'))
#         self.slide_name = []
#         self.slide_label = []
#         self.persistence = persistence
#         self.keep_same_psize = keep_same_psize
#         self.is_train = is_train

#         for i,_patient_name in enumerate(self.patient_name):
#             _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
#             _ids = np.where(_sides != '0')[0]
#             for _idx in _ids:
#                 if persistence:
#                     self.slide_name.append(torch.load(os.path.join(self.root,'pt_files',_sides[_idx])))
#                 else:
#                     self.slide_name.append(_sides[_idx])
#                 self.slide_label.append(self.patient_label[i])
#         self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]

#     def __len__(self):
#         return len(self.slide_name)

#     def __getitem__(self, idx):
#         """
#         Args
#         :param idx: the index of item
#         :return: image and its label
#         """
#         file_path = self.slide_name[idx]
#         label = self.slide_label[idx]

#         if self.persistence:
#             features = file_path
#         else:
#             features = torch.load(os.path.join(self.root,'pt_files',file_path))
#         return features , int(label)

class C16Dataset(Dataset):

    def __init__(self, file_name, file_label, root, persistence=False, keep_same_psize=0, is_train=False):
        """
        Args
        :param file_name: list of file names (without extension)
        :param file_label: list of labels
        :param root: root directory of the dataset
        :param persistence: whether to load all data into memory
        :param keep_same_psize: (not used in this logic, but kept for consistency)
        :param is_train: (not used in this logic, but kept for consistency)
        """
        super(C16Dataset, self).__init__()
        self.file_name = file_name
        self.slide_label = [int(_l) for _l in file_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        # --- 主要修改点开始 ---

        # 1. 决定使用哪个子文件夹 ('pt' or 'conch')
        pt_path = os.path.join(self.root, 'pt')
        conch_path = os.path.join(self.root, 'conch')

        if os.path.isdir(pt_path):
            self.data_subdir = 'pt'
        elif os.path.isdir(conch_path):
            self.data_subdir = 'conch'
        else:
            # 如果两个文件夹都不存在，抛出错误
            raise FileNotFoundError(f"Neither 'pt' nor 'conch' directory found in the root path: {self.root}")
        
        print(f"Data will be loaded from: '{self.data_subdir}' directory.") # 添加一个打印信息，方便调试

        # 2. 在持久化加载时使用正确的子文件夹
        if persistence:
            # 使用 self.data_subdir 变量来构建路径
            self.feats = [torch.load(os.path.join(root, self.data_subdir, _f + '.pt')) for _f in file_name]

        # --- 主要修改点结束 ---

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            # --- 另一个修改点 ---
            # 3. 在按需加载时也使用正确的子文件夹
            # 使用 self.data_subdir 变量来构建路径
            dir_path = os.path.join(self.root, self.data_subdir)
            file_path = os.path.join(dir_path, self.file_name[idx] + '.pt')
            features = torch.load(file_path, weights_only=True)
        
        #proto_path = os.path.join('/data3/shihuazhan/DiT', 'camelyon16', self.file_name[idx] + '.pt')
        proto_path = os.path.join('/data3/shihuazhan/DiT/generate/camelyon16', 't_20', self.file_name[idx] + '.pt')
        prototypes_pool = torch.load(proto_path, weights_only=True) # shape: (50, 1024)
        if self.is_train:
            random_idx = torch.randint(0, prototypes_pool.size(0), (1,)).item()
            a = prototypes_pool[random_idx]
        else:
            a = prototypes_pool[0]
        
        label = int(self.slide_label[idx])

        # return features, label
        return (features, a), label

    





class BRACSDataset(Dataset):
    
    def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=True,keep_same_psize=0,is_train=False,n_class=3):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(BRACSDataset, self).__init__()

        self.patient_name = file_name
        self.patient_label = file_label
        self.max_patch = max_patch
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'h5_files')) if keep_same_psize else os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.slide_label = []
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train
        dic3={'N':0,'PB':0,'UDH':0,'FEA':1,'ADH':1,'DCIS':2,'IC':2}
        dic7={'N':0,'PB':1,'UDH':2,'FEA':3,'ADH':4,'DCIS':5,'IC':6}
        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    self.slide_name.append(torch.load(os.path.join(self.root,'pt_files',_sides[_idx])))
                else:
                    self.slide_name.append(_sides[_idx])
                self.slide_label.append(self.patient_label[i])
        self.slide_label = [ dic3[_l] for _l in self.slide_label] if n_class==3 else  [ dic7[_l]  for _l in self.slide_label]

    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        # file_path = self.csv_file[idx] # ['1_1.png']
        # patient_path = file_path[1]
        label = self.slide_label[idx]

        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path))
        #features = os.path.join(self.root,'pt_files',file_path)

    


        
        #proto_path = os.path.join('/data3/shihuazhan/DiT', 'camelyon16', self.file_name[idx] + '.pt')
        proto_path = os.path.join('/data3/shihuazhan/DiT/generate/bracs', 't_5', self.patient_name[idx] + '.pt')
        prototypes_pool = torch.load(proto_path, weights_only=True) # shape: (50, 1024)
        if self.is_train:
            random_idx = torch.randint(0, prototypes_pool.size(0), (1,)).item()
            a = prototypes_pool[random_idx]
        else:
            a = prototypes_pool[0]
        


        # return features, label
        return (features, a), label

class TCGA_Survival(Dataset):
    def __init__(self, excel_file, root=None,persistence=True,all_fold=5):
        print('[dataset] loading dataset from %s' % (excel_file))
        self.root = root
        self.persistence = persistence
        self.all_pts = os.listdir(self.root)
        rows = pd.read_csv(excel_file)
        self.rows = self.disc_label(rows)
        label_dist = self.rows['Label'].value_counts().sort_index()
        print('[dataset] discrete label distribution: ')
        print(label_dist)
        print('[dataset] dataset from %s, number of cases=%d' % (excel_file, len(self.rows)))

        # 得到case ID下的所有相关pt文件，用字典保存
        self.slide_name = {}
        for index, row in rows.iterrows():
            case_name = row['ID']
            if self.persistence:
                slides = [ torch.load(os.path.join(self.root,slide)) for slide in self.all_pts if case_name in slide]
                slides = torch.cat(slides,dim=0)
            else:
                slides = [ slide for slide in self.all_pts if case_name in slide]
            self.slide_name[str(case_name)] = slides
        
        # 得到多折划分
        self.ratio=1. / all_fold
        self.all_fold = all_fold
        self.sample_index = random.sample(range(len(self.rows)), len(self.rows))
        self.num_split = round((len(self.rows) - 1) * self.ratio)

    def get_split(self, fold=0):
        # random.seed(1)
        assert 0 <= fold <= self.all_fold-1, 'fold should be in 0 ~ {}'.format(self.all_fold-1)
        if fold < 1 / self.ratio - 1:
            val_split = self.sample_index[fold * self.num_split: (fold + 1) * self.num_split]
        else:
            val_split = self.sample_index[fold * self.num_split:]
        train_split = [i for i in self.sample_index if i not in val_split]
        print("[dataset] training split: {}, validation split: {}".format(len(train_split), len(val_split)))
        return train_split, val_split 
    
    def read_WSI(self, path):
        wsi = [torch.load(os.path.join(self.root,x)) for x in path]
        wsi = torch.cat(wsi, dim=0)
        return wsi

    def __getitem__(self, index):
        case = self.rows.iloc[index, :].values.tolist()
        Study, ID, Event, Status= case[:4]
        Label = case[-1]
        Censorship = 1 if int(Status) == 0 else 0
        if self.persistence:
            WSI = self.slide_name[ID]
        else:
            WSI = self.read_WSI(self.slide_name[ID])
        return (ID, WSI, Event, Censorship, Label)

    def __len__(self):
        return len(self.rows)

    def disc_label(self, rows):
        n_bins, eps = 4, 1e-6
        uncensored_df = rows[rows['Status'] == 1]
        disc_labels, q_bins = pd.qcut(uncensored_df['Event'], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = rows['Event'].max() + eps
        q_bins[0] = rows['Event'].min() - eps
        disc_labels, q_bins = pd.cut(rows['Event'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        # missing event data
        disc_labels = disc_labels.values.astype(int)
        disc_labels[disc_labels < 0] = -1
        rows.insert(len(rows.columns), 'Label', disc_labels)
        return rows


class TCGASplitDataset(Dataset):
    def __init__(self, label_csv, root, split='train', persistence=True, keep_same_psize=False, _type='nsclc'):
        """
        Args:
            label_csv (str): 路径到包含三列的CSV文件: [slide_id, label, split]
            root (str): 数据根目录，pt 文件位于 root/pt_files/
            split (str): 'train', 'val', 或 'test'
            persistence (bool): 是否在初始化时就加载所有特征到内存
            keep_same_psize (bool): 未使用（保留以兼容旧代码）
            _type (str): 数据集子类型，目前仅支持 'nsclc'
        """
        super(TCGASplitDataset, self).__init__()

        self.root = root
        self.persistence = persistence
        self._type = _type
        self.pt_dir = os.path.join(root, 'pt_files')
        
        # 读取标签文件
        df = pd.read_csv(label_csv)
        assert df.shape[1] >= 3, "标签文件必须至少有三列: [slide_id, label, split]"
        df.columns = df.columns[:3].tolist() + list(df.columns[3:])  # 只关心前三列
        df = df.iloc[:, :3]
        df.columns = ['slide_id', 'label', 'split']
        
        # 过滤指定 split
        self.df = df[df['split'] == split].reset_index(drop=True)
        
        # 构建 slide_name 和 slide_label
        self.slide_name = []
        self.slide_label = []

        for _, row in self.df.iterrows():
            slide_id = row['slide_id']
            label = row['label']
            
            pt_path = f"{slide_id}.pt"
            if not os.path.exists(os.path.join(self.pt_dir, pt_path)):
                # 尝试不带 .pt 后缀？或者原始命名如 C3L-00140-21.pt
                # 假设 slide_id 就是文件名（不含扩展名）
                # 如果实际文件名就是 slide_id + '.pt'，则没问题
                pass

            if persistence:
                try:
                    features = torch.load(os.path.join(self.pt_dir, pt_path), weights_only=True)
                    self.slide_name.append(features)
                except Exception as e:
                    print(f"Warning: Failed to load {pt_path}: {e}")
                    continue
            else:
                self.slide_name.append(pt_path)

            # 标签映射：LUAD -> 0, 其他 -> 1
            if _type.lower() == 'nsclc':
                mapped_label = 0 if label == 'LUAD' else 1
            else:
                raise ValueError(f"Unsupported _type: {_type}")
            self.slide_label.append(mapped_label)

    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        label = self.slide_label[idx]
        if self.persistence:
            features = self.slide_name[idx]
        else:
            pt_file = self.slide_name[idx]
            features = torch.load(os.path.join(self.pt_dir, pt_file), weights_only=True)
        return features, int(label)