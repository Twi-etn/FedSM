import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import copy
import torch.nn.functional as F
import torch
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np


def classify_label_parallel(dataset, num_classes: int):
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    list1 = [[] for _ in range(num_classes)]
    offset = 0
    for _, labels in dataloader:
        for i, label in enumerate(labels):
            label = label.item()
            list1[label].append(i + offset)
        offset += 32
    return list1


def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1


def show_clients_data_distribution(dataset, clients_indices: list, num_classes):
    dict_per_client = []
    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset[idx][1]
            nums_data[label] += 1
        dict_per_client.append(nums_data)
        print(f'{client}: {nums_data} : sum{sum(nums_data)}')
    return dict_per_client


def partition_train_teach(list_label2indices: list, ipc, seed=None):
    random_state = np.random.RandomState(0)
    list_label2indices_teach = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_teach.append(indices[:ipc])

    return list_label2indices_teach


def partition_unlabel(list_label2indices: list, num_data_train: int):
    random_state = np.random.RandomState(0)
    list_label2indices_unlabel = []

    for indices in list_label2indices:
        random_state.shuffle(indices)
        list_label2indices_unlabel.append(indices[:num_data_train // 100])
    return list_label2indices_unlabel


def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


class Indices2Dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = None

    def load(self, indices: list, c_id = None):
        self.indices = indices
        self.c_id = c_id

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)


class Clip_Indices2Dataset(Dataset):
    def __init__(self, dataset, clip_dataset):
        self.dataset = dataset
        self.clip_dataset = clip_dataset
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        clip_image, *_ = self.clip_dataset[idx]
        # clip_image, _, _ = self.clip_dataset[idx] for Image Net
        return image, label, clip_image

    def __len__(self):
        return len(self.indices)


class Indices2Dataset_out(Dataset):
    def __init__(self, dataset, device="cuda"):
        self.dataset = dataset
        self.image, self.label = self.dataset[:][0], self.dataset[:][1]
        self.image, self.label = self.image.to(device), self.label.to(device)
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.image[idx], self.label[idx]
        # image, label = self.dataset[idx]
        return image, label

    def __len__(self):
        return len(self.indices)


class Clip_Indices2Dataset_out(Dataset):
    def __init__(self, dataset, clip_feature_list):
        self.dataset = dataset
        self.clip_feature_list = clip_feature_list
        self.indices = None

    def load(self, indices: list):
        self.indices = indices

    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, label = self.dataset[idx]
        clip_feature = self.clip_feature_list[idx]
        return image, label, clip_feature

    def __len__(self):
        return len(self.indices)


class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def get_class_num(class_list):
    index = []
    compose = []
    for class_index, j in enumerate(class_list):
        if j != 0:
            index.append(class_index)
            compose.append(j)
    return index, compose


class MixupDataset_norm(Dataset):
    def __init__(self, mean, fs_all, args, real_id, label_similarity):
        self.data = []
        self.labels = []
        self.means = mean
        self.num_classes = args.num_classes
        self.device = args.device
        self.crt_feat_num = args.crt_feat_num
        self.fs_all = fs_all
        self.fs_len = len(fs_all)
        self.args = args
        self.real_id = real_id.cpu().numpy()
        self.label_sim = label_similarity
        # print(self.fs_len, self.means)
        # self.__mixup_syn_feat_pure_rand_norm__()
        self.mix_up_head_tail_guided(args.hint)

    # def __mixup_syn_feat_pure_rand_norm__(self):
    #     num = self.crt_feat_num
    #     l = self.args.uniform_left
    #     r_arg = self.args.uniform_right - l
    #     for cls in range(self.num_classes):
    #         fs_shuffle_idx = torch.randperm(self.fs_len)
    #         # 打乱了一个随机顺序
    #         for i in range(num):
    #             lam = np.round(l + r_arg * np.random.random(), 2)
    #             # uniform_left * right 控制lam生成的范围， np.round保留两位小数
    #             neg_f = self.fs_all[fs_shuffle_idx[i]]
    #             mixup_f = lam * self.means[cls] + (1 - lam) * F.normalize(neg_f.view(1, -1), dim=1).view(-1)
    #             self.data.append(mixup_f)
    #         self.labels += [cls]*num
    #         # 每个类生成了num个合成特征， 用于优化分类器，合成是通过全局平均特征与随机抽取的局部特征插值形成的
    #     self.data = torch.stack(self.data).to(self.device)
    #     self.labels = torch.tensor(self.labels).long().to(self.device)

    def mix_up_head_tail_guided(self, project):
        # self.means = self.means.to(self.device)
        self.fs_all = self.fs_all.to(self.device)
        num_samples = self.crt_feat_num
        l = self.args.uniform_left
        r_arg = self.args.uniform_right - l
        # 这里想一个对概率的映射 试一下
        for cls in range(self.num_classes):
            # 为每个特征分配权重
            self.means[cls] = self.means[cls].to(self.device)
            weights_norm = self.label_sim[cls]
            # weights_norm = weights_norm * 10
            # weights_norm[cls] = weights_norm.sum() / self.num_classes
            # 降低自身类的选择概率
            # weights_norm = torch.from_numpy(weights_norm)
            # if project == 'softmax':
            #     weights_norm = F.softmax(weights_norm, dim=-1)
            # elif project == 'sqrt':
            #     weights_norm = torch.sqrt(weights_norm)
            # elif project == 'square':
            #     weights_norm = weights_norm ** 2
            # elif project == 'log':
            #     weights_norm = torch.log(weights_norm)
            weights = torch.tensor([weights_norm[label.item()] for label in self.real_id])
            selected_indices = torch.multinomial(weights, num_samples, replacement=False)  # 有放回和无放回
            for i in range(num_samples):
                lam = np.round(l + r_arg * np.random.random(), 2)
                neg_f = self.fs_all[selected_indices[i]]
                # 要试一下 sqrt(lam) + sqrt(1 - lam)
                mixup_f = lam * self.means[cls] + (1 - lam) * F.normalize(neg_f.view(1, -1), dim=1).view(-1)
                self.data.append(mixup_f)
            self.labels += [cls] * num_samples
        self.data = torch.stack(self.data).to(self.device)
        self.labels = torch.tensor(self.labels).long().to(self.device)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]
