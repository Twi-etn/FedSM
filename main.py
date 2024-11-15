from options import args_parser
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
from Model.Resnet8_256 import ResNet_cifar
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data.dataloader import DataLoader
from Dataset.dataset import classify_label, show_clients_data_distribution, Indices2Dataset, Clip_Indices2Dataset
from Dataset.sample_dirichlet import clients_indices
from Dataset.long_tailed_cifar10 import train_long_tail
from Participator.Server import Global
from DKD import DKD
import datetime
import copy
import random
import clip
import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from utils import *


class Logger(object):
    logfile = ""

    def __init__(self, filename=""):
        self.logfile = filename
        self.terminal = sys.stdout
        return

    def write(self, message):
        self.terminal.write(message)
        if self.logfile != "":
            try:
                self.log = open(self.logfile, "a")
                self.log.write(message)
                self.log.close()
            except:
                pass

    def flush(self):
        pass


def load_labels_name(filename):
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
    return obj


class KDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        kd_loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        return kd_loss


class Local(object):
    def __init__(self, data_client, class_list: list):
        args = args_parser()

        self.data_client = data_client
        self.device = args.device
        self.class_compose = class_list  # num of samples per class
        self.criterion = CrossEntropyLoss().to(args.device)
        if args.kd_type == 'KD':
            self.kd_criterion = KDLoss(T=args.T).to(args.device)
        else:
            self.kd_criterion = DKD(temperature=args.T, dkd_beta=args.dkd_beta).to(args.device)
        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
        self.pre_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    def local_train(self, args, global_params, text_features, communication_round):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        for local_ep in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True, num_workers=args.num_workers)
            for data_batch in data_loader:
                images, labels, clip_images = data_batch
                images, labels, clip_images = images.to(self.device), labels.to(self.device), clip_images.to(self.device)  # tensor
                images = transform_train(images)

                # compute client's output
                features, outputs = self.local_model(images)
                outputs = outputs.float()
                # get clip feature encode
                with torch.no_grad():
                    image_features = clip_model.encode_image(clip_images)
                image_features = image_features.float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                features = features.float()
                features_norm = features / features.norm(dim=-1, keepdim=True)
                clip_logits = (100.0 * image_features @ text_features.T)  # CLIP中的点乘N方矩阵  但是乘100干嘛
                # # clip_image_logits, clip_text_logits = clip_model(clip_images, text_input)
                # # Eq. 1
                loss1 = self.criterion(outputs, labels)
                if args.kd_type == 'KD':
                    loss2 = self.kd_criterion(outputs, clip_logits)
                else:
                    loss2 = self.kd_criterion(outputs, clip_logits, labels, communication_round)
                # KD loss & CE loss & Cos Embedding Loss??????
                loss = (loss1 + loss2 * args.kd_alpha)
                # labels_teacher = clip_logits.argmax(dim=-1, keepdim=False)  hard
                # loss = self.criterion(outputs, labels_teacher) * 0.5 + self.criterion(outputs, labels) * 0.5
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.local_model.state_dict()

    def get_local_centro(self, args):
        g_m = self.local_model
        data_loader = DataLoader(dataset=self.data_client, batch_size=args.batch_size_local_training,
                                 shuffle=True, num_workers=args.num_workers)

        global_feats_all, labels_all = [], []
        with torch.set_grad_enabled(False):
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                global_feat, _ = g_m(images)
                global_feats_all.append(global_feat.data.clone())
                labels_all.append(labels)
        cls_means = get_cls_mean_from_feats(global_feats_all, labels_all)
        # syn_c 平均特征向量
        real_id = []
        for k, _ in cls_means.items():
            real_id.append(k)
        self.real_id = real_id
        self.feats_all = torch.cat(global_feats_all, dim=0)
        self.labels_all = torch.cat(labels_all)
        # feats_all, labels_all: client_data's feature and label [len_data, fea_dim] [len_data]
        return cls_means, self.feats_all, self.labels_all

    def local_crt(self, glo_means, args, label_similarity):
        # 明天这里写一个参考CLIP2FL的  每次在这里插一个全新的Linear层
        fs_all = self.feats_all
        new_classifier = nn.Linear(512, args.num_classes).to(args.device)
        optim_new_classifier = SGD(new_classifier.parameters(), lr=args.lr_retrain)
        # re-train 时先冻住卷积层  抽样的时候应该是weights与样本数量一样多 而不是和样本类别一样多
        crt_dataset = MixupDataset_norm(glo_means, fs_all, args, self.labels_all, label_similarity)
        new_classifier.train()
        for i in range(args.local_bal_epochs):
            crt_loader = DataLoader(dataset=crt_dataset, batch_size=args.crt_batch_size, shuffle=True)
            for feat, cls in crt_loader:
                feat, cls = feat.to(self.device), cls.to(self.device)
                outputs = new_classifier(feat)
                loss = self.criterion(outputs, cls)
                optim_new_classifier.zero_grad()
                loss.backward()
                optim_new_classifier.step()
        new_classifier.eval()
        return copy.deepcopy(new_classifier.state_dict())


# clip_model, clip_transform = None, None


def CLIP2MIX(args):
    # global clip_model, clip_transform
    # clip_model, clip_transform = clip.load(args.clip_base_model, args.device, download_root='../CLIP2FL/clipmodel')
    print(
        'imb_factor:{ib}, non_iid:{non_iid}\n'
        'lr_local_training:{lr_local_training}, lr_retrain:{lr_retrain}, num_of_feature:{num_of_feature}\n '
        're_training_epoch:{crt_epoch}\n'.format(
            ib=args.imb_factor,
            non_iid=args.non_iid_alpha,
            lr_local_training=args.lr_local_training,
            lr_retrain=args.lr_retrain,
            num_of_feature=args.crt_feat_num,
            crt_epoch=args.crt_ep))
    random_state = np.random.RandomState(args.seed)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # logger 日志文件与目录创建
    current_time = datetime.datetime.now()
    current_time_str = current_time.strftime("%Y_%m_%d_%H_%M_%S")
    log_name = (current_time_str+'_CLIP2MIX_seed_' + str(args.seed) + 'IF' + str(args.imb_factor) + '.log')
    model_dir = os.path.join(args.result_save, args.dataset, 'main_clip2mix')
    if not os.path.exists(model_dir):
        print("Directory does not exist! Making directory {}".format(model_dir))
        os.makedirs(model_dir)
    sys.stdout = Logger(os.path.join(model_dir, str(log_name)))
    sys.stderr = Logger(os.path.join(model_dir, str(log_name)))
    if not os.path.exists(args.result_save):
        os.mkdir(args.result_save)

    # Load data
    transform_all = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.num_classes == 10:
        data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=transform_all)
        clip_data_local_training = datasets.CIFAR10(args.path_cifar10, train=True, download=True, transform=clip_transform)
        data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=transform_all)
    elif args.num_classes == 100:
        data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=transform_all)
        clip_data_local_training = datasets.CIFAR100(args.path_cifar100, train=True, download=True, transform=clip_transform)
        data_global_test = datasets.CIFAR100(args.path_cifar100, train=False, transform=transform_all)
    # clip_data_local_training 与 data_local_training用了不同的transform生成不同的数据集，主要是适用于普通CNN与CLIP的区别

    # get label_name from datasets
    if args.num_classes == 10:
        cifar10_path = "data/CIFAR10/cifar-10-batches-py"
        obj_cifar10 = load_labels_name(os.path.join(cifar10_path, 'batches.meta'))
        label_name = obj_cifar10['label_names']
    elif args.num_classes == 100:
        cifar100_path = "data/CIFAR100/cifar-100-python"
        obj_cifar100 = load_labels_name(os.path.join(cifar100_path, 'meta'))
        label_name = obj_cifar100['fine_label_names']

    # CLIP PART and Loading data
    clip_model.eval()
    text_inputs = clip.tokenize([f"a photo of a {c}" for c in label_name]).to(args.device)  # torch.size([10, 77])

    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)
    # print(text_features)
    text_features = text_features.float()
    text_features /= text_features.norm(dim=-1, keepdim=True)  # torch.size([10, 512])

    label_input = clip.tokenize([f"{c}" for c in label_name]).to(args.device)
    with torch.no_grad():
        label_features = clip_model.encode_text(label_input)

    # distance calc
    label_features = label_features.float()
    label_features /= label_features.norm(dim=-1, keepdim=True)  # torch.size([10, 512])
    label_similarity = torch.mm(label_features, label_features.T)
    # if args.hint == 'dot':
    #     label_features /= label_features.norm(dim=-1, keepdim=True)  # torch.size([10, 512])
    #     label_similarity = torch.mm(label_features, label_features.T)
    # elif args.hint == 'l2':
    #     l2_distances = torch.cdist(label_features, label_features, p=2)
    #     label_similarity = 1 / (1.0 + l2_distances)
    # elif args.hint == 'l1':
    #     l2_distances = torch.cdist(label_features, label_features, p=1)
    #     label_similarity = 1 / (1.0 + l2_distances)
    # elif args.hint == 'dot':
    #     # inner product
    #     label_similarity = torch.mm(label_features, label_features.T)

    label_similarity = label_similarity.cpu().numpy()
    row_means = np.mean(label_similarity, axis=1)
    np.fill_diagonal(label_similarity, row_means)
    # label_similarity = np.ones((args.num_classes, args.num_classes))

    # Distribute data
    list_label2indices = classify_label(data_local_training, args.num_classes)
    # print(list_label2indices)
    # print('-----------------------')
    # heterogeneous and long_tailed setting
    _, list_label2indices_train_new = train_long_tail(copy.deepcopy(list_label2indices), args.num_classes,
                                                      args.imb_factor, args.imb_type)
    # list_label2indices_train_new: lable2indices under long-tail
    list_client2indices = clients_indices(copy.deepcopy(list_label2indices_train_new), args.num_classes,
                                          args.num_clients, args.non_iid_alpha, args.seed)

    # len(list_client2indices) = 20  [indices of each client]
    original_dict_per_client = show_clients_data_distribution(data_local_training, list_client2indices,
                                                              args.num_classes)
    global_model = Global(num_classes=args.num_classes, device=args.device, args=args)
    total_clients = list(range(args.num_clients))
    # two type of dataset
    clip_indices2data = Clip_Indices2Dataset(data_local_training, clip_data_local_training)
    indices2data = Indices2Dataset(data_local_training)
    re_trained_acc = []
    best_acc = 0
    best_model = None

    for r in tqdm(range(1, args.num_rounds+1), desc='server-training'):
        global_params = global_model.download_params()
        online_clients = random_state.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []

        real_clients = copy.deepcopy(online_clients)
        if args.crt_ep != 0 and args.crt_ep >= args.num_rounds - r:
            # 不等于0 且要开始重训练的时候
            real_clients = total_clients

        print(real_clients)

        # local training
        for client in online_clients:
            # online_clients
            clip_indices2data.load(list_client2indices[client])
            data_client = clip_indices2data
            list_nums_local_data.append(len(data_client))
            local_model = Local(data_client=data_client, class_list=original_dict_per_client[client])
            # local update
            local_params = local_model.local_train(args, copy.deepcopy(global_params), text_features, r)
            list_dicts_local_params.append(copy.deepcopy(local_params))
        # aggregating local models with FedAvg
        fedavg_params = global_model.initialize_for_model_fusion(list_dicts_local_params, list_nums_local_data)
        eval_params = copy.deepcopy(fedavg_params)

        # classfier learning
        if args.crt_ep >= args.num_rounds - r:
            ccvr_means, list_nums_local_data_crt, clients = [], [], []
            for client in real_clients:
                indices2data = Indices2Dataset(data_local_training)
                indices2data.load(list_client2indices[client], client)
                data_client = indices2data
                list_nums_local_data_crt.append(len(data_client))
                local_model = Local(data_client=data_client, class_list=original_dict_per_client[client])
                if client in online_clients:
                    clients.append(local_model)
                local_model.local_model.load_state_dict(fedavg_params)
                real_mean, c_f, _ = local_model.get_local_centro(args)
                ccvr_means.append(real_mean)
            # print('ccvr_means', type(ccvr_means), ccvr_means[0])
            norm_means = global_model.cal_norm_mean(copy.deepcopy(ccvr_means), original_dict_per_client)
            # 全局平均特征向量  & 更新

            # for k, v in norm_means.items():
            #     global_avg_prototype[k] = v

            mixup_cls_params = []
            for c_id, l_model in enumerate(clients):
                mixup_cls_param = l_model.local_crt(norm_means, args, label_similarity)
                mixup_cls_params.append(mixup_cls_param)

            mixup_classifier = model_fusion(mixup_cls_params, list_nums_local_data_crt)
            # 如果有分类器重训练， 就将新的分类器插到验证模型上
            for name_param in reversed(eval_params):
                if name_param == 'classifier.bias':
                    eval_params[name_param] = mixup_classifier['bias']
                if name_param == 'classifier.weight':
                    eval_params[name_param] = mixup_classifier['weight']
                    break

        # global eval
        one_re_train_acc = global_model.global_eval(eval_params, data_global_test, args.batch_size_test)
        re_trained_acc.append(one_re_train_acc)
        if one_re_train_acc > best_acc:
            best_acc = one_re_train_acc
            best_model = copy.deepcopy(eval_params)
        global_model.syn_model.load_state_dict(copy.deepcopy(fedavg_params))
        if r % 10 == 0:
            print(re_trained_acc)
            # global_model.syn_model.load_state_dict(copy.deepcopy(best_model))

    print(re_trained_acc)
    print('Max_Acc = ', max(re_trained_acc))
    print('Hyper parameters:')
    print(vars(args))
    return max(re_trained_acc)


if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)  # cpu
    torch.cuda.manual_seed(args.seed)  # gpu
    np.random.seed(args.seed)  # numpy
    random.seed(args.seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn
    # CLIP Loading
    clip_model, clip_transform = clip.load(args.clip_base_model, args.device, download_root='../CLIP2FL/clipmodel')
    CLIP2MIX(args)

