import torch
import copy
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch import stack, max, eq, no_grad, tensor, unsqueeze, split
from Model.Resnet8_256 import ResNet_cifar
from Model.ResNet50 import ResNet50
import torch.nn.functional as F


class Global(object):
    def __init__(self, num_classes: int, device: str, args):
        self.device = device
        self.num_classes = num_classes
        if num_classes != 1000:
            self.syn_model = ResNet_cifar(resnet_size=8, scaling=4,
                                      save_activations=False, group_norm_num_groups=None,
                                      freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(device)
        else:
            self.syn_model = ResNet50().to(self.device)

    def initialize_for_model_fusion(self, list_dicts_local_params: list, list_nums_local_data: list):
        # fedavg
        fedavg_global_params = copy.deepcopy(list_dicts_local_params[0])
        for name_param in list_dicts_local_params[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            fedavg_global_params[name_param] = value_global_param
        return fedavg_global_params

    def global_eval(self, fedavg_params, data_test, batch_size_test):
        self.syn_model.load_state_dict(fedavg_params)
        self.syn_model.eval()
        with no_grad():
            test_loader = DataLoader(data_test, batch_size_test, num_workers=8)
            num_corrects = 0
            for data_batch in test_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                _, predicts = max(outputs, -1)
                num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            accuracy = num_corrects / len(data_test)
        return accuracy

    def global_eval_Image_Net(self, fedavg_params, data_test, batch_size_test, img_num_per_cls):
        self.syn_model.load_state_dict(fedavg_params)
        img_num_per_cls_tensor = torch.tensor(img_num_per_cls).to(self.device)

        many_mask = img_num_per_cls_tensor > 100
        few_mask = img_num_per_cls_tensor <= 20
        medium_mask = ~(many_mask | few_mask)

        test_loader = DataLoader(data_test, batch_size=batch_size_test, shuffle=True, num_workers=16)
        total_corrects = 0
        many_corrects, medium_corrects, few_corrects = 0, 0, 0
        many_total, medium_total, few_total = 0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                _, outputs = self.syn_model(images)
                # _, preds = max(outputs, -1)
                outputs = torch.nn.functional.softmax(outputs, dim=-1)
                preds = torch.argmax(outputs, -1)
                # print(preds)
                total_corrects += (preds == labels).sum().item()
                # 标签分类的布尔掩码
                many_mask_labels = many_mask[labels]
                medium_mask_labels = medium_mask[labels]
                few_mask_labels = few_mask[labels]
                # 计算各分类正确的预测
                many_corrects += (preds[many_mask_labels] == labels[many_mask_labels]).sum().item()
                medium_corrects += (preds[medium_mask_labels] == labels[medium_mask_labels]).sum().item()
                few_corrects += (preds[few_mask_labels] == labels[few_mask_labels]).sum().item()

                # 记录总数
                many_total += many_mask_labels.sum().item()
                medium_total += medium_mask_labels.sum().item()
                few_total += few_mask_labels.sum().item()

        accuracy = total_corrects / len(data_test)
        many_accuracy = many_corrects / many_total if many_total > 0 else 0
        medium_accuracy = medium_corrects / medium_total if medium_total > 0 else 0
        few_accuracy = few_corrects / few_total if few_total > 0 else 0

        return accuracy, many_accuracy, medium_accuracy, few_accuracy

    def download_params(self):
        return self.syn_model.state_dict()

    def cal_norm_mean(self, c_means, c_dis):
        glo_means = dict()
        # c_means 是一个列表，其中的每个元素是一个字典，每个字典对应一个客户端，每个字典是类别向其客户端上平均特征向量的映射
        c_dis = torch.tensor(c_dis).to(self.device)  # 创建的还是一个二维张量
        total_num_per_cls = c_dis.sum(dim=0)  # 每个类别在所有客户端上数量求和
        # print(total_num_per_cls)
        for i in range(self.num_classes):
            # if total_num_per_cls[i] == 0:
            #     # 这个类别在这一轮的clients 选取中缺失了
            #     continue
            for c_idx, c_mean in enumerate(c_means):
                if i not in c_mean.keys():
                    continue
                temp = glo_means.get(i, 0)
                # normalize the local prototypes, send the direction to the server
                glo_means[i] = temp + F.normalize(c_mean[i].view(1, -1), dim=1).view(-1) * c_dis[c_idx][i]
            if glo_means.get(i) == None:
                continue
            t = glo_means[i]
            glo_means[i] = t / total_num_per_cls[i]
        return glo_means