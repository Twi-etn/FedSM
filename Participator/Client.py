from options import args_parser
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
import torch.nn as nn
import torch.nn.functional as F
from Model.Resnet8_256 import ResNet_cifar
from torchvision.transforms import transforms
from torch.utils.data.dataloader import DataLoader


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
        self.kd_criterion = KDLoss(T=args.T).to(args.device)
        self.local_model = ResNet_cifar(resnet_size=8, scaling=4,
                                        save_activations=False, group_norm_num_groups=None,
                                        freeze_bn=False, freeze_bn_affine=False, num_classes=args.num_classes).to(args.device)
        self.optimizer = SGD(self.local_model.parameters(), lr=args.lr_local_training)

    def local_train(self, args, global_params, clip_model, text_features):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()])

        self.local_model.load_state_dict(global_params)
        self.local_model.train()
        for _ in range(args.num_epochs_local_training):
            data_loader = DataLoader(dataset=self.data_client,
                                     batch_size=args.batch_size_local_training,
                                     shuffle=True, num_workers=1)
            for data_batch in data_loader:
                images, labels, clip_images = data_batch
                images, labels, clip_images = images.to(self.device), labels.to(self.device), clip_images.to(self.device)  # tensor
                images = transform_train(images)

                # compute client's output
                _, outputs = self.local_model(images)
                outputs = outputs.float()

                # get clip feature encode
                with torch.no_grad():
                    image_features = clip_model.encode_image(clip_images)
                image_features = image_features.float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                clip_logits = (100.0 * image_features @ text_features.T)  # CLIP中的点乘N方矩阵  但是乘100干嘛
                # Eq. 1
                loss1 = self.criterion(outputs, labels)
                loss2 = self.kd_criterion(outputs, clip_logits)
                loss = loss1 + args.alpha * loss2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.local_model.state_dict()
