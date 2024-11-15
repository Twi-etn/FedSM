import argparse
import os
from Dataset.param_aug import ParamDiffAug


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    # CReFF
    parser.add_argument('--clip_base_model', type=str, default='ViT-B/32')
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--path_imagenet', type=str, default=os.path.join(path_dir, 'data/ImageNet'))
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)
    parser.add_argument('--num_rounds', type=int, default=200)
    parser.add_argument('--crt_feat_num', type=int, default=100)  #
    parser.add_argument('--num_epochs_local_training', type=int, default=10)  #
    parser.add_argument('--batch_size_local_training', type=int, default=32)
    parser.add_argument('--crt_batch_size', type=int, default=256)
    parser.add_argument('--crt_ep', type=int, default=90)
    parser.add_argument('--local_bal_epochs', type=int, default=60)  #
    # parser.add_argument('--batch_real', type=int, default=64)
    parser.add_argument('--batch_size_test', type=int, default=500)
    parser.add_argument('--lr_local_training', type=float, default=0.1)
    parser.add_argument('--lr_retrain', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--non_iid_alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--save_path', type=str, default=os.path.join(path_dir, 'result/'))
    parser.add_argument('--kd_type', type=str, default='KD')
    parser.add_argument('--method', type=str, default='DSA', help='DC/DSA')
    parser.add_argument('--uniform_left', type=float, default=0.35)
    parser.add_argument('--uniform_right', type=float, default=0.95)  # eq.18 alpha

    # CLIP2FL
    parser.add_argument('--scale', default=1, type=int)
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10/cifar100/imagenet')
    # parser.add_argument('--beta', default=0.998, type=float)
    parser.add_argument('--result_save', type=str, default='results')
    parser.add_argument('--T', default=3.0, type=float, help='KD loss Input the temperature: default(3.0)')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--kd_alpha', default=1.0, type=float, help='the hypeparameter for KD loss')
    parser.add_argument('--cos_alpha', default=1.0, type=float, help='the hypeparameter for Internal KD loss')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # FedProx
    parser.add_argument('--hint', type=str, default='')
    args = parser.parse_args()

    return args
# nohup sh -c 'python main.py --seed=5 --gpu=1 > /dev/null 2>&1; python main.py --seed=6 --gpu=2 > /dev/null 2>&1; python main.py --seed=8 --gpu=3 > /dev/null 2>&1; python main.py --seed=9 --gpu=4 > /dev/null 2>&1;' &
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100 --local_bal_epochs=10 --crt_ep=20 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100 --local_bal_epochs=10 --crt_ep=30 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100 --local_bal_epochs=10 --crt_ep=40 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100 --local_bal_epochs=10 --crt_ep=50 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100 --local_bal_epochs=10 --crt_ep=60 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100 --local_bal_epochs=10 --crt_ep=70 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100 --local_bal_epochs=10 --crt_ep=80 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.2 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.2 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.2 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.2 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.2 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.2 & \
#
#
#
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.35 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.35 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.35 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.35 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.35 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.35 & \
#
#
# python main.py --seed=7 --imb_factor=0.1 --gpu=5 --dataset='cifar10'  --num_classes=10  --uniform_left=0.5 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=5 --dataset='cifar10'  --num_classes=10  --uniform_left=0.5 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=5 --dataset='cifar10'  --num_classes=10  --uniform_left=0.5 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=5 --dataset='cifar100'  --num_classes=100  --uniform_left=0.5 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=5 --dataset='cifar100'  --num_classes=100  --uniform_left=0.5 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=5 --dataset='cifar100'  --num_classes=100  --uniform_left=0.5 & \


# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.65 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.65 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.65 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.65 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.65 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.65 & \
#
#
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.8 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.8 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=4 --dataset='cifar10'  --num_classes=10  --uniform_left=0.8 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.8 & \
# python main.py --seed=7 --imb_factor=0.02 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.8 & \
# python main.py --seed=7 --imb_factor=0.01 --gpu=4 --dataset='cifar100'  --num_classes=100  --uniform_left=0.8 & \
#
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100 --local_bal_epochs=10 --crt_ep=80 & \
# python main.py --seed=7 --imb_factor=0.1 --gpu=4 --dataset='cifar100'  --num_classes=100 --local_bal_epochs=10 --crt_ep=80 & \
