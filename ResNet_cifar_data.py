import argparse
import numpy as np
import pandas as pd
import torch
import random
import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

from src.NASWOT import naswot
from src.model import load_model
from src.prune_model import prune_model
from src.cluster_model import cluster_model
from src.cnt_param import calculate_Params_Flops_DR
from src.train_test import train_DR,test_DR,adjust_learning_rate,adjust_learning_rate_iccv

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--dataset', type=str, choices=['cifar10','cifar100'], default='cifar10',help='Dataset')
parser.add_argument('--model', type=str, choices=['ResNet18','ResNet50','ResNet56','ResNet110'], default='ResNet56',help='model architecture')
parser.add_argument('-j', '--workers', default=24, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', type=int, default=605, metavar='S',
                    help='random seed (default: 12346)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_output', type=int, default=10, metavar='S',
                    help='number of classes(default: 10)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gpu', type=int, default=[5], nargs='+', help='used gpu')
parser.add_argument('--compress-rate', type=float, default=0.8, help='compress rate of hybrid pruning')
parser.add_argument('--cluster-threshold', type=float, default=2, help='threshold of cluster pruning')
parser.add_argument('--filter-threshold', type=float, default=0.7, help='threshold of filter pruning')
parser.add_argument('--min-filter-list', default = None, nargs='+', type=int, help='lowerbound of filters of each layers')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#set all seeds for reproducability
def set_random_seed(seed):    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed(args.seed)

filepath = './data'

if args.dataset == "cifar10":
    print("using dataset cifar10")

    normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize,])
    transform_test = transforms.Compose([transforms.ToTensor(),normalize,])
    train_set = torchvision.datasets.CIFAR10(root=filepath, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=filepath, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    args.num_output = 10

elif args.dataset == "cifar100":
    print("using dataset cifar100")

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize,])
    transform_test = transform=transforms.Compose([transforms.ToTensor(),normalize,])
    
    trainset = torchvision.datasets.CIFAR100(root=filepath, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.CIFAR100(root=filepath, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    args.num_output = 100
else:
    print(f"Error: unsupported dataset {args.dataset}")
    raise


#--------------------------------load model--------------------------------#
print("Load model ======> \n")
path = f'./model/pretrain/{args.model}_{args.dataset}_pretrain.pth'
set_random_seed(args.seed)
if args.model == 'ResNet56':
    print("loading pretrained model ResNet56")
    pretrained_model = load_model(args,path)
    conv_layers = {3:0,9:0,15:0,21:0,27:0,33:0,39:0,45:0,51:0,57:0,65:0,71:0,77:0,83:0,89:0,95:0,101:0,107:0,113:0,121:0,127:0,133:0,139:0,145:0,151:0,157:0,163:0,}
    layer_num = 3

elif args.model == 'ResNet18':
    print("loading pretrained model ResNet18")
    pretrained_model = load_model(args,path)
    conv_layers = {3:0,10:0,17:0,25:0,32:0,40:0,47:0,55:0}
    layer_num = 4

elif args.model == 'ResNet50':
    print("loading pretrained model: ResNet50")
    pretrained_model = load_model(args,path)
    conv_layers = {3:0,6:0,14:0,17:0,24:0,27:0,34:0,37:0,45:0,48:0,55:0,58:0,65:0,68:0,75:0,78:0,86:0,89:0,96:0,99:0,106:0,109:0,116:0,119:0,126:0,129:0,136:0,139:0,147:0,150:0,157:0,160:0,}
    layer_num = 4

elif args.model == "ResNet110":
    print("loading pretrained model ResNet110")
    pretrained_model = load_model(args,path)
    conv_layers = {3:0,9:0,15:0,21:0,27:0,33:0,39:0,45:0,51:0,57:0,63:0,69:0,75:0,81:0,87:0,93:0,99:0,105:0,111:0,119:0,125:0,131:0,137:0,143:0,149:0,155:0,161:0,167:0,173:0,179:0,185:0,191:0,197:0,203:0,209:0,215:0,221:0,229:0,235:0,241:0,247:0,253:0,259:0,265:0,271:0,277:0,283:0,289:0,295:0,301:0,307:0,313:0,319:0,325:0,}
    layer_num = 3

else:
    print(f"unsupported model architecture {args.model}")
    raise

ori_param,ori_flops = calculate_Params_Flops_DR(pretrained_model.cpu(),input_res=32,verbose=False)
pretrained_model.cuda()
#orig_loss, orig_acc = test(args,pretrained_model,device,val_loader)
orig_loss, orig_acc = test_DR(pretrained_model,val_loader,device)
print(f"Origin model accuracy = {orig_acc}")

min_filter_list = np.array([2]*27).reshape(-1)
cluster_threshold = args.cluster_threshold

start = time.time()

for norm_threshold in np.arange(args.filter_threshold,1.5,0.01):
    resnet_pruned = copy.deepcopy(pretrained_model)
    pruning_args = {
        'prune_layers' : conv_layers,
        'criterion' : 'l2',
        'use_bias' : False,
        'conv_feature_size' : 1,
        'pruning_mode' : 'nonuniform',
        'norm_threshold' : norm_threshold,
        'compress_rate' : 0,
        'min_filter' : 0,
        'min_filter_list' : min_filter_list,
    }

    resnet_pruned.to("cpu")
    model_modifier = prune_model(resnet_pruned,pruning_args)
    resnet_pruned = model_modifier.prune_model()#[int(nodes*drop_percentage) for nodes in [500,300]])
        
    hp_param,hp_flops = calculate_Params_Flops_DR(resnet_pruned.cpu(),input_res=32,verbose=False)
    if 1-hp_flops/ori_flops >= args.compress_rate-0.01:
        print(1-hp_flops/ori_flops)
        print("finish searching")
        break

    resnet_flop = copy.deepcopy(resnet_pruned)
    cluster_args = {
        'cluster_layers' : conv_layers,
        'conv_feature_size' : 1,
        'features' : 'both',
        'channel_reduction' : 'fro',
        'use_bias' : False,
        'reshape_exists' : True,
        'linkage_method' : 'ward',
        'distance_metric' : 'euclidean',
        'cluster_mode' : 'nonuniform',
        'cluster_criterion' : 'hierarchical',
        'distance_threshold' : cluster_threshold,
        'merge_criterion' : 'max_l2_norm', 
        'compress_rate' : 0,   
        'verbose' : False,
        'min_filter' : 0,
        'min_filter_list' : min_filter_list,
    }
    resnet_pruned.to("cpu")
    model_modifier = cluster_model(resnet_pruned,cluster_args)
    resnet56_cluster_pruned = model_modifier.cluster_model()#[int(nodes*drop_percentage) for nodes in [500,300]])

    while True:
        resnet_flop = copy.deepcopy(resnet56_cluster_pruned)
        #print('--- stats for compressed model ---')
        hp_param,hp_flops = calculate_Params_Flops_DR(resnet_flop.cpu(),input_res=32,verbose=False)
        #fp_flops = print_model_param_flops(resnet_flop.cpu(),input_res=32)
        print((1-hp_flops/ori_flops))
        if 1-hp_flops/ori_flops <= args.compress_rate-0.01:
            resnet_pruned = copy.deepcopy(resnet56_cluster_pruned)
            break
        else:
            cluster_threshold-=0.01
            #print(f"cluster_threshold = {cluster_threshold}")
            cluster_args['distance_threshold'] = cluster_threshold
            resnet_pruned.to("cpu")
            model_modifier = cluster_model(resnet_pruned,cluster_args)
            resnet56_cluster_pruned = model_modifier.cluster_model()#[int(nodes*drop_percentage) for nodes in [500,300]])

    # NASWOT scoring
    scores=0
    resnet_pruned.cuda()
    model_naswot = copy.deepcopy(resnet_pruned)
    scores = naswot(model_naswot,device,val_loader,args)

    # Retraining
    # model = copy.deepcopy(resnet_pruned)
    # args.lr = 0.1
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=False)
    # path = "./test.pth"

    # print("Start Retraining ======> \n")
    best_val_acc = 0
    # for epoch in range(1, args.epochs+1):
    #     args.lr = adjust_learning_rate_iccv(args,optimizer,epoch)
    #     train_loss,train_acc = train_DR(model,train_loader,device,optimizer,criterion,epoch)
    #     val_loss,val_acc = test_DR(model,val_loader,device,epoch=epoch)
    #     if val_acc > best_val_acc:  
    #         # torch.save(resnet_pruned, path)            
    #         best_val_acc = val_acc 

#     csv_path = f"./csv/additional_data.csv"
#     if os.path.exists(csv_path):
#         df = pd.read_csv(csv_path)
#         new_df = pd.DataFrame({
#             "score" : [scores],
#             "acc" : [best_val_acc],
#             "cluster_pruning_threshold" : [cluster_threshold],
#             "filter_pruning_threshold" : [norm_threshold],
#             "flops_compress_rate" : [args.compress_rate],
#         })
#         df = pd.concat([df,new_df],axis=0)
#     else:
#         df = pd.DataFrame({
#             "score" : [scores],
#             "acc" : [best_val_acc],
#             "cluster_pruning_threshold" : [cluster_threshold],
#             "filter_pruning_threshold" : [norm_threshold],
#             "flops_compress_rate" : [args.compress_rate],
#         })
#     df.to_csv(csv_path,index=False)

#     fp = open(f"./log/additional_data.log",'a')
#     fp.write("\n------------------------------------------------------------\n")
#     fp.write(f"compress rate = {args.compress_rate}\n")
#     fp.write(f"cluster threshold = {cluster_threshold}\n")
#     fp.write(f"filter threshold = {norm_threshold}\n")
#     # fp.write(f"min filter = {min_filter_list}\n")
#     fp.write(f"NASOWT score = {scores}\n")
#     fp.write('Original accuracy {}, compressed model accuracy {}, accuracy drop {}\n'.format(orig_acc,best_val_acc,orig_acc-best_val_acc))
#     fp.close()


# fp = open(f"./log/additional_data.log",'a')
end = time.time()
# fp.write("\n------------------------------------------------------------\n")
# fp.write(f"total time = {end-start}sec\n")
# fp.close()

print(f"total time = {end-start}sec")