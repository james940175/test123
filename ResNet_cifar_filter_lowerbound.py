import argparse
import numpy as np
import torch
import random
import copy
import os
import pandas as pd

import torch
import torchvision
from torchvision import datasets, transforms

from src.prune_model import prune_model
from src.cluster_model import cluster_model
from src.model import load_model
from src.NASWOT import naswot
from src.cnt_param import calculate_Params_Flops_DR
from src.train_test import test_DR

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--dataset', type=str, choices=['cifar10','cifar100'], default='cifar10',help='Dataset')
parser.add_argument('--model', type=str, choices=['ResNet18','ResNet50','ResNet56','ResNet110'], default='ResNet56',help='model architecture')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--seed', type=int, default=605, metavar='S',
                    help='random seed (default: 12346)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_output', type=int, default=10, metavar='S',
                    help='number of classes(default: 10)')
parser.add_argument('--gpu', type=int, default=[5], nargs='+', help='used gpu')
parser.add_argument('--compress-rate', type=float, default=0.5, help='compress rate of hybrid pruning')
parser.add_argument('--cluster-threshold', type=float, default=10, help='threshold of cluster pruning')
parser.add_argument('--filter-threshold', type=float, default=10, help='threshold of filter pruning')

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

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
    # conv_layers = {3:0,10:0,17:0,24:0,31:0,38:0,45:0,52:0,59:0,66:0,74:0,81:0,88:0,95:0,102:0,109:0,116:0,123:0,130:0,138:0,145:0,152:0,159:0,166:0,173:0,180:0,187:0}
    conv_layers = {3:0,9:0,15:0,21:0,27:0,33:0,39:0,45:0,51:0,57:0,65:0,71:0,77:0,83:0,89:0,95:0,101:0,107:0,113:0,121:0,127:0,133:0,139:0,145:0,151:0,157:0,163:0,}


elif args.model == 'ResNet18':
    print("loading pretrained model ResNet18")
    pretrained_model = load_model(args,path)
    conv_layers = {3:0,10:0,17:0,25:0,32:0,40:0,47:0,55:0}

elif args.model == 'ResNet50':
    print("loading pretrained model: ResNet50")
    pretrained_model = load_model(args,path)
    conv_layers = {3:0,6:0,14:0,17:0,24:0,27:0,34:0,37:0,45:0,48:0,55:0,58:0,65:0,68:0,75:0,78:0,86:0,89:0,96:0,99:0,106:0,109:0,116:0,119:0,126:0,129:0,136:0,139:0,147:0,150:0,157:0,160:0,}

elif args.model == "ResNet110":
    print("loading pretrained model ResNet110")
    pretrained_model = load_model(args,path)
    # conv_layers = {3:0,10:0,17:0,24:0,31:0,38:0,45:0,52:0,59:0,66:0,73:0,80:0,87:0,94:0,101:0,108:0,115:0,122:0,129:0,137:0,144:0,151:0,158:0,165:0,172:0,179:0,186:0,193:0,200:0,207:0,214:0,221:0,228:0,235:0,242:0,249:0,256:0,264:0,271:0,278:0,285:0,292:0,299:0,306:0,313:0,320:0,327:0,334:0,341:0,348:0,355:0,362:0,369:0,376:0,}
    conv_layers = {3:0,9:0,15:0,21:0,27:0,33:0,39:0,45:0,51:0,57:0,63:0,69:0,75:0,81:0,87:0,93:0,99:0,105:0,111:0,119:0,125:0,131:0,137:0,143:0,149:0,155:0,161:0,167:0,173:0,179:0,185:0,191:0,197:0,203:0,209:0,215:0,221:0,229:0,235:0,241:0,247:0,253:0,259:0,265:0,271:0,277:0,283:0,289:0,295:0,301:0,307:0,313:0,319:0,325:0,}


else:
    print(f"unsupported model architecture {args.model}")
    raise

ori_param,ori_flops = calculate_Params_Flops_DR(pretrained_model.cpu(),input_res=32,verbose=False)
pretrained_model.cuda()
#orig_loss, orig_acc = test(args,pretrained_model,device,val_loader)
orig_loss, orig_acc = test_DR(pretrained_model,val_loader,device)
print(f"Origin model accuracy = {orig_acc}")

resnet_pruned = copy.deepcopy(pretrained_model)
#--------------------------------setting pruning parameters--------------------------------#

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
    'distance_threshold' : args.cluster_threshold,
    'merge_criterion' : 'max_l2_norm', 
    'compress_rate' : args.compress_rate,   
    'verbose' : False,
    'min_filter' : 0,
    'min_filter_list' : 0,
}

pruning_args = {
    'prune_layers' : conv_layers,
    'criterion' : 'l2',
    'use_bias' : False,
    'conv_feature_size' : 1,
    'pruning_mode' : 'nonuniform',
    'norm_threshold' : args.filter_threshold,
    'compress_rate' : args.compress_rate,
    'min_filter' : 0,
    'min_filter_list' : 0,
}

#--------------------------------searching best filter upperbound with naswot--------------------------------#

if args.model == "ResNet56" or args.model == "ResNet110":
    for layer1_pruned_rate in np.arange(args.compress_rate-0.2,min(args.compress_rate+0.26,0.9),0.05):
        for layer2_pruned_rate in np.arange(args.compress_rate-0.2,min(args.compress_rate+0.26,0.9),0.05):
            for layer3_pruned_rate in np.arange(args.compress_rate-0.2,min(args.compress_rate+0.26,0.9),0.05):
                if args.model == "ResNet56":
                    min_filter_list = np.array([[int(16*(1-layer1_pruned_rate))]*9,[int(32*(1-layer2_pruned_rate))]*9,[int(64*(1-layer3_pruned_rate))]*9]).reshape(-1)
                else:
                    min_filter_list = np.array([[int(16*(1-layer1_pruned_rate))]*18,[int(32*(1-layer2_pruned_rate))]*18,[int(64*(1-layer3_pruned_rate))]*18]).reshape(-1)
                
                resnet_pruned = copy.deepcopy(pretrained_model)
                resnet_pruned.to("cpu")

                pruning_args['min_filter_list'] = min_filter_list
                model_modifier = prune_model(resnet_pruned,pruning_args)
                resnet_pruned_f = model_modifier.prune_model()

                cluster_args['min_filter_list'] = min_filter_list
                model_modifier = cluster_model(resnet_pruned,cluster_args)
                resnet_pruned_c = model_modifier.cluster_model()

                hp_param,hp_flops = calculate_Params_Flops_DR(resnet_pruned_c.cpu(),input_res=32,verbose=False)
                print(f"compress rate: {round(1-hp_flops/ori_flops,2)}")

                if ((1-hp_flops/ori_flops) >= args.compress_rate+0.05 and (1-hp_flops/ori_flops) <= args.compress_rate+0.1)\
                    or (args.compress_rate==0.8 and (1-hp_flops/ori_flops)>=args.compress_rate):
                    scores_f = naswot(resnet_pruned_f,device,val_loader,args)
                    scores_c = naswot(resnet_pruned_c,device,val_loader,args)
                    scores = scores_c + scores_f
                    csv_path = f"./csv/{args.model}_{args.dataset}_min_filter_{int(args.compress_rate*100)}.csv"
                    if os.path.exists(csv_path):
                        df = pd.read_csv(csv_path)
                        new_df = pd.DataFrame({
                            "score" : [scores],
                            "layer_1" : [int(16*(1-layer1_pruned_rate))],
                            "layer_2" : [int(32*(1-layer2_pruned_rate))],
                            "layer_3" : [int(64*(1-layer3_pruned_rate))],
                            "flops_compress_rate" : [round(1-hp_flops/ori_flops,2)],
                            "param compress rate" : [round(1-hp_param/ori_param,2)]
                        })
                        df = pd.concat([df,new_df],axis=0)
                    else:
                        df = pd.DataFrame({
                            "score" : [scores],
                            "layer_1" : [int(16*(1-layer1_pruned_rate))],
                            "layer_2" : [int(32*(1-layer2_pruned_rate))],
                            "layer_3" : [int(64*(1-layer3_pruned_rate))],
                            "flops_compress_rate" : [round(1-hp_flops/ori_flops,2)],
                            "param compress rate" : [round(1-hp_param/ori_param,2)]
                        }) 
                    df.to_csv(csv_path,index=False)

else:
    for layer1_pruned_rate in np.arange(args.compress_rate-0.15,min(args.compress_rate+0.16,0.9),0.05):
        for layer2_pruned_rate in np.arange(args.compress_rate-0.15,min(args.compress_rate+0.16,0.9),0.05):
            for layer3_pruned_rate in np.arange(args.compress_rate-0.15,min(args.compress_rate+0.16,0.9),0.05):
                for layer4_pruned_rate in np.arange(args.compress_rate-0.15,min(args.compress_rate+0.16,0.9),0.05):
                    if True:#layer1_pruned_rate < args.compress_rate and layer2_pruned_rate < args.compress_rate and layer3_pruned_rate < args.compress_rate and layer4_pruned_rate < args.compress_rate:
                        if args.model == "ResNet18":
                            min_filter_list = np.array([[int(64*(1-layer1_pruned_rate))]*2,[int(128*(1-layer2_pruned_rate))]*2,[int(256*(1-layer3_pruned_rate))]*2,[int(512*(1-layer1_pruned_rate))]*2]).reshape(-1)
                        else:
                            min_filter_list = np.array([[int(64*(1-layer1_pruned_rate))]*6+[int(128*(1-layer2_pruned_rate))]*8+[int(256*(1-layer3_pruned_rate))]*12+[int(512*(1-layer1_pruned_rate))]*6]).reshape(-1)

                        resnet_pruned = copy.deepcopy(pretrained_model)
                        resnet_pruned.to("cpu")

                        pruning_args['min_filter_list'] = min_filter_list
                        model_modifier = prune_model(resnet_pruned,pruning_args)
                        resnet_pruned_f = model_modifier.prune_model()

                        cluster_args['min_filter_list'] = min_filter_list
                        model_modifier = cluster_model(resnet_pruned,cluster_args)
                        resnet_pruned_c = model_modifier.cluster_model()

                        hp_param,hp_flops = calculate_Params_Flops_DR(resnet_pruned_c.cpu(),input_res=32,verbose=False)
                        print(f"compress rate: {round(1-hp_flops/ori_flops,2)}")

                        if ((1-hp_flops/ori_flops) >= args.compress_rate+0.02 and (1-hp_flops/ori_flops) <= args.compress_rate+0.1)\
                            or (args.compress_rate==0.8 and (1-hp_flops/ori_flops)>=args.compress_rate):
                            scores_f = naswot(resnet_pruned_f,device,val_loader,args)
                            scores_c = naswot(resnet_pruned_c,device,val_loader,args)
                            scores = scores_c + scores_f
                            csv_path = f"./csv/{args.model}_{args.dataset}_min_filter_{int(args.compress_rate*100)}.csv"
                            if os.path.exists(csv_path):
                                df = pd.read_csv(csv_path)
                                new_df = pd.DataFrame({
                                    "score" : [scores],
                                    "layer_1" : [int(64*(1-layer1_pruned_rate))],
                                    "layer_2" : [int(128*(1-layer2_pruned_rate))],
                                    "layer_3" : [int(256*(1-layer3_pruned_rate))],
                                    "layer_4" : [int(512*(1-layer4_pruned_rate))],
                                    "flops_compress_rate" : [round(1-hp_flops/ori_flops,2)],
                                    "param compress rate" : [round(1-hp_param/ori_param,2)]
                                })
                                df = pd.concat([df,new_df],axis=0)
                            else:
                                df = pd.DataFrame({
                                    "score" : [scores],
                                    "layer_1" : [int(64*(1-layer1_pruned_rate))],
                                    "layer_2" : [int(128*(1-layer2_pruned_rate))],
                                    "layer_3" : [int(256*(1-layer3_pruned_rate))],
                                    "layer_4" : [int(512*(1-layer4_pruned_rate))],
                                    "flops_compress_rate" : [round(1-hp_flops/ori_flops,2)],
                                    "param compress rate" : [round(1-hp_param/ori_param,2)]
                                }) 
                            df.to_csv(csv_path,index=False)