import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn

import numpy as np
import random
import os
import time

from src.cnt_param import calculate_Params_Flops_DR
from src.model import ResNet56,ResNet18,ResNet110,ResNet20,load_model
from src.train_test import train,test,adjust_learning_rate,adjust_learning_rate_iccv

parser = argparse.ArgumentParser(description='ResNet Argument')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--dataset', type=str, choices=['cifar10','cifar100'], default='cifar10',help='Dataset')
parser.add_argument('--model', type=str, choices=['ResNet18','ResNet50','ResNet56','ResNet110'], default='ResNet56',help='model architecture')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=12346, metavar='S',
                    help='random seed (default: 12346)')
parser.add_argument('--num_output', type=int, default=10, metavar='S',
                    help='number of classes(default: 10)')
parser.add_argument('--gpu', type=int, default=[5], nargs='+', help='used gpu')

#set device to CPU or GPU
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

#----------------------------------------load data----------------------------------------#

filepath = './data'
kwargs = {'num_workers': 16, 'pin_memory': True} if use_cuda else {}

if args.dataset == "cifar10":
    print("using dataset cifar10")

    # normalize = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    # transform_train = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize,])
    # transform_test = transform=transforms.Compose([transforms.ToTensor(),normalize,])
    
    # trainset = torchvision.datasets.CIFAR10(root=filepath, train=True, download=True, transform=transform_train)
    # train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # testset = torchvision.datasets.CIFAR10(root=filepath, train=False, download=True, transform=transform_test)
    # val_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=filepath, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root=filepath, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)

    num_output = 10

elif args.dataset == "cifar100":
    print("using dataset cifar100")

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=filepath, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers)

    val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(root=filepath, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers)
    num_output = 100
else:
    print(f"Error: unsupported dataset {args.dataset}")
    raise

#----------------------------------------model architecture----------------------------------------#

if args.model == "ResNet18":
    model = ResNet18(num_classes=num_output)
if args.model == "ResNet20":
    model = ResNet20(num_classes=num_output)
elif args.model == "ResNet56":
    model = ResNet56(num_classes=num_output)
elif args.model == "ResNet110":
    model = ResNet110(num_classes=num_output)
else:
    print(f"unsupported model architecture {args.model}")

model = model.to(device)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
else:
    raise

#----------------------------------------pretrain----------------------------------------#

print("start retraining =====>")
best_val_acc = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.weight_decay,nesterov=False)
path = f"./model/pretrain/{args.model}_{args.dataset}_pretrain.pth"
#fp_rec = open(f"./{args.model}_{args.dataset}_pretrain.log",'a')

if not os.path.isfile(path):
    for epoch in range(1, args.epochs + 1):
        if args.dataset == "cifar10":
            args.lr = adjust_learning_rate_iccv(args,optimizer,epoch)
        else:
            args.lr = adjust_learning_rate(args,optimizer,epoch)
        
        start = time.time()
        # train for one epoch
        train_loss,train_acc = train(model,train_loader,device,optimizer,criterion)
        val_loss,val_acc = test(model,val_loader,device,epoch=epoch)
        if val_acc > best_val_acc: 
            best_val_acc = val_acc
            torch.save(model.state_dict(),path)
        #fp_rec.write(f"Test set: epoch: {epoch} | Learning rate: {args.lr} | Accuracy: {val_acc}\n")
        print('Time taken for epoch {} : {}\n'.format(epoch,time.time()-start))


else:   
    args.num_output = num_output
    model = load_model(args,path)
    orig_loss, orig_acc = test(model,val_loader,device)
    best_val_acc = orig_acc
#fp_rec.close()

print(f"{args.model} best pretrain model accuracy = {round(best_val_acc,4)}")
#print('Original accuracy {}, compressed model accuracy {}, accuracy drop {}'.format(orig_acc,best_val_acc,orig_acc-best_val_acc))

ori_param,ori_flops = calculate_Params_Flops_DR(model.cpu(),input_res=32)
print('  - Number of params: %.4fM\n' % (ori_param ))
print('  - Number of FLOPs: %.4fM\n' % (ori_flops ))
