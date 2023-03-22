import os
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math

# Without integer bit
def quant_signed_1(original, bit=6):
    bit = bit - 2
    original = original.clamp(max=1.9375,min=-1.9375)
    return ((original * (2**bit)).int()) / (2**bit)

def quant_signed_0(original, bit=16):
    bit = bit - 1
    original = original.clamp(max=0.96875,min=-0.96875)
    return ((original * (2**bit)).int()) / (2**bit)

def get_mnist(batch_size=256, distributed=None, workers=2):

    
    if distributed:
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        trainsampler = None
    

    print("Loading MNIST data ... ")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5), (0.5))])
    trainset = torchvision.datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(trainsampler is None), 
                num_workers=workers, pin_memory=True, sampler=trainsampler)

    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5), (0.5))])
    testset = torchvision.datasets.MNIST(root='./MNIST', train=False, download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=workers, pin_memory=True)

    return trainloader, trainsampler, testloader


def get_cifar10(batch_size=256, distributed=None, workers=2):
    print("Loading cifar10 data ... ")

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor()])
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.CIFAR10(root='./Cifar10', train=True, download=True, transform=transform)

    if distributed:
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        trainsampler = None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(trainsampler is None), 
                  num_workers=workers, pin_memory=True, sampler=trainsampler)

    val_transform = transforms.Compose([transforms.ToTensor()])

    testset = torchvision.datasets.CIFAR10(root='./Cifar10', train=False, download=True, transform=val_transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=workers, pin_memory=True)
    
    return trainloader, trainsampler, testloader


def get_cifar100(batch_size=256, distributed=None, workers=2):
    print("Loading cifar100 data ... ")

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.CIFAR100(root='/tmpssd/pabillam/data', train=True, download=True, transform=transform)

    if distributed:
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    else:
        trainsampler = None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(trainsampler is None), 
                  num_workers=workers, pin_memory=True, sampler=trainsampler)

    val_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    testset = torchvision.datasets.CIFAR100(root='/tmpssd/pabillam/data', train=False, download=True, transform=val_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=workers, pin_memory=True)
    
    return trainloader, trainsampler, testloader

def get_imagenet(data_dir, batch_size=128, distributed=None, workers=2):
    print("Loading imagenet data ... ")

    if distributed:
        os.system("cd ..; source ./prepare_imagenet_dataset.sh; cd src/")

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=100, shuffle=False,
        num_workers=workers, pin_memory=True)

    return train_loader, train_sampler, val_loader
