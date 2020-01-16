from __future__ import print_function
import os
import errno

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torchvision


class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def synchronize_between_processes(self, device):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.sum], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.sum = t[1]


def accuracy(output, target, topk=(1,)):
    '''Computes the accuracy over the k top predictions for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)


def initialize_model(num_classes, args):
    print('Loading model')
    model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, progress=False)
    
    print('Change model settings')
    if 'resnet' in args.model or 'resnext' in args.model:
        ''' resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
        '''
        set_parameter_requires_grad(model, args)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif 'alexnet' in args.model:
        ''' alexnet
        '''
        set_parameter_requires_grad(model, args)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif 'vgg' in args.model:
        ''' vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
        '''
        set_parameter_requires_grad(model, args)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif 'squeezenet' in args.model:
        ''' squeezenet1_0, squeezenet1_1
        '''
        set_parameter_requires_grad(model, args)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model.num_classes = num_classes
        input_size = 224

    elif 'densenet' in args.model:
        ''' densenet121, densenet169, densenet201, densenet161
        '''
        set_parameter_requires_grad(model, args)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif 'inception' in args.model:
        ''' inception_v3
        Be careful, expects (299,299) sized images and has auxiliary output
        '''
        set_parameter_requires_grad(model, args)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    
    elif 'mobilenet' in args.model:
        ''' mobilenet_v2
        '''
        set_parameter_requires_grad(model, args)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif 'shufflenet' in args.model:
        ''' shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0
        '''
        set_parameter_requires_grad(model, args)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif 'mnasnet' in args.model:
        ''' mnasnet0_5, mnasnet0_75, mnasnet1_0, mnasnet1_3
        '''
        set_parameter_requires_grad(model, args)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
    
    # distribute model
    if not args.distributed:
        model = torch.nn.DataParallel(model).to(args.device)
    else:
        model.to(args.device)
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Gather the parameters to be optimized/updated in this run.
    print('Get relevant model parameters')
    params_to_update = get_params_to_update(model, args)

    return model, input_size, params_to_update


def set_parameter_requires_grad(model, args):
    if args.finetuning:
        for param in model.parameters():
            param.requires_grad = False


def get_params_to_update(model, args):
    params_to_update = []
    if args.finetuning:
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model.parameters()
    return params_to_update