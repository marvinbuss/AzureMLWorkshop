from __future__ import print_function
import argparse
import time
import os
import onnx

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from torchvision import transforms

import utils

from azureml.core import Dataset, Run
run = Run.get_context() # get the Azure ML run object


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    start_time = time.time()

    # Create objects for tracking parameters
    img_processing = utils.AverageMeter()
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    acc1 = utils.AverageMeter()
    acc5 = utils.AverageMeter()
    lr = utils.AverageMeter()
    
    # Put model in train mode 
    model.train()

    for i, (image, target) in enumerate(data_loader):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure accuracy and loss
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        losses.update(val=loss.item(), n=batch_size)
        acc1.update(val=acc1.item(), n=batch_size)
        acc5.update(val=acc5.item(), n=batch_size)
        batch_time.update(val=time.time() - start_time)
        img_processing.update(val=batch_size / (time.time() - start_time))
        lr.update(val=optimizer.param_groups[0]["lr"])

        # Log metrics to Azure ML
        run.log(name='train_loss', value=loss.item())
        run.log(name='train_acc1', value=acc1.item())
        run.log(name='train_acc5', value=acc5.item())
        run.log(name='train_imgs/s', value=batch_size / (time.time() - start_time))
        run.log(name='train_lr', value=optimizer.param_groups[0]["lr"])
    
    # Synchronize AverageMeters between processes
    #img_processing.synchronize_between_processes(device=device)
    #losses.synchronize_between_processes(device=device)
    #acc1.synchronize_between_processes(device=device)
    #acc5.synchronize_between_processes(device=device)

    # Log metrics to Azure ML
    run.log(name='train_loss_avg', value=losses.avg)
    run.log(name='train_acc1_avg', value=acc1.avg)
    run.log(name='train_acc5_avg', value=acc5.avg)
    run.log(name='train_imgs/s_avg', value=img_processing.avg)

    print('[Training] Epoch {epoch} Acc@1 {acc1.avg:.3f} Acc@5 {acc5.global_avg:.3f} Loss {loss.avg:.3f} Took {time}'
          .format(epoch=epoch, acc1=acc1, acc5=acc5, loss=loss, time=(time.time()-start_time)))


def evaluate(model, criterion, data_loader, device, epoch):
    # Create objects for tracking parameters
    losses = utils.AverageMeter()
    acc1 = utils.AverageMeter()
    acc5 = utils.AverageMeter()
    
    # Put model in eval mode 
    model.eval()

    with torch.no_grad():
        for i, (image, target) in enumerate(data_loader):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            # Measure accuracy and loss
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            losses.update(val=loss.item(), n=batch_size)
            acc1.update(val=acc1.item(), n=batch_size)
            acc5.update(val=acc5.item(), n=batch_size)

            # Log metrics to Azure ML
            run.log(name='val_loss', value=loss.item())
            run.log(name='val_acc1', value=acc1.item())
            run.log(name='val_acc5', value=acc5.item())
    
    # Synchronize AverageMeters between processes
    #losses.synchronize_between_processes(device=device)
    #acc1.synchronize_between_processes(device=device)
    #acc5.synchronize_between_processes(device=device)

    # Log metrics to Azure ML
    run.log(name='val_loss_avg', value=losses.avg)
    run.log(name='val_acc1_avg', value=acc1.avg)
    run.log(name='val_acc5_avg', value=acc5.avg)

    print('[Validation] Epoch {epoch} Acc@1 {acc1.avg:.3f} Acc@5 {acc5.global_avg:.3f} Loss {loss.avg:.3f}'
          .format(epoch=epoch, acc1=acc1, acc5=acc5, loss=loss))


def load_data(traindir, valdir, distributed, input_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Load training data
    print('Loading training data')
    st = time.time()
    dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    print('Took {0}'.format(time.time() - st))

    # Load validation data
    print('Loading validation data')
    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(input_size + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))
    
    # Create data sampler
    print('Creating data sampler')
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    print(args)

    # Create output directory
    print('Create output dir and paths')
    if args.output_dir:
        utils.mkdir(path=os.path.join('.', args.output_dir))
    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')
    num_classes = len(train_dir)
    
    # Initialize distributed mode
    print('Initialize distributed mode')
    utils.init_distributed_mode(args=args):

    # Create model
    print('Create model')
    model, input_size, params_to_update = utils.initialize_model(num_classes=num_classes, args=args)
    if not args.distributed:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model)
    
    # Load data
    print('Load data')
    dataset, dataset_test, train_sampler, test_sampler = load_data(traindir=train_dir,
                                                                   valdir=val_dir,
                                                                   distributed=args.distributed,
                                                                   input_size=input_size)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=args.batch_size,
                                              sampler=train_sampler,
                                              num_workers=args.workers,
                                              pin_memory=True)
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=args.batch_size,
                                                   sampler=test_sampler, 
                                                   num_workers=args.workers,
                                                   pin_memory=True)
    
    # Create criterion
    print('Creating criterion')
    criterion = nn.CrossEntropyLoss().to(device)

    # Create optimizer
    print('Creating optimizer')
    optimizer = torch.optim.SGD(
        params_to_update,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    
    # Create lr scheduler
    print('Creating lr scheduler')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma)
    
    # Resume from checkpoint
    if args.resume:
        print('Resuming from checkpoint {0}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.module.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
    
    if args.test_only:
        # Test script only
        print('Testing only')
        evaluate(model, criterion, data_loader_test, device=device, epoch=0)

    else:
        # Train model
        print('Start training')
        start_time = time.time()

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Train one epoch
            train_one_epoch(model, criterion, optimizer, data_loader, device, epoch)
            lr_scheduler.step()

            # Evauluate on val data
            evaluate(model, criterion, data_loader_test, device, epoch)

            # Save checkpoints after each epoch
            if args.output_dir:
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args}
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
                utils.save_on_master(
                    checkpoint,
                    os.path.join(args.output_dir, 'checkpoint.pth'))

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
    
    # Save model as pt and ONNX
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    dummy_input = torch.randn(args.batch_size, 3, input_size, input_size, requires_grad=True, device=device)
    torch.save(model, os.path.join(args.output_dir, 'model.pt'))
    torch.onnx.export(model,
                      dummy_input,
                      os.path.join(args.output_dir, 'model.onnx'),
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      verbose=True,
                      input_names = ['input'],
                      output_names = ['output'],
                      dynamic_axes={'input' : {0 : 'batch_size'},
                                    'output' : {0 : 'batch_size'}})
    
    # Check ONNX model
    print('Checking ONNX model')
    onnx_model = onnx.load(os.path.join(args.output_dir, 'model.onnx'))
    onnx.checker.check_model(onnx_model)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    
    # Training parameters
    parser.add_argument('--data-path', dest='data_path', default='/tmp/dataset/',
                        help='dataset path')
    parser.add_argument('--dataset-name', dest='dataset_name', default=None,
                        help='dataset name')
    parser.add_argument('--model', dest='model', default='resnet18',
                        help='model name')
    parser.add_argument('--device', dest='device', default='cuda',
                        help='device')
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=32, type=int,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', dest='epochs', default=10, type=int, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('-j', '--workers', dest='workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', dest='lr', default=0.01, type=float,
                        help='initial learning rate (default 0.01)')
    parser.add_argument('--momentum', dest='momentum', default=0.9, type=float, metavar='M',
                        help='SGD momentum (default 0.9)')
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-step-size', dest='lr_step_size', default=30, type=int,
                        help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', dest='lr_gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--output-dir', dest='output_dir', default='outputs',
                        help='path where to save')
    parser.add_argument('--resume', dest='resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start-epoch', dest='start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
   parser.add_argument('--test-only', dest='test_only', action='store_true',
                        help='Only test the model')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='Use pre-trained models from torchvision')
    parser.add_argument('--finetuning', dest='finetuning', action='store_true',
                        help='Finetune only last layer of CNN')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    # Distributed training parameters
    parser.add_argument('--world-size', dest='world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-backend', dest='dist_backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--dist-url', dest='dist_url', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--rank', dest='rank', default=-1, type=int,
                        help='rank of the worker')
    
    args = parser.parse_args()

    # Load data and checkpoint path from run
    try:
        args.data_path = run.input_datasets[args.dataset_name]
        print('Loaded registered dataset')
    except:
        print('Could not find registered dataset. Loading default data path.')
    try:
        args.resume = run.input_datasets[args.resume]
        print('Loaded checkpoint path')
    except:
        args.resume = None
        print('Could not find checkpoint path')
    
    # set distributed mode
    args.distributed = args.world_size >= 2
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args=args)