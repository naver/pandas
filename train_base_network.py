######
# Modified version of: https://github.com/pytorch/vision/blob/main/references/detection/train.py
######

import datetime
import os
import time
import platform
import json

import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from torchvision.models import resnet50
import torch.nn as nn
from model_components.faster_rcnn_predictors import FastRCNNPredictorClassAgnosticRegressor
from torchvision.models.detection.faster_rcnn import _resnet_fpn_extractor

from detection_utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from engine import train_one_epoch, evaluate_coco
from dataset_utils.coco_utils import get_coco_dataset
from dataset_utils.voc2012_utils import get_voc_datasets

from detection_utils.presets import get_transforms
import detection_utils.utils_od as utils


def bool_var(s):
    if s == '0' or s == 'False' or s == 'false':
        return False
    elif s == '1' or s == 'True' or s == 'true':
        return True
    msg = 'Invalid value "%s" for bool variable (should be 0/1 or True/False or true/false)'
    raise ValueError(msg % s)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training', add_help=add_help)

    parser.add_argument('--dataset', default='voc', help='dataset', choices=['voc', 'coco_half'])
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--data_path', type=str)

    parser.add_argument('--output_dir', help='path where to save')
    parser.add_argument('--num_classes', default=21, type=int, help='number of classes (including background)')
    parser.add_argument('--backbone_ckpt', type=str)

    # voc parameters
    parser.add_argument('--voc_split', type=str, choices=['10-10', 'full'], default='10-10')
    parser.add_argument('--split_path', type=str)

    # coco parameters
    parser.add_argument('--coco_anno_root', type=str)
    parser.add_argument('--coco_anno_file_template', default='coco_half_{}.json')

    # network parameters
    parser.add_argument('-b', '--batch-size', default=2, type=int,
                        help='images per gpu, the total batch size is $NGPU x batch_size')
    parser.add_argument('--epochs', default=26, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.02 / 8, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-scheduler', default="multisteplr", help='the lr scheduler (default: multisteplr)')
    parser.add_argument('--lr-step-size', default=8, type=int,
                        help='decrease lr every step-size epochs (evalmultisteplr scheduler only)')
    parser.add_argument('--lr-steps', default=[16, 22], nargs='+', type=int,
                        help='decrease lr every step-size epochs (multisteplr scheduler only)')
    parser.add_argument('--lr-gamma', default=0.1, type=float,
                        help='decrease lr by a factor of lr-gamma (multisteplr scheduler only)')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--warmup_iters', default=1000, type=int)
    parser.add_argument('--warmup_factor', default=1. / 1000, type=float)
    parser.add_argument('--feature_dim', default=1024, type=int, help='feature size')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    parser.add_argument('--data-augmentation', default="hflip", help='data augmentation policy (default: hflip)')
    parser.add_argument('--nesterov', type=bool_var, default=False)
    parser.add_argument('--freeze_bn', type=bool_var, default=True)

    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )

    return parser


def get_model(args):
    backbone_ckpt = args.backbone_ckpt
    # load a pre-trained model for classification and return only the features
    res50_model = resnet50(pretrained=False)
    if backbone_ckpt is not None:
        # load a different pre-trained backbone into the model (e.g., from SSL)
        print('loading ckpt from: %s' % backbone_ckpt)
        initial_bbone_ckpt = torch.load(backbone_ckpt)
        updated_bbone_ckpt = {}
        for k, v in initial_bbone_ckpt['state_dict'].items():
            if 'fc' in k:
                continue
            elif 'module.encoder_q' in k:
                k2 = k.split('module.encoder_q.')[1]
                updated_bbone_ckpt[k2] = v

        if 'fc.weight' and 'fc.bias' not in updated_bbone_ckpt.keys():
            # these parameters will be skipped in the next line anyways
            updated_bbone_ckpt['fc.weight'] = res50_model.state_dict()['fc.weight']
            updated_bbone_ckpt['fc.bias'] = res50_model.state_dict()['fc.bias']
        res50_model.load_state_dict(updated_bbone_ckpt)

    # FasterRCNN needs to know the number of output channels in a backbone
    backbone = res50_model
    backbone.out_channels = 2048

    print('\nUsing Class Agnostic Regressor...')
    box_predictor = FastRCNNPredictorClassAgnosticRegressor(args.feature_dim, args.num_classes)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        num_classes=args.num_classes, trainable_backbone_layers=3)
    model.backbone = _resnet_fpn_extractor(backbone, 3, norm_layer=nn.BatchNorm2d)
    model.roi_heads.box_predictor = box_predictor

    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.requires_grad_(False)
                m.bias.requires_grad_(False)
    print('Model:', model)
    return model


def train_with_epochs(args, model, model_without_ddp, optimizer, data_loader, data_loader_test, device,
                      lr_scheduler):
    print("Start training")
    start_time = time.time()
    model_best_performance = 0.
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq,
                        freeze_bn=args.freeze_bn, warmup_factor=args.warmup_factor,
                        warmup_iters=args.warmup_iters)
        lr_scheduler.step()

        # evaluate after every epoch
        evaluator, stats = evaluate_coco(model, data_loader_test, device=device, per_class=True)
        test_stats = evaluator.coco_eval['bbox'].stats
        print('mAP IoU 0.5:0.95 ', test_stats[0])
        print('mAP IoU 0.5 ', test_stats[1])

        # save most recent model and best model
        if args.output_dir:
            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args,
                'epoch': epoch,
                'performance': test_stats[0]
            }
            if test_stats[0] > model_best_performance:
                model_best_performance = test_stats[0]
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'model_best.pth'))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, 'model_recent.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")
    if args.dataset == 'voc':
        dataset, _, dataset_test = get_voc_datasets(args.data_path, args.split_path, split=args.voc_split,
                                                    augmentation=args.data_augmentation)
    elif args.dataset == 'coco_half':
        dataset = get_coco_dataset(root=args.data_path, image_set='train',
                                   transforms=get_transforms(True, args.data_augmentation),
                                   anno_root=args.coco_anno_root,
                                   anno_file_template=args.coco_anno_file_template)
        dataset_test = get_coco_dataset(root=args.data_path, image_set='val',
                                        transforms=get_transforms(False, args.data_augmentation),
                                        anno_root=args.coco_anno_root,
                                        anno_file_template=args.coco_anno_file_template)
    else:
        raise NotImplementedError

    print('Dataset Len: ', len(dataset))

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    if args.aspect_ratio_group_factor >= 0:
        group_ids = create_aspect_ratio_groups(dataset, k=args.aspect_ratio_group_factor)
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = get_model(args)

    model.to(device)

    model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == 'multisteplr':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosineannealinglr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        raise RuntimeError("Invalid lr scheduler '{}'. Only MultiStepLR and CosineAnnealingLR "
                           "are supported.".format(args.lr_scheduler))

    if args.resume and not args.test_only:
        print('\nLoading from: %s' % args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        print('\nLoading from: %s' % args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        evaluate_coco(model, data_loader_test, device=device, per_class=True)
        return

    train_with_epochs(args, model, model_without_ddp, optimizer, data_loader, data_loader_test, device,
                      lr_scheduler)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    print('gpu node ', platform.node())
    main(args)
