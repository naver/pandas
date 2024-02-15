######
# Modified version of: https://github.com/pytorch/vision/blob/main/references/detection/engine.py
######

import math
import sys
import time
import torch
import numpy as np
import pickle
import os
import torchvision.models.detection.mask_rcnn

from dataset_utils.coco_utils import get_coco_api_from_dataset
from dataset_utils.coco_eval import CocoEvaluator
from dataset_utils.lvis_evaluator import LvisEvaluator
import detection_utils.utils_od as utils


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, freeze_bn=False, warmup_factor=0.001,
                    warmup_iters=1000):
    model.train()

    if freeze_bn:
        model.apply(set_bn_eval)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        lr_scheduler = utils.warmup_lr_scheduler(
            optimizer, warmup_iters, warmup_factor)

    for images, targets_ in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = []
        for t in targets_:
            d = {}
            for k, v in t.items():
                if torch.is_tensor(v):
                    d[k] = v.to(device)
            targets.append(d)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate_coco(model, data_loader, device, class_names=None, per_class=True):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)  # bbox
    coco_evaluator = CocoEvaluator(coco, iou_types, class_names, per_class=per_class)
    partial_stats = None
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator, partial_stats


@torch.no_grad()
def evaluate_lvis(model, data_loader, device, class_names=None, save_dir=None):
    print('\nUSING LVIS EVALUATOR...')
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    iou_types = ["bbox"]
    lvis = data_loader.dataset.lvis_api
    lvis_evaluator = LvisEvaluator(lvis, iou_types)
    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        lvis_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    lvis_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    lvis_evaluator.accumulate()
    lvis_evaluator.summarize()

    # grab and print per class results for AP and AP50
    lvis_eval = lvis_evaluator.lvis_eval_per_iou['bbox']
    for metric_name in ['AP', 'AP50']:
        per_class_results = {}
        base_class_results = {}
        novel_class_results = {}
        c_freq_results = {}
        f_freq_results = {}
        r_freq_results = {}
        for class_id in lvis_eval.results_per_class:  # {class_id: [AP, AP50, ...]}
            class_name = class_names[class_id][0]
            class_place = class_names[class_id][1]
            freq_ = class_names[class_id][-1]
            metric_value = lvis_eval.results_per_class[class_id][metric_name]
            per_class_results[class_name] = float(metric_value * 100)
            if class_place == 'novel':
                novel_class_results[class_name] = float(metric_value * 100)
            elif class_place == 'base':
                base_class_results[class_name] = float(metric_value * 100)

            if freq_ == 'c':
                c_freq_results[class_name] = float(metric_value * 100)
            elif freq_ == 'f':
                f_freq_results[class_name] = float(metric_value * 100)
            elif freq_ == 'r':
                r_freq_results[class_name] = float(metric_value * 100)

        # print per class results
        print('\nMetric: ', metric_name)
        keys, values = tuple(zip(*per_class_results.items()))
        for k, v in zip(keys, values):
            print('%s: %0.3f' % (k, v))
        if save_dir is not None:
            save_name = metric_name + '_lvis_results'
            print('saving lvis per class results for %s to: %s' % (metric_name,
                                                                   os.path.join(save_dir, save_name + '.pickle')))
            with open(os.path.join(save_dir, save_name + '.pickle'), 'wb') as f:
                pickle.dump(per_class_results, f)

        # print base, novel, and fcr class results
        base_ap = gather_avg_results(base_class_results)
        novel_ap = gather_avg_results(novel_class_results)

        f_freq_ap = gather_avg_results(f_freq_results)
        c_freq_ap = gather_avg_results(c_freq_results)
        r_freq_ap = gather_avg_results(r_freq_results)

        all_ap = gather_avg_results(per_class_results)

        print('\nBase Score (%d cls): %0.3f' % (len(base_class_results), base_ap))
        print('Novel Score (%d cls): %0.3f' % (len(novel_class_results), novel_ap))
        print('All Score (%d cls): %0.3f' % (len(per_class_results), all_ap))

        print('\nf Classes Score (%d cls): %0.3f' % (len(f_freq_results), f_freq_ap))
        print('c Classes Score (%d cls): %0.3f' % (len(c_freq_results), c_freq_ap))
        print('r Classes Score (%d cls): %0.3f' % (len(r_freq_results), r_freq_ap))
    torch.set_num_threads(n_threads)
    return lvis_evaluator


def gather_avg_results(class_results):
    # sum up results
    values = list(class_results.values())
    values = np.array(values)
    ap_ = np.mean(values[values > -1])
    return ap_
