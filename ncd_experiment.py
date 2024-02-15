# ---
# PANDAS
# Copyright (C) 2023 NAVER Corp.
# CC BY-NC-SA 4.0 license
# ---

import os
import torch
import torch.utils.data
import numpy as np
import platform
import time
import json

print('gpu node ', platform.node())

import detection_utils.utils_od as utils
from model_components.pandas_model import NCDModel
from dataset_utils.lvis_dataset import LvisDataset
from dataset_utils.voc2012_utils import get_voc_datasets


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='NCD Training', add_help=add_help)

    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'lvis'])
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--base_num_classes', type=int)
    parser.add_argument('--batch_size', default=20, type=int)
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--num_clusters', default=100, type=int)
    parser.add_argument('--feature_dim', type=int, default=1024)

    # checkpoint files
    parser.add_argument('--base_detection_ckpt', type=str, default=None)  # base phase supervised checkpoint
    parser.add_argument('--ncd_ckpt', type=str,
                        default=None)  # optionally provide the already trained ncd model checkpoint

    # dataset parameters
    parser.add_argument('--data_path', type=str)

    # voc specific parameters
    parser.add_argument('--voc_split', type=str, choices=['10-10', 'full'], default='10-10')
    parser.add_argument('--split_path', type=str)

    # lvis specific parameters
    parser.add_argument('--coco_half_train_json', type=str)
    parser.add_argument('--coco_second_half_train_json', type=str)
    parser.add_argument('--lvis_train_json', type=str)
    parser.add_argument('--lvis_val_json', type=str)

    # for ground truth vs. clusters
    parser.add_argument('--prototype_init', type=str, default='pandas',
                        choices=['cluster_all', 'pandas', 'gt_prototypes'])
    parser.add_argument('--similarity_metric', type=str, default='invert_square', choices=['invert',
                                                                                           'invert_square',
                                                                                           'cosine',
                                                                                           'dot_prod'])

    parser.add_argument('--proba_norm', type=str, default='l1', choices=['l1', 'softmax'])
    parser.add_argument('--background_classifier', type=str, default='softmax', choices=['softmax', 'none'])

    # post processing
    parser.add_argument('--dets_per_image', type=int, default=100)
    parser.add_argument('--score_thresh', type=float, default=0.05)

    # for kmeans
    parser.add_argument('--kmeans_n_init', type=int, default=10)
    parser.add_argument('--kmeans_max_iter', type=int, default=1000)
    parser.add_argument('--kmeans_seed', type=int, default=42)

    parser.add_argument('--save_cluster_mapping', type=str,
                        default=None)  # path for saving the cluster mapping out (optional)

    return parser


def get_voc_key():
    # key mapping voc ids to names, base/novel
    voc_key = {
        0: ['background', '-'],
        1: ['aeroplane', 'base'],
        2: ['bicycle', 'base'],
        3: ['bird', 'base'],
        4: ['boat', 'base'],
        5: ['bottle', 'base'],
        6: ['bus', 'base'],
        7: ['car', 'base'],
        8: ['cat', 'base'],
        9: ['chair', 'base'],
        10: ['cow', 'base'],
        11: ['diningtable', 'novel'],
        12: ['dog', 'novel'],
        13: ['horse', 'novel'],
        14: ['motorbike', 'novel'],
        15: ['person', 'novel'],
        16: ['pottedplant', 'novel'],
        17: ['sheep', 'novel'],
        18: ['sofa', 'novel'],
        19: ['train', 'novel'],
        20: ['tvmonitor', 'novel']
    }
    return voc_key


def get_lvis_key(lvis_train_file):
    # key mapping lvis ids to names, base/novel, f/c/r
    with open(lvis_train_file) as f:
        dict2 = json.load(f)
    id_to_cat = {}
    for k in dict2['categories']:
        id_to_cat[k['id']] = [k['name'], k['frequency']]
    # RNCDL line 127: https://github.com/vlfom/RNCDL/blob/main/configs/train/discovery/coco50pct_lvis.py
    base_ids = [3, 12, 34, 35, 36, 41, 45, 58, 60, 76, 77, 80, 90, 94, 99, 118, 127, 133, 139, 154, 173, 183,
                207, 217, 225, 230, 232, 271, 296, 344, 367, 378, 387, 421, 422, 445, 469, 474, 496, 534, 569,
                611, 615, 631, 687, 703, 705, 716, 735, 739, 766, 793, 816, 837, 881, 912, 923, 943, 961, 962,
                964, 976, 982, 1000, 1019, 1037, 1071, 1077, 1079, 1095, 1097, 1102, 1112, 1115, 1123, 1133,
                1139, 1190, 1202]  # missing hot dog --> only 79 classes

    lvis_key = {}
    for i, (curr_id, vals) in enumerate(id_to_cat.items()):
        curr_class = vals[0]
        freq = vals[1]
        if curr_id in base_ids:
            base_or_novel = 'base'
        else:
            base_or_novel = 'novel'
        lvis_key[i] = [curr_class, base_or_novel, freq]
    return lvis_key, base_ids


def make_loader(dataset, shuffle=False, batch_size=1, num_workers=4):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        collate_fn=utils.collate_fn)
    return dataloader


def get_loaders(args):
    # make datasets
    if args.dataset == 'voc':
        dataset_base, dataset_novel, _ = get_voc_datasets(args.data_path, args.split_path,
                                                          split=args.voc_split,
                                                          augmentation=None)
        dataset_discovery, _, dataset_test = get_voc_datasets(args.data_path, args.split_path,
                                                              split='full',
                                                              augmentation=None)
        class_names = get_voc_key()
        base_ids = np.arange(1, 11)
        novel_ids = np.arange(11, 21)
    elif args.dataset == 'lvis':
        class_names, base_ids = get_lvis_key(args.lvis_train_json)

        dataset_base = LvisDataset(root=args.data_path,
                                   anno_root=args.coco_half_train_json,
                                   included_cats=base_ids)

        dataset_novel = LvisDataset(root=args.data_path,
                                    anno_root=args.coco_second_half_train_json)

        dataset_discovery = LvisDataset(root=args.data_path,
                                        anno_root=args.lvis_train_json)

        # full test set
        dataset_test = LvisDataset(root=args.data_path,
                                   anno_root=args.lvis_val_json)
        base_ids = np.array(base_ids)
        all_ids = np.arange(1, 1204)
        novel_ids = np.setdiff1d(all_ids, base_ids)
    else:
        raise NotImplementedError

    # make data loaders
    if dataset_base is not None:
        base_loader = make_loader(dataset_base, shuffle=False, batch_size=args.batch_size, num_workers=4)
    else:
        base_loader = None
    if dataset_novel is not None:
        novel_loader = make_loader(dataset_novel, shuffle=False, batch_size=args.batch_size, num_workers=4)
    else:
        novel_loader = None
    discovery_loader = make_loader(dataset_discovery, shuffle=False, batch_size=args.batch_size, num_workers=4)
    test_loader = make_loader(dataset_test, shuffle=False, batch_size=1, num_workers=4)

    return base_loader, novel_loader, discovery_loader, test_loader, base_ids, novel_ids, class_names


def get_model(args, base_ids, novel_ids, class_names):
    # create model
    model = NCDModel(
        num_classes=args.num_classes,
        num_base_classes=args.base_num_classes,
        base_checkpoint=args.base_detection_ckpt,
        ncd_checkpoint=args.ncd_ckpt,
        num_clusters=args.num_clusters,
        prototype_init=args.prototype_init,
        class_names=class_names,
        base_ids=base_ids,
        novel_ids=novel_ids,
        similarity_metric=args.similarity_metric,
        proba_norm=args.proba_norm,
        background_classifier=args.background_classifier,
        device=args.device,
        dataset=args.dataset,
        dets_per_img=args.dets_per_image,
        score_thresh=args.score_thresh,
        output_dir=args.output_dir,
        kmeans_n_init=args.kmeans_n_init,
        kmeans_max_iter=args.kmeans_max_iter,
        kmeans_seed=args.kmeans_seed,
        save_cluster_mapping=args.save_cluster_mapping)
    return model


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    base_loader, novel_loader, discovery_loader, test_loader, base_ids, novel_ids, class_names = get_loaders(args)

    model = get_model(args, base_ids, novel_ids, class_names)

    # initialize model
    if args.prototype_init == 'pandas':
        print('Len Novel ', len(novel_loader.dataset))
        print('Len Base ', len(base_loader.dataset))
        model.initialize_model(discovery_loader, novel_loader, base_loader)
    else:
        model.initialize_model(discovery_loader)

    if args.ncd_ckpt is None:
        # only save if it was a new run
        if args.save_name is not None:
            name = args.save_name
        else:
            name = 'ncd_model.pth'
        model.save_model(os.path.join(args.output_dir, name))

    # evaluate model
    print('Testing...')
    model.evaluate(test_loader)


if __name__ == '__main__':
    start = time.time()
    args = get_args_parser().parse_args()
    print("Arguments {}".format(json.dumps(vars(args), indent=4, sort_keys=True)))

    main(args)
    print('\nFINAL TIME: ', time.time() - start)
