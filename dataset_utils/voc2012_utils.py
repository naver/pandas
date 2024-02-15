# ---
# PANDAS
# Copyright (C) 2023 NAVER Corp.
# CC BY-NC-SA 4.0 license
# ---

import os
import torchvision.datasets as datasets
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import detection_utils.presets as presets

# assign classes labels
voc_classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def get_transform(data_augmentation):
    if data_augmentation is not None:
        return presets.DetectionPresetTrain(data_augmentation)
    else:
        return presets.DetectionPresetEval()


class VOCDetection_COCOTrain(datasets.VOCDetection):
    def __init__(self, root, file_names, categories=voc_classes, augmentation=None, included_cats=[]):

        transform = get_transform(augmentation)
        self.transform = transform
        self.CLASSES = categories
        self.root = root

        image_dir = os.path.join(root, 'JPEGImages')
        annotation_dir = os.path.join(root, 'Annotations')

        # to mask off unseen or previous classes
        self.included_cats = included_cats

        self.images = [os.path.join(image_dir, x) for x in file_names]
        self.annotations_ = [os.path.join(annotation_dir, x.split('.')[0] + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations_))

    def convert(self, target, idx):
        anno = target['annotation']
        H, W = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        area = []
        iscrowd = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            cat = self.CLASSES.index(obj['name'])

            if self.included_cats == [] or cat in self.included_cats:
                boxes.append(bbox)
                classes.append(cat)
                iscrowd.append(False)
                area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)

        # image_name = anno['filename'].split('.')[0]
        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes.long()
        target["image_id"] = image_id
        target["size"] = torch.as_tensor([int(H), int(W)])
        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, index):
        img = pil_loader(self.images[index])
        target = self.get_annotations(index)
        return self.transform(img, target)

    def get_annotations(self, index):
        target = self.parse_voc_xml(
            ET.parse(self.annotations_[index]).getroot())
        return self.convert(target, index)


def get_files_split(file_split_root, split_f, split_type, file):
    with open(os.path.join(split_f), "r") as f:
        file_names = [x[:-1].split(' ') for x in f.readlines()]
    file_names = [fn[0].split('/')[-1] for fn in file_names]

    filename = os.path.join(file_split_root, os.path.join(split_type, file))
    train_ix = np.load(filename, allow_pickle=True)
    files = [file_names[ix] for ix in train_ix]
    return files


def get_full_train(file_split_root, voc_root, augmentation, split, split_type='10-10'):
    # split_type is dummy since we just need to get all data

    if split == 'train':
        split_f = os.path.join(file_split_root, 'train_aug.txt')
        aug = augmentation
        file0 = 'train-0.npy'
        file1 = 'train-1.npy'
    else:
        split_f = os.path.join(file_split_root, 'val.txt')
        aug = None
        if split == 'val':
            file0 = 'val-0.npy'
            file1 = 'val-1.npy'
        else:
            file = 'test_on_val-1.npy'

    # include all labels
    included_cats = []

    if split == 'test':
        files = get_files_split(file_split_root, split_f, split_type, file)
    else:
        files0 = get_files_split(file_split_root, split_f, split_type, file0)
        files1 = get_files_split(file_split_root, split_f, split_type, file1)
        files = files0 + files1

    voc_data = VOCDetection_COCOTrain(voc_root, file_names=files, included_cats=included_cats,
                                      augmentation=aug)
    return voc_data


def load_voc_splits(file_split_root, split_type, voc_root, augmentation='hflip'):
    dataset_dict = {}
    if split_type == 'full':
        # return full datasets
        dataset_dict['train'] = get_full_train(file_split_root, voc_root, augmentation, 'train')
        dataset_dict['val'] = get_full_train(file_split_root, voc_root, augmentation, 'val')
        dataset_dict['test'] = get_full_train(file_split_root, voc_root, augmentation, 'test')
    else:
        files = ['test_on_val-0.npy', 'train-0.npy', 'val-0.npy',
                 'test_on_val-1.npy', 'train-1.npy', 'val-1.npy']
        for file in files:

            if 'train' in file:
                split_f = os.path.join(file_split_root, 'train_aug.txt')
                aug = augmentation
            else:
                split_f = os.path.join(file_split_root, 'val.txt')
                aug = None

            # mask off unseen or previous labels
            if split_type == '10-10' and file in ['train-0.npy', 'val-0.npy', 'test_on_val-0.npy']:
                included_cats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            elif split_type == '10-10' and file in ['train-1.npy', 'val-1.npy']:
                included_cats = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            elif file == 'test_on_val-1.npy':
                included_cats = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            else:
                raise NotImplementedError

            files0 = get_files_split(file_split_root, split_f, split_type, file)
            voc_data0 = VOCDetection_COCOTrain(voc_root, file_names=files0, included_cats=included_cats,
                                               augmentation=aug)
            dataset_dict[file] = voc_data0
    return dataset_dict


def get_voc_datasets(data_path, split_path, split='10-10', augmentation=None):
    split_data = load_voc_splits(split_path, split, data_path, augmentation=augmentation)

    print('split data type ', type(split_data))
    print('split data keys ', split_data.keys())

    if split == 'full':
        return split_data['train'], split_data['val'], split_data['test']
    else:
        base_dataset = split_data['train-0.npy']
        novel_dataset = split_data['train-1.npy']
        test_dataset = split_data['test_on_val-1.npy']
        print('base dataset ', type(base_dataset))
        print('[] ', base_dataset[0][1])
        print('Successfully acquired datasets')
        return base_dataset, novel_dataset, test_dataset
