######
# Modified version of: https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py
######

import os
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision

import detection_utils.transforms as T

from pycocotools import mask as coco_mask
from pycocotools.coco import COCO

from PIL import Image

orig_label_mapping = {'airplane': 5,
                      'apple': 53,
                      'backpack': 27,
                      'banana': 52,
                      'baseball bat': 39,
                      'baseball glove': 40,
                      'bear': 23,
                      'bed': 65,
                      'bench': 15,
                      'bicycle': 2,
                      'bird': 16,
                      'boat': 9,
                      'book': 84,
                      'bottle': 44,
                      'bowl': 51,
                      'broccoli': 56,
                      'bus': 6,
                      'cake': 61,
                      'car': 3,
                      'carrot': 57,
                      'cat': 17,
                      'cell phone': 77,
                      'chair': 62,
                      'clock': 85,
                      'couch': 63,
                      'cow': 21,
                      'cup': 47,
                      'dining table': 67,
                      'dog': 18,
                      'donut': 60,
                      'elephant': 22,
                      'fire hydrant': 11,
                      'fork': 48,
                      'frisbee': 34,
                      'giraffe': 25,
                      'hair drier': 89,
                      'handbag': 31,
                      'horse': 19,
                      'hot dog': 58,
                      'keyboard': 76,
                      'kite': 38,
                      'knife': 49,
                      'laptop': 73,
                      'microwave': 78,
                      'motorcycle': 4,
                      'mouse': 74,
                      'orange': 55,
                      'oven': 79,
                      'parking meter': 14,
                      'person': 1,
                      'pizza': 59,
                      'potted plant': 64,
                      'refrigerator': 82,
                      'remote': 75,
                      'sandwich': 54,
                      'scissors': 87,
                      'sheep': 20,
                      'sink': 81,
                      'skateboard': 41,
                      'skis': 35,
                      'snowboard': 36,
                      'spoon': 50,
                      'sports ball': 37,
                      'stop sign': 13,
                      'suitcase': 33,
                      'surfboard': 42,
                      'teddy bear': 88,
                      'tennis racket': 43,
                      'tie': 32,
                      'toaster': 80,
                      'toilet': 70,
                      'toothbrush': 90,
                      'traffic light': 10,
                      'train': 7,
                      'truck': 8,
                      'tv': 72,
                      'umbrella': 28,
                      'vase': 86,
                      'wine glass': 46,
                      'zebra': 24}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask():
    def __init__(self, mapping=None, included_cats=[]):
        super(ConvertCocoPolysToMask, self).__init__()
        self.mapping = mapping
        if included_cats != []:
            self.included_cats = [orig_label_mapping[l] for l in included_cats]
        else:
            self.included_cats = []

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # mask off labels for classes not included
        keep_cat = torch.BoolTensor((len(classes)))
        for ii, cat in enumerate(classes):
            if self.included_cats == [] or cat in self.included_cats:
                keep_cat[ii] = True
            else:
                keep_cat[ii] = False

        boxes = boxes[keep_cat]
        classes = classes[keep_cat]
        masks = masks[keep_cat]
        if keypoints is not None:
            keypoints = keypoints[keep_cat]

        if self.mapping is not None:
            classes = [self.mapping[c.item()] for c in classes]
            classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations_indices(dataset, cat_list=None):
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        # keypoints task have a slight different criteria for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    if not isinstance(dataset, torchvision.datasets.CocoDetection):
        raise TypeError(
            f"This function expects dataset of type torchvision.datasets.CocoDetection, instead  got {type(dataset)}"
        )
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)
    return ids


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    ids = _coco_remove_images_without_annotations_indices(dataset, cat_list)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    print(f'Converting dataset to the COCO API (using generic route)')
    coco_ds = COCO()
    coco_ds.dataset = convert_to_coco_format(ds)
    coco_ds.createIndex()
    return coco_ds


def convert_to_coco_format(ds, include_file_names=False):
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = {}
    warned = False
    if isinstance(ds, torch.utils.data.dataset.Subset):  # or isinstance(ds, torch.utils.data.Subset):
        print('INFO: patching torch.utils.data.dataset.Subset with get_annotations on-the-fly')
        ds.get_annotations = lambda i: ds.dataset.get_annotations(ds.indices[i])
        ds.get_categories = lambda: ds.dataset.get_categories()
    if hasattr(ds, 'get_categories'):
        categories = ds.get_categories()
    else:
        print('WARNING: gathering up categories on-the-fly (may lead to inconsistencies)')
    for img_idx in tqdm(range(len(ds)), total=len(ds), desc='COCO conversion'):
        img_dict = {}
        size = None
        if hasattr(ds, 'get_annotations'):
            targets = ds.get_annotations(img_idx)
            if 'height' in targets and 'width' in targets:
                img_dict['size'] = targets['height'], targets[
                    'width']  # give preference as its more likely to be in correct order
            elif 'size' in targets:
                img_dict['size'] = targets["size"]  # assuming its HxW
        if 'size' not in img_dict:
            if not warned:
                print('WARNING: following slow conversion path')
            warned = True
            img, targets = ds[img_idx]
            img_dict['size'] = img.shape[-2], img.shape[-1]
        img_dict["height"] = img_dict['size'][0]
        img_dict["width"] = img_dict['size'][1]
        if include_file_names:
            if 'file_name' in targets:
                img_dict['file_name'] = targets['file_name']
            else:
                img_dict['file_name'] = ds.images[img_idx]
            assert os.path.isabs(img_dict['file_name'])
        image_id = targets["image_id"].item()
        img_dict["id"] = image_id
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        if len(bboxes) > 0:
            bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            if labels[i] not in categories:
                categories[labels[i]] = {
                    'id': labels[i],
                    'name': f'{labels[i]}'
                }
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [categories[k] for k in sorted(categories.keys())]

    return dataset


def get_coco_api_from_dataset(dataset):
    print(f'Getting COCO api from dataset {dataset}')
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    if hasattr(dataset, 'to_coco'):
        print(f'Calling dataset to_coco()')
        coco_ds = COCO()
        coco_ds.dataset = dataset.to_coco()
        coco_ds.createIndex()
        return coco_ds
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, included=[], lvis=False, mapping=None):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.included_cats = included
        self.ids = self.get_ids(included)
        self.lvis = lvis
        self.mapping = mapping

    def get_ids(self, included_cats):
        all_ids = list(sorted(self.coco.imgs.keys()))
        finalset = set()
        if included_cats == []:
            return all_ids
        else:
            for query_name in included_cats:
                query_id = self.coco.getCatIds(catNms=[query_name])[0]
                img_ids = self.coco.getImgIds(catIds=[query_id])
                finalset = finalset.union(img_ids)
            return list(sorted(finalset))

    def _load_image(self, id):
        if self.lvis:
            full_name = self.coco.loadImgs(id)[0]['coco_url']
            path = '/'.join(full_name.split('/')[-2:])
        else:
            path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if 'iscrowd' not in target:
            num_objs = len(target['labels'])
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            target['iscrowd'] = iscrowd
        return img, target


def get_coco_dataset(root, image_set, transforms, mode="instances", included=[], anno_root="../annotations",
                     anno_file_template="{}_{}2017.json",
                     mapping=None):
    if anno_file_template == "{}_{}2017.json":
        PATHS = {
            "train": ("train2017", os.path.join(anno_root, anno_file_template.format(mode, "train"))),
            "val": ("val2017", os.path.join(anno_root, anno_file_template.format(mode, "val"))),
        }
    else:
        PATHS = {
            "train": ("train2017", os.path.join(anno_root, anno_file_template.format("train"))),
            "val": ("val2017", os.path.join(anno_root, anno_file_template.format("val"))),
        }

    t = [ConvertCocoPolysToMask(mapping=mapping, included_cats=included)]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    if image_set != 'train' and included == []:
        included = []

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms, included=included, mapping=mapping)
    categories = dataset.coco.cats

    if image_set == "train":
        dataset = _coco_remove_images_without_annotations(dataset)

    dataset.class_names = {}
    for class_id, class_info in categories.items():
        dataset.class_names[class_id] = class_info['name']
    dataset.num_classes = 91
    dataset.num_bboxes = -1
    dataset.num_images = len(dataset)
    return dataset


class ConvertLvisPolysToMask:
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        # faked for this dataset
        for obj in anno:
            obj['iscrowd'] = 0

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        segmentations = [obj["segmentation"] for obj in anno]
        # masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        # masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        # target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([0 for _ in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target
