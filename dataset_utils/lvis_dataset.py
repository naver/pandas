######
# Modified version of: https://github.com/ContinualAI/avalanche/blob/master/avalanche/benchmarks/datasets/lvis_dataset/lvis_dataset.py
######

################################################################################
# Copyright (c) 2022 ContinualAI                                               #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 18-02-2022                                                             #
# Author: Lorenzo Pellegrini                                                   #
#                                                                              #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

""" LVIS PyTorch Object Detection Dataset """

from pathlib import Path
from typing import Union, List, Sequence
import os

import torch
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToTensor
from typing_extensions import TypedDict
from .coco_utils import ConvertLvisPolysToMask
import detection_utils.transforms as T
import detection_utils.presets as presets

try:
    from lvis import LVIS
except ImportError:
    raise ModuleNotFoundError(
        "LVIS not found, if you want to use detection "
        "please install avalanche with the detection "
        "dependencies: "
        "pip install avalanche-lib[detection]"
    )

lvis_name_to_coco_name = {
    'airplane': 'airplane',
    'apple': 'apple',
    'backpack': 'backpack',
    'ball': 'sports ball',
    'banana': 'banana',
    'baseball_bat': 'baseball bat',
    'baseball_glove': 'baseball glove',
    'bear': 'bear',
    'bed': 'bed',
    'bench': 'bench',
    'bicycle': 'bicycle',
    'bird': 'bird',
    'boat': 'boat',
    'book': 'book',
    'bottle': 'bottle',
    'bowl': 'bowl',
    'broccoli': 'broccoli',
    'bus_(vehicle)': 'bus',
    'cake': 'cake',
    'car_(automobile)': 'car',
    'carrot': 'carrot',
    'cat': 'cat',
    'cellular_telephone': 'cell phone',
    'chair': 'chair',
    'clock': 'clock',
    'computer_keyboard': 'keyboard',
    'cow': 'cow',
    'cup': 'cup',
    'dining_table': 'dining table',
    'dog': 'dog',
    'doughnut': 'donut',
    'elephant': 'elephant',
    'fireplug': 'fire hydrant',
    'flowerpot': 'potted plant',
    'fork': 'fork',
    'frisbee': 'frisbee',
    'giraffe': 'giraffe',
    'hair_dryer': 'hair drier',
    'handbag': 'handbag',
    'horse': 'horse',
    'kite': 'kite',
    'knife': 'knife',
    'laptop_computer': 'laptop',
    'microwave_oven': 'microwave',
    'motorcycle': 'motorcycle',
    'mouse_(computer_equipment)': 'mouse',
    'necktie': 'tie',
    'orange_(fruit)': 'orange',
    'oven': 'oven',
    'parking_meter': 'parking meter',
    'person': 'person',
    'pizza': 'pizza',
    'refrigerator': 'refrigerator',
    'remote_control': 'remote',
    'sandwich': 'sandwich',
    'scissors': 'scissors',
    'sheep': 'sheep',
    'sink': 'sink',
    'skateboard': 'skateboard',
    'ski': 'skis',
    'snowboard': 'snowboard',
    'sofa': 'couch',
    'spoon': 'spoon',
    'stop_sign': 'stop sign',
    'suitcase': 'suitcase',
    'surfboard': 'surfboard',
    'teddy_bear': 'teddy bear',
    'television_set': 'tv',
    'tennis_racket': 'tennis racket',
    'toaster': 'toaster',
    'toilet': 'toilet',
    'toothbrush': 'toothbrush',
    'traffic_light': 'traffic light',
    'train_(railroad_vehicle)': 'train',
    'truck': 'truck',
    'umbrella': 'umbrella',
    'vase': 'vase',
    'wineglass': 'wine glass',
    'zebra': 'zebra'}


def get_transform(data_augmentation):
    if data_augmentation is not None:
        return presets.DetectionPresetTrain(data_augmentation)
    else:
        return presets.DetectionPresetEval()


class LvisDataset(object):
    """LVIS PyTorch Object Detection Dataset"""

    def __init__(
            self,
            root: Union[str, Path] = None,
            *,
            anno_root=None,
            transform=None,
            loader=default_loader,
            download=False,
            lvis_api=None,
            img_ids: List[int] = None,
            included_cats=[]
    ):
        """
        Creates an instance of the LVIS dataset.
        :param root: The directory where the dataset can be found or downloaded.
            Defaults to None, which means that the default location for
            "lvis" will be used.
        :param train: If True, the training set will be returned. If False,
            the test set will be returned.
        :param transform: The transformation to apply to (img, annotations)
            values.
        :param loader: The image loader to use.
        :param download: If True, the dataset will be downloaded if needed.
        :param lvis_api: An instance of the LVIS class (from the lvis-api) to
            use. Defaults to None, which means that annotations will be loaded
            from the annotation json found in the root directory.
        :param img_ids: A list representing a subset of images to use. Defaults
            to None, which means that the dataset will contain all images
            in the LVIS dataset.
        """

        transform = get_transform(transform)
        self.transform = transform
        self.loader = loader
        self.bbox_crop = True
        self.img_ids = img_ids
        self.anno_root = anno_root
        self.download = download
        self.verbose = True
        self.root = root

        self.targets = None
        self.lvis_api = lvis_api
        self.included_cats = included_cats

        super(LvisDataset, self).__init__()

        self._load_dataset()

    def _load_dataset(self) -> None:
        """
        The standardized dataset download and load procedure.
        For more details on the coded procedure see the class documentation.
        This method shouldn't be overridden.
        This method will raise and error if the dataset couldn't be loaded
        or downloaded.
        :return: None
        """
        metadata_loaded = False
        metadata_load_error = None
        try:
            metadata_loaded = self._load_metadata()
        except Exception as e:
            metadata_load_error = e

        if metadata_loaded:
            if self.verbose:
                print("Files already downloaded and verified")
            return

        if not self.download:
            msg = (
                "Error loading dataset metadata (dataset download was "
                'not attempted as "download" is set to False)'
            )
            if metadata_load_error is None:
                raise RuntimeError(msg)
            else:
                print(msg)
                raise metadata_load_error

        try:
            self._download_dataset()
        except Exception as e:
            err_msg = self._download_error_message()
            print(err_msg, flush=True)
            raise e

        if not self._load_metadata():
            err_msg = self._download_error_message()
            print(err_msg)
            raise RuntimeError(
                "Error loading dataset metadata (... but the download "
                "procedure completed successfully)"
            )

    def _load_metadata(self) -> bool:
        must_load_api = self.lvis_api is None
        must_load_img_ids = self.img_ids is None
        try:
            # Load metadata
            if must_load_api:
                ann_json_path = self.anno_root

                self.lvis_api = LVIS(ann_json_path)

            if must_load_img_ids:
                self.img_ids = list(sorted(self.lvis_api.get_img_ids()))

            self.targets = LVISDetectionTargets(self.lvis_api, self.img_ids)

            # Try loading an image
            if len(self.img_ids) > 0:
                img_id = self.img_ids[0]
                img_dict: LVISImgEntry = self.lvis_api.load_imgs(ids=[img_id])[
                    0
                ]
                assert self._load_img(img_dict) is not None
        except BaseException:
            if must_load_api:
                self.lvis_api = None
            if must_load_img_ids:
                self.img_ids = None

            self.targets = None
            raise

        return True

    def _download_error_message(self) -> str:
        return (
                "[LVIS] Error downloading the dataset. Consider "
                "downloading it manually at: https://www.lvisdataset.org/dataset"
                " and placing it in: " + str(self.root)
        )

    def get_categories(self):
        return self.lvis_api.cats

    def get_annotations(self, index, include_img_dict=False):
        img_id = self.img_ids[index]
        img_dict: LVISImgEntry = self.lvis_api.load_imgs(ids=[img_id])[0]
        annotation_dicts = self.targets[index]

        # Transform from LVIS dictionary to torchvision-style target
        num_objs = len(annotation_dicts)

        boxes = []
        labels = []
        for i in range(num_objs):
            xmin = annotation_dicts[i]["bbox"][0]
            ymin = annotation_dicts[i]["bbox"][1]
            xmax = xmin + annotation_dicts[i]["bbox"][2]
            ymax = ymin + annotation_dicts[i]["bbox"][3]
            cat = annotation_dicts[i]["category_id"]

            if self.included_cats == [] or cat in self.included_cats:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cat)

        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([img_id])
        areas = []
        for i in range(num_objs):
            areas.append(annotation_dicts[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = dict()
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd
        target['height'] = img_dict['height']
        target['width'] = img_dict['width']
        target['size'] = img_dict['width'] * img_dict['height']

        if include_img_dict:
            return target, img_dict
        return target

    def __getitem__(self, index):
        """
        Loads an instance given its index.
        :param index: The index of the instance to retrieve.
        :return: a (sample, target) tuple where the target is a
            torchvision-style annotation for object detection
            https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        """

        target, img_dict = self.get_annotations(index, include_img_dict=True)
        img = self._load_img(img_dict)

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target

    def __len__(self):
        return len(self.img_ids)

    def _load_img(self, img_dict: "LVISImgEntry"):
        coco_url = img_dict["coco_url"]
        splitted_url = coco_url.split("/")
        img_path = splitted_url[-2] + "/" + splitted_url[-1]
        # <root>/train2017/<img_id>.jpg
        final_path = os.path.join(self.root, img_path)
        return self.loader(str(final_path))


class LVISImgEntry(TypedDict):
    id: int
    date_captured: str
    neg_category_ids: List[int]
    license: int
    height: int
    width: int
    flickr_url: str
    coco_url: str
    not_exhaustive_category_ids: List[int]


class LVISAnnotationEntry(TypedDict):
    id: int
    area: float
    segmentation: List[List[float]]
    image_id: int
    bbox: List[int]
    category_id: int


class LVISDetectionTargets(Sequence[List[LVISAnnotationEntry]]):
    def __init__(self, lvis_api: LVIS, img_ids: List[int] = None):
        super(LVISDetectionTargets, self).__init__()
        self.lvis_api = lvis_api
        if img_ids is None:
            img_ids = list(sorted(lvis_api.get_img_ids()))

        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        annotation_ids = self.lvis_api.get_ann_ids(img_ids=[img_id])
        annotation_dicts: List[LVISAnnotationEntry] = self.lvis_api.load_anns(
            annotation_ids
        )
        return annotation_dicts


def _test_to_tensor(a, b):
    return ToTensor()(a), b


def _standard_transforms(a, b, transforms=None):
    t = [ConvertLvisPolysToMask()]
    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)
    return transforms(a, b)


def _detection_collate_fn(batch):
    return tuple(zip(*batch))


__all__ = [
    "LvisDataset",
    "LVISImgEntry",
    "LVISAnnotationEntry",
    "LVISDetectionTargets",
]
