######
# Modified version of: https://github.com/facebookresearch/detectron2/blob/main/datasets/prepare_cocofied_lvis.py
######

import os
import json
from copy import deepcopy

# This mapping is extracted from the official LVIS mapping:
# https://github.com/lvis-dataset/lvis-api/blob/master/data/coco_to_synset.json
COCO_SYNSET_CATEGORIES = [
    {"synset": "person.n.01", "coco_cat_id": 1},
    {"synset": "bicycle.n.01", "coco_cat_id": 2},
    {"synset": "car.n.01", "coco_cat_id": 3},
    {"synset": "motorcycle.n.01", "coco_cat_id": 4},
    {"synset": "airplane.n.01", "coco_cat_id": 5},
    {"synset": "bus.n.01", "coco_cat_id": 6},
    {"synset": "train.n.01", "coco_cat_id": 7},
    {"synset": "truck.n.01", "coco_cat_id": 8},
    {"synset": "boat.n.01", "coco_cat_id": 9},
    {"synset": "traffic_light.n.01", "coco_cat_id": 10},
    {"synset": "fireplug.n.01", "coco_cat_id": 11},
    {"synset": "stop_sign.n.01", "coco_cat_id": 13},
    {"synset": "parking_meter.n.01", "coco_cat_id": 14},
    {"synset": "bench.n.01", "coco_cat_id": 15},
    {"synset": "bird.n.01", "coco_cat_id": 16},
    {"synset": "cat.n.01", "coco_cat_id": 17},
    {"synset": "dog.n.01", "coco_cat_id": 18},
    {"synset": "horse.n.01", "coco_cat_id": 19},
    {"synset": "sheep.n.01", "coco_cat_id": 20},
    {"synset": "beef.n.01", "coco_cat_id": 21},
    {"synset": "elephant.n.01", "coco_cat_id": 22},
    {"synset": "bear.n.01", "coco_cat_id": 23},
    {"synset": "zebra.n.01", "coco_cat_id": 24},
    {"synset": "giraffe.n.01", "coco_cat_id": 25},
    {"synset": "backpack.n.01", "coco_cat_id": 27},
    {"synset": "umbrella.n.01", "coco_cat_id": 28},
    {"synset": "bag.n.04", "coco_cat_id": 31},
    {"synset": "necktie.n.01", "coco_cat_id": 32},
    {"synset": "bag.n.06", "coco_cat_id": 33},
    {"synset": "frisbee.n.01", "coco_cat_id": 34},
    {"synset": "ski.n.01", "coco_cat_id": 35},
    {"synset": "snowboard.n.01", "coco_cat_id": 36},
    {"synset": "ball.n.06", "coco_cat_id": 37},
    {"synset": "kite.n.03", "coco_cat_id": 38},
    {"synset": "baseball_bat.n.01", "coco_cat_id": 39},
    {"synset": "baseball_glove.n.01", "coco_cat_id": 40},
    {"synset": "skateboard.n.01", "coco_cat_id": 41},
    {"synset": "surfboard.n.01", "coco_cat_id": 42},
    {"synset": "tennis_racket.n.01", "coco_cat_id": 43},
    {"synset": "bottle.n.01", "coco_cat_id": 44},
    {"synset": "wineglass.n.01", "coco_cat_id": 46},
    {"synset": "cup.n.01", "coco_cat_id": 47},
    {"synset": "fork.n.01", "coco_cat_id": 48},
    {"synset": "knife.n.01", "coco_cat_id": 49},
    {"synset": "spoon.n.01", "coco_cat_id": 50},
    {"synset": "bowl.n.03", "coco_cat_id": 51},
    {"synset": "banana.n.02", "coco_cat_id": 52},
    {"synset": "apple.n.01", "coco_cat_id": 53},
    {"synset": "sandwich.n.01", "coco_cat_id": 54},
    {"synset": "orange.n.01", "coco_cat_id": 55},
    {"synset": "broccoli.n.01", "coco_cat_id": 56},
    {"synset": "carrot.n.01", "coco_cat_id": 57},
    {"synset": "frank.n.02", "coco_cat_id": 58},
    {"synset": "pizza.n.01", "coco_cat_id": 59},
    {"synset": "doughnut.n.02", "coco_cat_id": 60},
    {"synset": "cake.n.03", "coco_cat_id": 61},
    {"synset": "chair.n.01", "coco_cat_id": 62},
    {"synset": "sofa.n.01", "coco_cat_id": 63},
    {"synset": "pot.n.04", "coco_cat_id": 64},
    {"synset": "bed.n.01", "coco_cat_id": 65},
    {"synset": "dining_table.n.01", "coco_cat_id": 67},
    {"synset": "toilet.n.02", "coco_cat_id": 70},
    {"synset": "television_receiver.n.01", "coco_cat_id": 72},
    {"synset": "laptop.n.01", "coco_cat_id": 73},
    {"synset": "mouse.n.04", "coco_cat_id": 74},
    {"synset": "remote_control.n.01", "coco_cat_id": 75},
    {"synset": "computer_keyboard.n.01", "coco_cat_id": 76},
    {"synset": "cellular_telephone.n.01", "coco_cat_id": 77},
    {"synset": "microwave.n.02", "coco_cat_id": 78},
    {"synset": "oven.n.01", "coco_cat_id": 79},
    {"synset": "toaster.n.02", "coco_cat_id": 80},
    {"synset": "sink.n.01", "coco_cat_id": 81},
    {"synset": "electric_refrigerator.n.01", "coco_cat_id": 82},
    {"synset": "book.n.01", "coco_cat_id": 84},
    {"synset": "clock.n.01", "coco_cat_id": 85},
    {"synset": "vase.n.01", "coco_cat_id": 86},
    {"synset": "scissors.n.01", "coco_cat_id": 87},
    {"synset": "teddy.n.01", "coco_cat_id": 88},
    {"synset": "hand_blower.n.01", "coco_cat_id": 89},
    {"synset": "toothbrush.n.01", "coco_cat_id": 90},
]


def save_coco_half_with_lvis_annotations(lvis_root, coco_half_root, save_path):
    # load lvis train json
    with open(os.path.join(lvis_root, 'lvis_v1_train.json')) as f:
        lvis_json = json.load(f)

    # load coco half train json
    with open(os.path.join(coco_half_root, 'coco_half_train.json')) as f:
        coco_json = json.load(f)

    lvis_annos = lvis_json.pop("annotations")
    lvis_json["annotations"] = lvis_annos

    # Mapping from lvis cat id to coco cat id via synset
    lvis_cat_id_to_synset = {cat["id"]: cat["synset"] for cat in lvis_json["categories"]}
    lvis_cat_id_to_name = {cat["id"]: cat["name"] for cat in lvis_json["categories"]}
    coco_cat_id_to_name = {cat["id"]: cat["name"] for cat in coco_json["categories"]}
    synset_to_coco_cat_id = {x["synset"]: x["coco_cat_id"] for x in COCO_SYNSET_CATEGORIES}
    # Synsets that we will keep in the dataset
    synsets_to_keep = set(synset_to_coco_cat_id.keys())

    # make copy of coco_half_train annotation file with lvis annotations
    coco_id_to_lvis_id = {}
    coco_name_to_lvis_name = {}
    for lvis_id, lvis_synset in lvis_cat_id_to_synset.items():
        lvis_name = lvis_cat_id_to_name[lvis_id]
        if lvis_synset in synsets_to_keep:
            coco_id = synset_to_coco_cat_id[lvis_synset]
            coco_id_to_lvis_id[coco_id] = lvis_id
            coco_name_to_lvis_name[coco_cat_id_to_name[coco_id]] = lvis_name
            

    # make json for base data (with lvis labels)
    base_dict = deepcopy(coco_json)
    for it in base_dict['categories']:
        coco_id = it['id']
        coco_name = it['name']
        if coco_id not in coco_id_to_lvis_id:  # hot dog not in lvis
            it['id'] = -1 
            it['name'] = 'hot dog'
            continue
        it['id'] = coco_id_to_lvis_id[coco_id]
        it['name'] = coco_name_to_lvis_name[coco_name]

    for it in base_dict['annotations']:
        coco_id = it['category_id']
        if coco_id not in coco_id_to_lvis_id:  # hot dog not in lvis
            it['category_id'] = -1
            continue
        it['category_id'] = coco_id_to_lvis_id[coco_id]
        
    # open a file for writing base dict
    with open(os.path.join(save_path, 'coco_half_train_lvis_ann.json'), 'w') as f:
        # write the dictionary to the file in JSON format
        print('\nSaving base_dict json to: ', os.path.join(save_path, 'coco_half_train_lvis_ann.json'))
        json.dump(base_dict, f)


if __name__ == '__main__':
    lvis_root = '/LVISv1/'
    coco_half_root = '/Microsoft-COCO-2017/annotations/'
    save_path = '/pandas_files'
    save_coco_half_with_lvis_annotations(lvis_root, coco_half_root, save_path)
    