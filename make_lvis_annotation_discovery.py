# ---
# PANDAS
# Copyright (C) 2023 NAVER Corp.
# CC BY-NC-SA 4.0 license
# ---

import os
import json
from copy import deepcopy


def make_discovery_annotation_file(lvis_root, coco_half_root, save_path):
    
    # load lvis train json
    with open(os.path.join(lvis_root, 'lvis_v1_train.json')) as f:
        lvis_train = json.load(f)
        
    # load coco half train json
    with open(os.path.join(coco_half_root, 'coco_half_train.json')) as f:
        coco_half_train = json.load(f)
    
    # make json file for discovery data
    coco_half_ids = [coco_half_train['images'][i]['id'] for i in range(len(coco_half_train['images']))]
    lvis_ids = [lvis_train['images'][i]['id'] for i in range(len(lvis_train['images']))]
    discovery_ids = list(set(lvis_ids) - set(coco_half_ids))  # determine discovery images
    discovery_images = []
    for img in lvis_train['images']:
        curr_id = img['id']
        if curr_id in discovery_ids:
            discovery_images.append(img)
    discovery_dict = deepcopy(lvis_train)
    discovery_dict['images'] = discovery_images 
    
    # open a file for writing discovery dict
    with open(os.path.join(save_path, 'coco_second_half_train_lvis_ann.json'), 'w') as f:
        # write the dictionary to the file in JSON format
        print('\nSaving discovery_dict json to: ', os.path.join(save_path, 'coco_second_half_train_lvis_ann.json'))
        json.dump(discovery_dict, f) 


if __name__ == '__main__':
    lvis_root = '/LVISv1/'
    coco_root = '/Microsoft-COCO-2017/annotations/'
    save_path = '/pandas_files'
    make_discovery_annotation_file(lvis_root, coco_root, save_path)
    
