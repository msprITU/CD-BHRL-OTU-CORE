import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

import mmcv
import math
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
# from pycocotools.coco import COCO
from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS

from .custom import CustomDataset

import copy
import random
import time

def find_nth(string, substring, n):
	"""Searches substring in the string.
	Args:
		string           = String in which the substring is searched.
		substring        = String to be searched.
		n                = The number indicating which string will be searched.
	"""
	if (n == 1):
		return string.find(substring)
	else:
		return string.find(substring, find_nth(string, substring, n - 1) + 1)

def calculate_iou(box1, box2):
    # Extract coordinates of the bounding boxes
    x1_1, y1_1, x2_1, y2_1 = box1[:-1]
    x1_2, y1_2, x2_2, y2_2 = box2[:-1]

    # Calculate the area of each bounding box
    area_box1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_box2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Calculate the coordinates of the intersection rectangle
    x_intersection = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
    y_intersection = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))

    # Calculate the area of intersection
    area_intersection = x_intersection * y_intersection

    # Calculate the union area
    area_union = area_box1 + area_box2 - area_intersection

    # Calculate the IoU
    iou = area_intersection / area_union

    return iou


@DATASETS.register_module()
class OneShotVOCDataset(CustomDataset):
    if False:
      CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                'tvmonitor')
    CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt',
        'bridge', 'chimney', 'dam', 'Expressway-Service-area',
        'Expressway-toll-station', 'golffield', 'groundtrackfield',
        'harbor', 'overpass', 'ship', 'stadium', 'storagetank',
        'tenniscourt', 'trainstation', 'vehicle', 'windmill')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 test_seen_classes=False,
                 position=0,
                 requested_class=None):
        self.split = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        #self.split=[]
        self.test_seen_classes = test_seen_classes
        #self.test_seen_classes = False
        self.position = 0
        classes = None 
        super(OneShotVOCDataset,
              self).__init__(ann_file, pipeline, classes, data_root, img_prefix,
                             seg_prefix, proposal_file, test_mode)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.split_cats()
        self.img_ids = self.coco.get_img_ids()
        img_infos, img_cates = self.generate_infos()
        self.cates = img_cates
        return img_infos

    def split_cats(self):
        self.train_cat = []
        self.test_cat = []
        for i in range(len(self.cat_ids)):
            if (i + 1) in self.split:
                self.test_cat.append(self.cat_ids[i])
                #self.train_cat.append(self.cat_ids[i])
            else:
                self.train_cat.append(self.cat_ids[i])
        if self.test_seen_classes:
            self.test_cat = self.train_cat

        self.train_cat = self.test_cat


    def generate_infos(self):
        img_infos = []
        img_cates = []
        for i in self.img_ids:
            if not self.test_mode:
                img_infos, img_cates = self.generate_train(i, img_infos) 
            else:
                img_infos, img_cates = self.generate_test(i, img_infos, img_cates)
        return img_infos, img_cates

    def generate_train(self, i, img_infos):
        info = self.coco.load_imgs([i])[0]
        info['filename'] = info['file_name']
        img_anns_ids = self.coco.get_ann_ids(img_ids=[i])
        img_anns = self.coco.load_anns(img_anns_ids)
        for img_ann in img_anns:
            if img_ann['category_id'] in self.train_cat:
                img_infos.append(info)
                break
        return img_infos, None

    def generate_test(self, i, img_infos, img_cates):
        info = self.coco.loadImgs([i])[0]
        info['filename'] = info['file_name']
        img_anns_ids = self.coco.getAnnIds(imgIds=i)
        img_anns = self.coco.loadAnns(img_anns_ids)
        img_cats = list()
        for img_ann in img_anns:
            if img_ann['category_id'] in img_cats:
                continue
            elif img_ann['category_id'] in self.test_cat:
                img_cats.append(img_ann['category_id'])
                img_infos.append(info)
                img_cates.append(img_ann['category_id'])
            else:
                continue
        return img_infos, img_cates

    def get_ann_info(self, idx, cate=None):
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids) 
        if cate is None: 
            cate = self.random_cate(ann_info)
        return self._parse_ann_info(self.data_infos[idx], ann_info, cate)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        ids_in_cat &= ids_with_ann
        
        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def random_cate(self, ann_info):
        index = np.random.randint(len(ann_info))
        cate = ann_info[index]['category_id']

        if not self.test_mode:
            cates = self.train_cat
        else:
            cates = self.test_cat

        while cate not in cates:
            index = np.random.randint(len(ann_info))
            cate = ann_info[index]['category_id']
        return cate

    def _parse_ann_info(self, img_info, ann_info, cate_select):
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            elif ann['category_id'] == cate_select:
                gt_bboxes.append(bbox)
                gt_labels.append(0)
                gt_masks_ann.append(ann['segmentation'])
            else:
                continue

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(bboxes=gt_bboxes,
                   labels=gt_labels,
                   bboxes_ignore=gt_bboxes_ignore,
                   masks=gt_masks_ann,
                   seg_map=seg_map,
                   cate=cate_select)

        return ann

    def prepare_train_ref_img(self, idx, cate):
        rf_img_info = dict()
        img_id = self.data_infos[idx]['id']
        rf_ids = self.coco.getImgIds(catIds=[cate])
        while True:
            rf_id = rf_ids[np.random.randint(0, len(rf_ids))]
            while rf_id == img_id and len(rf_ids) > 1:
                rf_id = rf_ids[np.random.randint(0, len(rf_ids))]
            rf_anns = self.coco.loadAnns(
                self.coco.getAnnIds(imgIds=rf_id, catIds=cate, iscrowd=False))
            if len(rf_anns) > 0:
                rand_index = np.random.randint(len(rf_anns))
                rf_img_info['ann'] = rf_anns[rand_index]
                rf_img_info['file_name'] = self.coco.loadImgs([rf_id])[0]['file_name']
                break
        rf_img_info['img_info'] = self.coco.loadImgs([rf_id])[0]
        return rf_img_info

    def prepare_test_ref_img(self, idx, cate):
        rf_img_info = dict()
        img_id = self.data_infos[idx]['id']
        rf_ids = self.coco.getAnnIds(catIds=[cate], iscrowd=False)

        random.seed(img_id)
        l = list(range(len(rf_ids)))
        random.shuffle(l)

        position = 0
        #position = idx
        ref = rf_ids[position]

        rf_anns = self.coco.loadAnns(ref)[0]
        rf_img_info['ann'] = rf_anns
        rf_img_info['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
        rf_img_info['img_info'] = self.coco.loadImgs(rf_anns['image_id'])[0]

        seq = rf_img_info['file_name'][find_nth(rf_img_info['file_name'], "/", 7) + 1 : find_nth(rf_img_info['file_name'], "/", 8)]
        seq = seq[:seq.find("_")]

        if idx >= 50:
            with open("/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_aug/updates/updates_{}.txt". format(seq), "r") as f_updates:
                lines = f_updates.readlines()
            f_updates.close()
            last_second_line = lines[-1].strip()
            integer_list = [float(element) for element in last_second_line.split(",")]

        elif idx == 49:
            dets = np.load("/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_aug/preds/{}.npy". format(seq))
            last_det = dets[-1]
            x = last_det[0]
            y = last_det[1]
            w = last_det[2] - last_det[0]
            h = last_det[3] - last_det[1]
            f_updates = open("/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_aug/updates/updates_{}.txt". format(seq), "a")
            f_updates.write(str(49)  + "," + str(x) + "," + str(y) + "," + str(w) + "," + str(h) + "\n")
            f_updates.close()
            with open("/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_aug/updates/updates_{}.txt". format(seq), "r") as f_updates:
                lines = f_updates.readlines()
            f_updates.close()
            last_second_line = lines[-1].strip()
            integer_list = [float(element) for element in last_second_line.split(",")]
        
        if not ((idx+1) % 50):
            max_retries = 3  # You can adjust the number of retries
            retry_count = 0

            while retry_count < max_retries:
                try:
                    dets = np.load("/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_aug/preds/{}.npy". format(seq))
                    break
                except:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying in 5 seconds... (Attempt {retry_count}/{max_retries})")
                        time.sleep(5)
                    else:
                        print("Max retries reached. Operation failed.")

            last_det = dets[-1]
            dets = dets[-50:]
            sorted_indices = sorted(range(len(dets)), key=lambda i: dets[i][4], reverse=True)
            best_idx = sorted_indices[0]
            best_det = dets[best_idx]
            iou = calculate_iou(last_det, best_det)

            if (best_det[4] > 0.85) and (iou > 0.7):
                x = best_det[0]
                y = best_det[1]
                w = best_det[2] - best_det[0]
                h = best_det[3] - best_det[1]
                f = open("/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_aug/observer/observer_{}.txt". format(seq), "a")
                f.write(str(idx+1) + "," + str(x) + "," + str(y) + "," + str(w) + "," + str(h) + "," + str(best_det[4]) + "," + str(w*h) + "," + str(best_idx+1) + "\n")
                f.close()
                rf_img_info["ann"]['bbox'] = [x,y,w,h]
                rf_img_info["ann"]['area'] = w*h
                ref = rf_ids[best_idx + ( 50 * math.floor((idx/50)) ) ]
                rf_anns = self.coco.loadAnns(ref)[0]
                rf_img_info["ann"]['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
                rf_img_info['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
                rf_img_info['img_info'] = self.coco.loadImgs(rf_anns['image_id'])[0]
                f_updates = open("/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_aug/updates/updates_{}.txt". format(seq), "a")
                f_updates.write(str(best_idx + ( 50 * math.floor((idx/50))))  + "," + str(x) + "," + str(y) + "," + str(w) + "," + str(h) + "\n")
                f_updates.close()
            else:
                pos = integer_list[0]
                ref = rf_ids[int(pos)]
                rf_anns = self.coco.loadAnns(ref)[0]
                rf_img_info["ann"]['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
                rf_img_info['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
                rf_img_info['img_info'] = self.coco.loadImgs(rf_anns['image_id'])[0]
                rf_img_info["ann"]['bbox'] = [integer_list[1],integer_list[2],integer_list[3],integer_list[4]]
                rf_img_info["ann"]['area'] = integer_list[3]*integer_list[4]

        elif (idx >= 50) and ((idx+1) % 50):
            pos = integer_list[0]
            ref = rf_ids[int(pos)]
            rf_anns = self.coco.loadAnns(ref)[0]
            rf_img_info["ann"]['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
            rf_img_info['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
            rf_img_info['img_info'] = self.coco.loadImgs(rf_anns['image_id'])[0]
            rf_img_info["ann"]['bbox'] = [integer_list[1],integer_list[2],integer_list[3],integer_list[4]]
            rf_img_info["ann"]['area'] = integer_list[3]*integer_list[4]

        return rf_img_info

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        rf_img_info = self.prepare_train_ref_img(idx, ann_info['cate'])
        results = dict(img_info=img_info,
                       ann_info=ann_info,
                       rf_img_info=rf_img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx, self.cates[idx])
        rf_img_info = self.prepare_test_ref_img(idx, self.cates[idx])
        results = dict(img_info=img_info,
                       ann_info=ann_info,
                       rf_img_info=rf_img_info,
                       label=self.cat2label[self.cates[idx]])
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)
