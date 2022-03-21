#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import io
import os
import random

import cv2
import json
import contextlib
import numpy as np
from easydict import EasyDict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data.data_augment import TrainTransform
from data.mosaicdetection import MosaicDetection
from data.datasets_wrapper import Dataset
import torch

class COCODataset(Dataset):
    """
    COCO dataset class.
    """

    def __init__(self,
                 data_dir=None,
                 json_file="instances_train2017.json",
                 name="train2014",
                 img_size=(416, 416),
                 tracking=False,
                 preproc=None,
                 ):
        super().__init__(img_size)
        self.data_dir = data_dir
        self.json_file = json_file
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.tracking = tracking
        #################
        # self.name = "val2017"
        # self.json_file = self.json_file.replace("train", "val")
        #################
        assert os.path.isfile(json_file), 'cannot find {}'.format(json_file)
        print("==> Loading annotation {}".format(json_file))
        self.coco = COCO(self.json_file)
        self.ids = self.coco.getImgIds()
        print("images number {}".format(len(self.ids)))
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes = [c["name"] for c in cats]
        self.annotations = self._load_coco_annotations()

        if "val" in self.name:
            print("classes index:", self.class_ids)
            print("class names in dataset:", self.classes)

    def __len__(self):
        return len(self.ids)

    def convert_eval_format(self, all_bboxes):
        detections = []
        for image_id in all_bboxes.keys():
            one_img_res = all_bboxes[image_id]
            for res in one_img_res:
                cls, conf, bbox = res[0], res[1], res[2]
                detections.append({
                    'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                    'category_id': self.class_ids[self.classes.index(cls)],
                    'image_id': int(image_id),
                    'score': float(conf)})
        return detections

    def run_coco_eval(self, results, save_dir):
        json.dump(self.convert_eval_format(results), open('{}/results.json'.format(save_dir), 'w'))
        coco_det = self.coco.loadRes('{}/results.json'.format(save_dir))

        coco_eval = COCOeval(self.coco, coco_det, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()

        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            coco_eval.summarize()
        str_result = redirect_string.getvalue()
        ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large = coco_eval.stats[:6]
        return ap, ap_0_5, ap_7_5, ap_small, ap_medium, ap_large, str_result

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj["bbox"][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj["bbox"][3] - 1))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)
        res = np.zeros((num_objs, 6 if self.tracking else 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls
            if self.tracking:
                assert "tracking_id" in obj.keys(), 'cannot find "tracking_id" in your dataset'
                res[ix, 5] = obj['tracking_id']
                # print('errorrrrrrrr: replace tracking_id to cls')
                # res[ix, 5] = cls

        img_info = (height, width)
        file_name = im_ann["file_name"]

        del im_ann, annotations

        return res, img_info, file_name

    def load_anno(self, index):
        return self.annotations[index][0]
    def getBgImgByFilename(self,file_name):
        tmpstr = file_name.split('.')[0]
        tmpstr = tmpstr.split('_')
        videoname = '_'.join(tmpstr[0:-1])
        if 'train' in self.name:
            bg_img_file = "/home/tony/CLEF2015 Fish detection/training_set/video_bg/" +videoname+'.jpg'
        else:
            bg_img_file = "/home/tony/CLEF2015 Fish detection/test_set/video_bg/" + videoname + '.jpg'
        img = cv2.imread(bg_img_file)
        assert img is not None, "error img {}".format(bg_img_file)
        return img
    def pull_item(self, index):
        id_ = self.ids[index]

        res, img_info, file_name = self.annotations[index]
        # load image and preprocess
        img_file = self.data_dir + "/" + self.name + "/" + file_name
        img_bg = self.getBgImgByFilename(file_name)
        img = cv2.imread(img_file)
        assert img is not None, "error img {}".format(img_file)

        return img, res.copy(), img_info, id_,img_bg

    @Dataset.resize_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id,img_bg = self.pull_item(index)

        if self.preproc is not None:
            betaflag = random.randrange(2)
            beta = random.uniform(-32, 32)
            alphaflag = random.randrange(2)
            alpha = random.uniform(0.5, 1.5)
            param1flag = random.randrange(2)
            param1 = random.randint(-18, 18)
            param2flag = random.randrange(2)
            param2 = random.uniform(0.5, 1.5)
            mirrorFlag = random.randrange(2)
            img, target = self.preproc(img, target, self.input_dim,betaflag,beta,alphaflag,alpha,param1flag,param1,param2flag,param2,mirrorFlag)
            img_bg, _ = self.preproc(img_bg, target, self.input_dim,betaflag,beta,alphaflag,alpha,param1flag,param1,param2flag,param2,mirrorFlag)


        targets={}
        targets['boxes']=torch.Tensor(target[:,1:])
        targets['labels']=torch.Tensor(target[:,0]+1).type(torch.int64)

        return img, targets, img_info, img_id,img_bg
def unproc(img):
    img = np.transpose(img,(1,2,0))*255

    return img
if __name__ == '__main__':
    opt = EasyDict()
    opt.dataset_path = "/media/tony/系统文件/DataSets/fish_COCO"
    opt.data_dir = opt.dataset_path + "/images"
    opt.val_ann = opt.dataset_path + "/annotations/instances_val2014.json"
    opt.test_size = (480, 640)
    opt.input_size = (480, 640)
    opt.rgb_means = [0.485, 0.456, 0.406]
    opt.std = [0.229, 0.224, 0.225]
    opt.degrees = 10.0  # rotate angle
    opt.translate = 0.1
    opt.scale = (0.1, 2)
    opt.shear = 2.0
    opt.perspective = 0.0
    val_dataset = COCODataset(
        data_dir=opt.data_dir,
        json_file=opt.val_ann,
        name="val2014",
        img_size=opt.test_size,
        tracking=False,
        preproc=TrainTransform(rgb_means=None, std=None, max_labels=120, tracking=False,
                               augment=False))
    val_dataset = MosaicDetection(
        val_dataset,
        mosaic=True,
        img_size=opt.input_size,
        preproc=TrainTransform(rgb_means=opt.rgb_means, std=opt.std, max_labels=110, tracking=False),
        degrees=opt.degrees,
        translate=opt.translate,
        scale=opt.scale,
        shear=opt.shear,
        perspective=opt.perspective,
        enable_mixup=True,
        tracking=False,
    )
    img, target, img_info, img_id,bg_img = val_dataset.__getitem__(11)
    path = '/media/tony/系统/FishProject/frcnn_bg_v2'

    cv2.imwrite(path+'1.jpg',unproc(img))
    cv2.imwrite(path+'2.jpg',unproc(bg_img))
