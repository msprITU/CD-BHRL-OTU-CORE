import json
import shutil
import os
import copy
import glob
import cv2
import re

cls_id = 15
cls_name = "freestyle"
# gt_file = "/root/BHRL/data/VOT/augmented_imgs/{}/{}.txt". format(cls_name, cls_name)
gt_file = "/root/BHRL/data/VOT/augmented_imgs/{}/{}_no_illum_change.txt". format(cls_name, cls_name)
sample_data = "/root/BHRL/vot_annotation/ft/bike1_first_ft.json"
save_path = "/root/BHRL/vot_annotation/aug_anns/{}". format(cls_name+ "_aug_no_illum_change.json")
img_path = "/root/BHRL/data/VOT/augmented_imgs/{}/*.jpg". format(cls_name)

def prepare_anns(gt_file, img_path, sample_ann, main_save_path):
    f = open(sample_ann)
    data = json.load(f)  
    copy_voc_data = copy.deepcopy(data)

    with open(gt_file, "r") as file:
        lines = file.readlines()

    dirFiles = glob.glob(img_path)
    dirFiles.sort(key=lambda f: int(re.sub('\D', '', f)))
    for idx, img in enumerate(dirFiles):

        numbers_list = []
        line = lines[idx].strip()
        numbers = line.split(",")  # Split the line into individual numbers
        for num in numbers:
            numbers_list.append(int(num))  # C
        
        np_img = cv2.imread(img)
        images_templ = {
                        "file_name": img,
                        "height": np_img.shape[0],
                        "width": np_img.shape[1],
                        "id": idx+1
                        }
        anns_templ = {
                        "segmentation": [
                            [
                                47,
                                239,
                                47,
                                371,
                                195,
                                371,
                                195,
                                239
                            ]
                        ],
                        "area": np_img.shape[0] * np_img.shape[1],
                        "iscrowd": 0,
                        "image_id": idx+1,
                        "bbox": numbers_list,
                        "category_id": cls_id,
                        "id": idx+1,
                        "ignore": 0
                    }


        if not idx:
            copy_voc_data["images"] = [images_templ]
            copy_voc_data["annotations"] = [anns_templ]
        else:
            copy_voc_data["images"].append(images_templ)
            copy_voc_data["annotations"].append(anns_templ)

    json_object = json.dumps(copy_voc_data, indent=4)
    with open(save_path, "w") as outfile:
        outfile.write(json_object)

prepare_anns(gt_file, img_path, sample_data, save_path)