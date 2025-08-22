import json
import shutil
import os
import copy

cls_id = 3
cls_name = "bird1"
gt_file = "/root/BHRL/data/VOCdevkit/voc_annotation/voc_test.json"
main_voc_path = "/root/BHRL/data/VOCdevkit"
sample_data = "/root/BHRL/vot_annotation/ft/bird1_first_ft.json"
save_path = "/root/BHRL/work_dirs/VOC/{}/ann/". format(cls_name)

if not os.path.exists("/root/BHRL/work_dirs/VOC/{}/imgs". format(cls_name)):
    os.makedirs("/root/BHRL/work_dirs/VOC/{}/imgs". format(cls_name))
    os.makedirs("/root/BHRL/work_dirs/VOC/{}/ann". format(cls_name))

def read_gt(gt_file, cls_id, max_count):
    image_ids = []
    bboxes = []
    ann_dicts = []
    f = open(gt_file)
    data = json.load(f)
    for key in data:
        if key == "annotations":
            for ann in data[key]:
                if ann["category_id"] == cls_id:
                    if not ann["image_id"] in image_ids and ann["image_id"] > 370:
                        image_ids.append(ann["image_id"])
                        bboxes.append(ann["bbox"])
                        ann_dicts.append(ann)
                    else:
                        pass

                    if len(image_ids) == max_count:
                        break
                    else:
                        continue
    return image_ids, bboxes, ann_dicts

def get_images(gt_file, image_ids):
    file_names = []
    f = open(gt_file)
    data = json.load(f)
    image_dicts = []
    for key in data:
        if key == "images":
            for img in data[key]:
                if img["id"] in image_ids:
                    file_names.append(img["file_name"])
                    image_dicts.append(img)
    return file_names, image_dicts

def copy_images(main_voc_path, file_names):
    for file_name in file_names:
        image_path = os.path.join(main_voc_path, file_name)
        shutil.copy(image_path, "/root/BHRL/work_dirs/VOC/{}/imgs/{}". format(cls_name, os.path.split(image_path)[-1]))

def prepare_anns(sample_ann, main_save_path, ann_dicts, image_dicts):
    f = open(sample_ann)
    data = json.load(f)  

    for idx, img in enumerate(image_dicts): 
        copy_voc_data = copy.deepcopy(data)
        copy_voc_data["images"] = [img]
        copy_voc_data["annotations"] = [ann_dicts[idx]]
        save_path = os.path.join(main_save_path, str(img["id"]).zfill(6) + ".json")

        json_object = json.dumps(copy_voc_data, indent=4)
        with open(save_path, "w") as outfile:
            outfile.write(json_object)



image_ids, bboxes, ann_dicts = read_gt(gt_file, cls_id, 1)
file_names, image_dicts = get_images(gt_file, image_ids)
copy_images(main_voc_path, file_names)
prepare_anns(sample_data, save_path, ann_dicts, image_dicts)