
import numpy as np
import sys

import cv2 
import os
import json



def calculate_iou(box1, box2):
    # Extract coordinates of the bounding boxes
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

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

def main():

    take_first_target = False
    seq = sys.argv[1]
    part = sys.argv[2]
    part_size = int(sys.argv[3])
    part_filename = '/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/voc/results/{}/results_{}_part_{}.txt'. format(seq, seq, part)
    img_folder = "/root/BHRL/data/VOT/{}_imgs". format(seq) 
    out_folder = "/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/voc/update_frames/{}". format(seq)
    ann_file = "/root/BHRL/vot_annotation/{}/vot_{}_test.json". format(seq, seq)
    template_json = "/root/BHRL/vot_annotation/ft/{}_first_ft.json". format(seq)
    last_update_json = "/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/voc/update_frames/{}/target_update.json". format(seq)

    json_file = open(ann_file, "r")
    voc_data = json.load(json_file)

    json_file = open(template_json, "r")
    template_data = json.load(json_file)

    next_part_filename = '/root/BHRL/vot_annotation/{}/{}_part_{}.json'. format(seq, seq, str(int(part)+1))
    json_file = open(next_part_filename, "r")
    next_part_data = json.load(json_file)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    with open(part_filename, 'r') as pred_file:
        pred_lines = pred_file.readlines()
    num_samples = len(pred_lines)

    scores = []
    bboxes = []

    for i in range(num_samples):
        pred_line = pred_lines[i].strip().split(',')

        pred_bbox = list(map(float, pred_line[1:5]))
        score = float(pred_line[5])
        scores.append(score)
        bboxes.append(pred_bbox)

        
    best_score = max(scores)
    best_idx = scores.index(best_score)
    print("best score idx = ", best_idx + (int(part) * part_size))
    best_bbox = bboxes[best_idx]
    best_bbox_xyxy = [int(best_bbox[0]), int(best_bbox[1]), int(best_bbox[0] + best_bbox[2]), int(best_bbox[1] + best_bbox[3])]

    last_score = scores[-1]
    last_bbox = bboxes[-1]
    last_idx = num_samples-1
    last_bbox_xyxy = [int(last_bbox[0]), int(last_bbox[1]), int(last_bbox[0] + last_bbox[2]), int(last_bbox[1] + last_bbox[3])]


    print("LAST BBOX: ", last_bbox_xyxy)
    print("BEST BBOX: ", best_bbox_xyxy)

    if all(v == 0 for v in best_bbox_xyxy):
        take_first_target = True
        print("BEST DETECTION IS ZERO! - TAKE INITIAL FRAME AS TARGET")
    else:
        if all(v == 0 for v in last_bbox_xyxy):
            print("BEST DETECTION IS NOT ZERO! - LAST DETECTION IS ZERO!")
            if best_score > 0.85:
                iou = 1.0
                print("SCORE {} IS HIGHER THAN 0.85 ! TAKE BEST DETECTION AS TARGET". format(best_score))
            else:
                take_first_target = True
                print("SCORE {} IS NOT HIGHER THAN 0.85 ! TAKE INITIAL FRAME AS TARGET". format(best_score))
        else:
            print("BEST & LAST DETECTION IS NOT ZERO !")
            iou = calculate_iou(last_bbox_xyxy, best_bbox_xyxy)
            if iou > 0.7:
                print("IOU {} IS HIGHER THAN 0.7". format(iou))
                if best_score > 0.85:
                    iou = 1.0
                    print("SCORE {} IS HIGHER THAN 0.85 ! TAKE BEST DETECTION AS TARGET". format(best_score))
                else:
                    if last_score > 0.85:
                        print("SCORE {} IS HIGHER THAN 0.85 ! TAKE LAST DETECTION AS TARGET". format(last_score))
                        iou = 0.0
                    else:
                        print("SCORE {} IS NOT HIGHER THAN 0.85 ! TAKE BEST DETECTION AS TARGET". format(last_score))
                        take_first_target = True
            else:
                print("IOU {} IS NOT HIGHER THAN 0.7". format(iou))
                if last_score > 0.85:
                    print("SCORE {} IS HIGHER THAN 0.85 ! TAKE LAST DETECTION AS TARGET". format(last_score))
                    iou = 0.0
                else:
                    print("SCORE {} IS NOT HIGHER THAN 0.85 ! TAKE BEST DETECTION AS TARGET". format(last_score))
                    take_first_target = True

    if take_first_target:
        best_score = 1.0
        best_img_idx = 1
        json_file = open(template_json, "r")
        first_target_data = json.load(json_file)
        best_bbox = first_target_data["annotations"][0]["bbox"]
    else:
        if iou > 0.7:
            print("Best scenario!")
        else:
            best_score = last_score
            best_idx = last_idx
            best_bbox = last_bbox

        best_idx = best_idx + (int(part) * part_size)
        best_img_idx = voc_data["images"][best_idx]["id"]

    print("AT FINAL:")
    print("Updted img idx: ", best_img_idx)
    print("Updted img bbox: ", best_bbox)
    print("Updted img score: ",best_score)


    # Img saving
    img_path = os.path.join(img_folder, str(best_img_idx).zfill(8) + ".jpg")
    img = cv2.imread(img_path)              
    cv2.rectangle(img, (int(best_bbox[0]), int(best_bbox[1])), (int(best_bbox[0] + best_bbox[2]), int(best_bbox[1] + best_bbox[3])), (0,0,255), 3)
    img = cv2.putText(img, 
        str(best_score), 
        (int(best_bbox[0])+15, int(best_bbox[1])+15), 
        fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
        fontScale=0.5, 
        color=(255, 255, 255), 
        thickness=1, 
        lineType=cv2.LINE_AA, 
        bottomLeftOrigin=False)
    save_path = os.path.join(out_folder, str(best_img_idx).zfill(8) + ".jpg")
    cv2.imwrite(save_path, img)


    template_data["images"][0]["file_name"] = "/root/BHRL/data/VOT/{}_imgs/{}.jpg". format(seq, str(best_img_idx).zfill(8))
    template_data["images"][0]["id"] = best_img_idx
    template_data["annotations"][0]["area"] = best_bbox[2] * best_bbox[3]
    template_data["annotations"][0]["bbox"] = [int(best_bbox[0]), int(best_bbox[1]), int(best_bbox[2]), int(best_bbox[3])]
    template_data["annotations"][0]["image_id"] = best_img_idx


    json_object = json.dumps(template_data, indent=4)
    with open(os.path.join(out_folder, "target_update.json"), "w") as outfile:
        outfile.write(json_object)

    next_part_data["images"].insert(0, template_data["images"][0])
    next_part_data["annotations"].insert(0, template_data["annotations"][0])

    json_object = json.dumps(next_part_data, indent=4)
    with open(next_part_filename, "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main()
