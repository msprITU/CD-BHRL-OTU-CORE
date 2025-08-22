
import matplotlib.pyplot as plt
import numpy as np
import sys

import cv2 
import os
import glob
import json

def main():
    seq = sys.argv[1]
    predictions_folder = '/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/results/{}/results*'. format(seq)
    img_folder = "/root/BHRL/data/VOT/{}_imgs". format(seq) 
    out_folder = "/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/result_imgs/{}". format(seq)
    ann_file = "/root/BHRL/vot_annotation/{}/vot_{}_test.json". format(seq, seq)

    file_paths = glob.glob(predictions_folder)

    json_file = open(ann_file, "r")
    voc_data = json.load(json_file)

    sorted_file_paths = sorted(file_paths, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    for part_idx, predictions_filename in enumerate(sorted_file_paths):
        print(predictions_filename)
        with open(predictions_filename, 'r') as pred_file:
            predictions_lines = pred_file.readlines()

            num_samples = len(predictions_lines)

            for i in range(num_samples):
                pred_line = predictions_lines[i].strip().split(',')

                pred_box = list(map(float, pred_line[1:5]))
                score = float(pred_line[5])
                if not part_idx:
                    img_idx = int(pred_line[0]) -1
                else:
                    img_idx = int(pred_line[0]) -2

                img_idx += part_idx*100
                img_idx = voc_data["images"][img_idx]["id"]
                img_path = os.path.join(img_folder, str(img_idx).zfill(8) + ".jpg")
                img = cv2.imread(img_path)

                cv2.rectangle(img, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[0] + pred_box[2]), int(pred_box[1] + pred_box[3])), (0,0,255), 3)
                img = cv2.putText(img, 
                    str(score), 
                    (int(pred_box[0])+15, int(pred_box[1])+15), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=1, 
                    color=(255, 255, 255), 
                    thickness=3, 
                    lineType=cv2.LINE_AA, 
                    bottomLeftOrigin=False)
                img = cv2.putText(img, str(img_idx), (25, 50), cv2.FONT_HERSHEY_SIMPLEX ,  
                                2, (0,0,255), 6, cv2.LINE_AA) 
                
                save_path = os.path.join(out_folder, str(img_idx).zfill(8) + ".jpg")
                cv2.imwrite(save_path, img)


if __name__ == "__main__":
    main()
