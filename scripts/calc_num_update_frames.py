import os
import glob
import numpy as np

main_folder = "/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_200/update_frames"
print(len(os.listdir(main_folder)))
values = []
for sub_folder in os.listdir(main_folder):
    folder = "/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_200/update_frames/{}/*.jpg". format(sub_folder)
    print(sub_folder, len(glob.glob(folder))-1) 
    values.append(len(glob.glob(folder))-1)

print(np.mean(values))