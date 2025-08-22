import glob
import numpy as np
import sys
import os


#PARTS

seq = sys.argv[1]
#folder = "/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/logs/{}/{}_part*". format(seq, seq)
main_folder = "/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/voc/logs/"
print(len(os.listdir(main_folder)))
main_mean_map_values = []
for sub_folder in os.listdir(main_folder):
    folder = "/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/voc/logs/{}/{}_part*". format(sub_folder, sub_folder)
    map_values = []
    for file in glob.glob(folder): 
        with open(file, 'r') as map_file:
            map_lines = map_file.readlines()
            try:
                map_values.append(float(map_lines[-11][map_lines[-11].find("]")+4:].strip()))
            except:
                print("########################################################################")
                print(file)
                map_values.append(0)
                continue
    main_mean_map_values.append(np.mean(map_values))
    print(sub_folder, np.mean(map_values))
print("DONE")
print(np.mean(main_mean_map_values))



# main_folder = "/root/BHRL/vot_results/real_bhrl_voc_eval/logs/"
# print(len(os.listdir(main_folder)))
# main_mean_map_values = []
# seq_means = []
# avg = 5
# seq_cnt=0
# for sub_folder in sorted(os.listdir(main_folder)):
#     seq = sub_folder.split("_")[0]
#     seq_cnt += 1
#     file = os.path.join(main_folder, sub_folder)
#     with open(file, 'r') as map_file:
#         map_lines = map_file.readlines()
#         try:
#             seq_means.append(float(map_lines[-11][map_lines[-11].find("]")+4:].strip()))
#         except:
#             print("########################################################################", file)
#             continue
#     if seq_cnt == 5:
#         if len(seq_means) != 5:
#             print("NOT COMPLETED", seq, len(seq_means))
#         seq_cnt=0
#         main_mean_map_values.append(np.mean(seq_means))
#         print(seq, np.mean(seq_means), seq_means)
#         seq_means = []

# print("DONE")
# print(np.mean(main_mean_map_values))