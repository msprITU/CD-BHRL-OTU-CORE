
import numpy as np
seq = "person19"
dets = np.load("/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/preds/person19_part_0.txt.npy". format(seq))
print(dets[0])