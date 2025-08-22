import numpy as np

def calculate_iou(bbox1, bbox2):

    bbox1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]
    bbox2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]

    # Calculate intersection
    x_inter_min = max(bbox1[0], bbox2[0])
    y_inter_min = max(bbox1[1], bbox2[1])
    x_inter_max = min(bbox1[2], bbox2[2])
    y_inter_max = min(bbox1[3], bbox2[3])
    inter_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)

    # Calculate union
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def read_file_and_identify_outliers(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the data
    bboxes = []
    for line in lines:
        parts = line.strip().split(',')
        frame_index = int(parts[0])
        bbox = list(map(float, parts[1:5]))  # Convert x1, y1, x2, y2 to float
        bboxes.append((frame_index, bbox))

    # Calculate IoU changes
    iou_changes = []
    for i in range(1, len(bboxes)):
        prev_bbox = bboxes[i-1][1]
        curr_bbox = bboxes[i][1]
        iou_change = calculate_iou(prev_bbox, curr_bbox)
        iou_changes.append(iou_change)

    # Standard Deviation Method for Outlier Detection
    # mean_iou = np.mean(iou_changes)
    # std_iou = np.std(iou_changes)
    # outliers = []
    # for i, iou_change in enumerate(iou_changes):
    #     if abs(iou_change - mean_iou) > 2 * std_iou:  # 2 standard deviations from the mean
    #         outliers.append((bboxes[i][0], bboxes[i][1]))  # Frame index and bbox

    # Z-Score Method for Outlier Detection
    mean_iou = np.mean(iou_changes)
    std_iou = np.std(iou_changes)
    z_scores = [(iou_change - mean_iou) / std_iou for iou_change in iou_changes]
    outliers = []
    threshold = 4  # Typically a Z-score of 2 or -2 is considered an outlier
    for i, z_score in enumerate(z_scores):
        if abs(z_score) > threshold:
            outliers.append((bboxes[i][0], bboxes[i][1]))  # Frame index and bbox
    # IQR Method for Outlier Detection
    # q1, q3 = np.percentile(iou_changes, [25, 75])
    # iqr = q3 - q1
    # lower_bound = q1 - (1.5 * iqr)
    # upper_bound = q3 + (1.5 * iqr)

    # outliers = []
    # for i, iou_change in enumerate(iou_changes):
    #     if iou_change < lower_bound or iou_change > upper_bound:
    #         outliers.append((bboxes[i][0], bboxes[i][1]))  # Frame index and bbox

    return outliers

# Usage
for i in range(40):
    file_path = '/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/results/person19/results_person19_part_{}.txt'.format(str(i))  # Replace with your file path
    outliers = read_file_and_identify_outliers(file_path)
    if len(outliers):
        print(i)
        print("Outliers:", outliers)

    