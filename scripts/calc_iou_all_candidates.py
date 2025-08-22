
import matplotlib.pyplot as plt
import numpy as np
import sys



def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xmin = max(x1, x2)
    ymin = max(y1, y2)
    xmax = min(x1 + w1, x2 + w2)
    ymax = min(y1 + h1, y2 + h2)

    intersection_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    area_box1 = w1 * h1
    area_box2 = w2 * h2
    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area
    return iou

def main():
    seq = sys.argv[1]
    epoch = sys.argv[2]
    ious = []
    scores = []
    indexes = []
    groups = {}
    max_elements= []
    final_scores = []
    predictions_filename = '/root/BHRL/target_update_studies/real_time/{}/results/{}_all_candidates.txt'. format(epoch,seq)
    ground_truth_filename = '/root/BHRL/data/VOT/{}/groundtruth.txt'. format(seq)

    with open(predictions_filename, 'r') as pred_file, open(ground_truth_filename, 'r') as gt_file:
        predictions_lines = pred_file.readlines()
        ground_truth_lines = gt_file.readlines()
        print(len(predictions_lines))
        print(len(ground_truth_lines))
        # assert len(predictions_lines) == len(ground_truth_lines), "Number of predictions and ground truths do not match"

        total_iou = 0.0
        num_samples = len(predictions_lines)

        for i in range(num_samples):
            pred_line = predictions_lines[i].strip().split(',')
            # gt_line = ground_truth_lines[i].strip().split(',')

            key = int(pred_line[0])

            if key in groups:
                groups[key].append(pred_line)
            else:
                groups[key] = [pred_line]

            
            # pred_box = list(map(float, pred_line[1:5]))
            # gt_box = list(map(float, gt_line))
            # score = float(pred_line[5])

            # iou = calculate_iou(pred_box, gt_box)
            # total_iou += iou

            # print(f"IoU for sample {i+1}: {iou:.4f}")
            # ious.append(iou)
            # scores.append(score)
        # Now, you can access each group by its key

        for key, group in groups.items():
            print(f"Group {key}:")
            gt_line = ground_truth_lines[int(key)-1].strip().split(',')
            gt_box = list(map(float, gt_line))
            ious = []
            for candidate in group:
                pred_box = list(map(float, candidate[1:5]))
                iou = calculate_iou(pred_box, gt_box)
                ious.append(iou)
                scores.append(float(candidate[-1]))
            max_element = max(ious)
            max_index = ious.index(max_element)
            indexes.append(max_index+1)
            max_elements.append(max_element)
            final_scores.append(scores[max_index]*10) 

        print(len(indexes))
    
        equal_to = np.array(indexes) == 1
        count_1 = np.sum(equal_to)
        masked_scores_1 = np.array(final_scores)[equal_to] / 10
        masked_ious_1 = np.array(max_elements)[equal_to]
        equal_to = np.array(indexes) == 2
        count_2 = np.sum(equal_to)
        masked_scores_2 = np.array(final_scores)[equal_to] / 10
        masked_ious_2 = np.array(max_elements)[equal_to]
        equal_to = np.array(indexes) == 3
        count_3 = np.sum(equal_to)
        masked_scores_3 = np.array(final_scores)[equal_to] / 10
        masked_ious_3 = np.array(max_elements)[equal_to]

    x_values = list(range(1, len(indexes)+1))

    # Plot the data from both lists on the same graph
    plt.plot(x_values, indexes, label='Max IoU at first idx: {} (%{:.2f}) (Avg. Score {:.2f}) (Avg. IoU {:.2f}) \n Max IoU at second idx: {} (%{:.2f}) (Avg. Score {:.2f}) (Avg. IoU {:.2f}) \n Max IoU at third idx: {} (%{:.2f}) (Avg. Score {:.2f}) (Avg. IoU {:.2f})' 
                                         .format(count_1, (count_1/len(indexes))*100, np.mean(masked_scores_1), np.mean(masked_ious_1), count_2, 
                                                (count_2/len(indexes))*100, np.mean(masked_scores_2), np.mean(masked_ious_2), count_3,
                                                (count_3/len(indexes))*100, np.mean(masked_scores_3), np.mean(masked_ious_3)))
    plt.plot(x_values, final_scores, 'rx', label='Confidence Score', ms=2)

    # Add labels and a legend
    plt.xlabel('Frame')
    plt.ylabel('Index of max scores')
    # plt.ylim(0, max(indexes))
    plt.yticks(np.arange(0, max(indexes), 1))
    plt.title('{}'. format(seq))
    plt.legend()

    # Show the plot
    plt.savefig('/root/BHRL/work_dirs/vot/BHRL/first_images_seperate/{}/indexed_scores_{}_target_update.png'. format(seq, epoch))

if __name__ == "__main__":
    main()
