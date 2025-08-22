
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
    predictions_filename = '/root/BHRL/target_update_studies/real_time/{}/results/results_{}.txt'. format(epoch,seq)
    ground_truth_filename = '/root/BHRL/data/VOT/{}/groundtruth.txt'. format(seq)

    with open(predictions_filename, 'r') as pred_file, open(ground_truth_filename, 'r') as gt_file:
        predictions_lines = pred_file.readlines()
        ground_truth_lines = gt_file.readlines()
        print(len(predictions_lines))
        print(len(ground_truth_lines))
        assert len(predictions_lines) == len(ground_truth_lines), "Number of predictions and ground truths do not match"

        total_iou = 0.0
        num_samples = len(predictions_lines)

        for i in range(num_samples):
            pred_line = predictions_lines[i].strip().split(',')
            gt_line = ground_truth_lines[i].strip().split(',')

            pred_box = list(map(float, pred_line[1:5]))
            gt_box = list(map(float, gt_line))
            score = float(pred_line[5])

            iou = calculate_iou(pred_box, gt_box)
            total_iou += iou

            print(f"IoU for sample {i+1}: {iou:.4f}")
            ious.append(iou)
            scores.append(score)

        avg_iou = total_iou / num_samples

        print(f"Average IoU: {avg_iou:.4f}")

    x_values = list(range(1, len(ious)+1))


    print(scores[0])

    # Plot the data from both lists on the same graph
    plt.plot(x_values, ious, label='IoU/ Avg. : {}'.format(np.mean(ious)))
    plt.plot(x_values, scores, 'rx', label='Score / Avg. : {}'.format(np.mean(scores)), ms=2)

    # Add labels and a legend
    plt.xlabel('Frame')
    plt.ylabel('Scores / IoU')
    plt.title('{} - Scores vs IoU'. format(seq))
    plt.legend()

    # Show the plot
    # plt.show()
    plt.savefig('/root/BHRL/work_dirs/vot/BHRL/first_images_seperate/{}/scores_iou_{}_target_update.png'. format(seq, epoch))

if __name__ == "__main__":
    main()
