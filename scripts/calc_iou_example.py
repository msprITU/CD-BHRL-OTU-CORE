# Box 1 coordinates
x1_box1, y1_box1, x2_box1, y2_box1 = 952, 245, 1082, 527

# Box 2 coordinates
x1_box2, y1_box2, x2_box2, y2_box2 = 896, 287, 1034, 611

# Calculate areas
area_box1 = (x2_box1 - x1_box1) * (y2_box1 - y1_box1)
area_box2 = (x2_box2 - x1_box2) * (y2_box2 - y1_box2)

# Calculate intersection area
intersection_area = max(0, min(x2_box1, x2_box2) - max(x1_box1, x1_box2)) * max(0, min(y2_box1, y2_box2) - max(y1_box1, y1_box2))

# Calculate IoU
iou = intersection_area / (area_box1 + area_box2 - intersection_area)

print(f"IoU: {iou}")