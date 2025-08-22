import cv2
import json
import os

# Load an image
image = cv2.imread('/root/BHRL/data/VOT/augmented_imgs/ballet_bck/1.jpg')  # Replace 'your_image.jpg' with your image file path
# seq = "car3"
# ann_file = "/root/BHRL/vot_annotation/{}/vot_{}_test.json". format(seq, seq)
# img_folder = "/root/BHRL/data/VOT/{}_imgs". format(seq) 
# json_file = open(ann_file, "r")
# voc_data = json.load(json_file)
# img_idx = 897
# img_idx = voc_data["images"][img_idx]["id"]
# print(img_idx)
# img_path = os.path.join(img_folder, str(img_idx).zfill(8) + ".jpg")
# image = cv2.imread(img_path)

# x1, y1 = 1249, 572  # Top-left corner
# x2, y2 = 1279, 635  # Bottom-right corner

x1,y1,x2,y2 =  [360,325,25,111]
x2 = x1+x2
y2 = y1+y2
color = (0, 255, 0) 
thickness = 2
cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

# cv2.imwrite("/root/BHRL/scripts/deneme_last.png", image)

# image = cv2.imread('/root/BHRL/scripts/deneme_last.png')  # Replace 'your_image.jpg' with your image file path
# x1,y1,x2,y2 =  [737,621,69+737,95+621]
# color = (0, 0, 255)  
# thickness = 2
# cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

cv2.imwrite("/root/BHRL/scripts/deneme_2.png", image)