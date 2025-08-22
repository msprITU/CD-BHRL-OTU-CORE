import cv2
import glob

img_path = '/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7/result_imgs/person19/*.jpg'
out_path = '/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7/result_imgs/person19/person19.avi'

img_array = []
img = None
for idx, filename in enumerate(sorted(glob.glob(img_path))):
    img = cv2.imread(filename)
    
    # img = cv2.putText(img, 
    #     str(500 + idx), 
    #     (50, 50), 
    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
    #     fontScale=1, 
    #     color=(255, 255, 255), 
    #     thickness=3, 
    #     lineType=cv2.LINE_AA, 
    #     bottomLeftOrigin=False)
                
    img = cv2.resize(img, (640,384), interpolation=cv2.INTER_AREA)
    img_array.append(img)

size = (img.shape[1], img.shape[0])

out = cv2.VideoWriter(out_path, 0, 10, size)

for img in img_array:
    out.write(img)

out.release()