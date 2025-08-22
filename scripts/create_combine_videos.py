import cv2
import os
from moviepy.editor import VideoFileClip, clips_array


# Function to create video from images in a folder
def create_video_from_images(folder_path, output_video_name, frame_rate):
    images = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]
    images.sort()  # Sort the images if needed

    # Determine the width and height from the first image
    frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = frame.shape

    # Change here for AVI format
    video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (width, height))

    for idx, image in enumerate(images):
        if idx > 1698:
            continue
        video.write(cv2.imread(os.path.join(folder_path, image)))

    cv2.destroyAllWindows()
    video.release()

# Paths to your folders

folder1 = '/root/BHRL/vot_results/real_bhrl_voc_eval/visual_results/following'
folder2 = '/root/BHRL/vot_results/target_update_studies/real_time/e100_IoU_0_7_parts/parts_100/result_imgs/following'

# Create videos from each folder (in AVI format)
create_video_from_images(folder1, '/root/BHRL/vot_results/result_comparison/following_base.avi', frame_rate=10)
create_video_from_images(folder2, '/root/BHRL/vot_results/result_comparison/following_oscda.avi', frame_rate=10)

# Combine videos side by side
clip1 = VideoFileClip("/root/BHRL/vot_results/result_comparison/following_base.avi")
clip2 = VideoFileClip("/root/BHRL/vot_results/result_comparison/following_oscda.avi")

final_clip = clips_array([[clip1, clip2]])
# Output also in AVI format
final_clip.write_videofile("/root/BHRL/vot_results/result_comparison/following_base_vs_oscda.avi", codec='mpeg4', fps=10)

