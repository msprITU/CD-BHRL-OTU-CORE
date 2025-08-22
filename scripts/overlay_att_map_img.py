from PIL import Image
import os

# Define the paths to your directories
input_images_dir = '/root/BHRL/oscda/liverRun_refs/'
attention_maps_dir = '/root/BHRL/oscda/att_maps/liverRun'
output_dir = '/root/BHRL/oscda/overlay/liverRun'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over the files in the input images directory
for filename in os.listdir(input_images_dir):
    if not filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
        continue

    # Construct the full file paths
    input_image_path = os.path.join(input_images_dir, filename)
    attention_map_path = os.path.join(attention_maps_dir, filename)

    # Check if the corresponding attention map exists
    if not os.path.exists(attention_map_path):
        print(f"Attention map for {filename} not found.")
        continue

    # Open the input image and attention map
    input_image = Image.open(input_image_path).convert("RGBA")
    attention_map = Image.open(attention_map_path).convert("RGBA")

    # Resize the attention map to match the input image size, if necessary
    if input_image.size != attention_map.size:
        attention_map = attention_map.resize(input_image.size, Image.BILINEAR)

    # Blend the input image with the attention map
    blended_image = Image.blend(input_image, attention_map, alpha=0.85)  # Adjust alpha to taste

    # Save the blended image
    blended_image_path = os.path.join(output_dir, filename)
    blended_image.save(blended_image_path)

    print(f"Overlay image saved for {filename}")
