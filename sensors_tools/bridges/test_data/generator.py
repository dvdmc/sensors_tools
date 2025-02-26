"""
    This script is used to generate test data for the bridges.
    It consists in  a ser of:
     - RGB images of size 512x512 with a number in the center in black over white background.
     - Depth images of size 512x512 with the white color at 10m and the number at 3m.
     - Semantic images of size 512x512 with the number in the center being class 1 and the rest class 0.
     - Pose files with a translation of the number along the X axis.
     - A file with the camera intrinsics.
"""
import os
import cv2
import numpy as np
import json

# Import debuggers
import pdb

NUM_IMAGES = 5
WIDTH = 512
HEIGHT = 512

def generate_data(num: int) -> tuple:
    """
        Generate the image, pose, depth and semantic images
        The image is generated with a number in the center using the cv2 library
        The depth image is generated by using the rgb as a mask and setting the values
        The semantic image is generated by using the rgb as a mask and setting the values
    """

    # Generate the rgb image
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    img.fill(255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    textsize = cv2.getTextSize(str(num), font, 5, 2)[0]
    textX = (WIDTH - textsize[0]) // 2
    textY = (HEIGHT + textsize[1]) // 2
    cv2.putText(img, str(num), (textX, textY), font, 5, (0, 0, 0), 10, cv2.LINE_AA)

    # Generate the depth image
    depth = np.zeros((HEIGHT, WIDTH), np.float32)
    depth.fill(10*1000)
    mask = img[:,:,0] == 0
    depth[mask] = 3*1000

    # Generate the semantic image
    semantic = np.zeros((HEIGHT, WIDTH), np.uint16)
    semantic[mask] = 1

    # Generate the semantic image in rgb with 0 in red and 1 in blue
    semantic_rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    semantic_rgb[semantic == 0] = [255, 0, 0]
    semantic_rgb[semantic == 1] = [0, 0, 255]

    # Generate the pose file
    pose = np.eye(4)
    pose[0,3] = num

    return img, depth, semantic, semantic_rgb, pose

def generate_sheeps_img(num: int) -> tuple:
    """
    Generate the images for the "sheeps.png" image and the "sheeps_sem.png" semantic image
    Instead of the number, use the sheeps and use the sheeps_sem.png with red RGB color for the gt
    mask.
    """
    # Generate the RGB image
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    img.fill(255)

    # Load the sheeps image
    sheeps = cv2.imread("sheeps.png")
    
    # Calculate the scale to fill the frame while maintaining aspect ratio
    scale_w = WIDTH / sheeps.shape[1]
    scale_h = HEIGHT / sheeps.shape[0]
    scale = max(scale_w, scale_h)
    sheeps = cv2.resize(sheeps, (0, 0), fx=scale, fy=scale)

    # Crop the sheep image to fit exactly in the frame
    start_x = (sheeps.shape[1] - WIDTH) // 2
    start_y = (sheeps.shape[0] - HEIGHT) // 2
    sheeps_cropped = sheeps[start_y:start_y + HEIGHT, start_x:start_x + WIDTH]
    img = sheeps_cropped

    # Generate the depth image
    depth = np.zeros((HEIGHT, WIDTH), np.float32)
    depth.fill(10 * 1000)
    mask = img[:, :, 0] != 255
    depth[mask] = 3 * 1000

    # Generate the semantic image
    # Load the semantic image
    semantic_img = cv2.imread("sheeps_sem.png")
    
    # Resize and crop the semantic image to match the sheep's appearance
    semantic_img = cv2.resize(semantic_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    semantic_cropped = semantic_img[start_y:start_y + HEIGHT, start_x:start_x + WIDTH]

    semantic = np.zeros((HEIGHT, WIDTH), np.uint16)

    # Create semantic_rgb
    semantic_rgb = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    semantic_rgb = semantic_cropped
    # Get the mask
    mask_sheep = semantic_rgb[:, :, 2] == 47
    mask_human = semantic_rgb[:, :, 2] == 53
    semantic[mask_sheep] = 17
    semantic[mask_human] = 15

    # Generate the pose file
    pose = np.eye(4)
    pose[0, 3] = num

    return img, depth, semantic, semantic_rgb, pose

if __name__ == "__main__":
    # Create the directories
    os.makedirs("dataset/color", exist_ok=True)
    os.makedirs("dataset/depth", exist_ok=True)
    os.makedirs("dataset/label", exist_ok=True)
    os.makedirs("dataset/label_rgb", exist_ok=True)
    os.makedirs("dataset/pose", exist_ok=True)

    # Generate the data
    for i in range(NUM_IMAGES):
        if i == 0:
            img, depth, semantic, semantic_rgb, pose = generate_sheeps_img(i)
        else:
            img, depth, semantic, semantic_rgb, pose = generate_data(i)

        cv2.imwrite(f"dataset/color/{i:04d}.png", img)
        # Save depth as 16-bit png (depth shift 1000)
        cv2.imwrite(f"dataset/depth/{i:04d}.png", depth.astype(np.uint16))
        cv2.imwrite(f"dataset/label/{i:04d}.png", semantic)
        # Save the images with no compression in label
        cv2.imwrite(f"dataset/label_rgb/{i:04d}.png", semantic_rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # Save the pose
        np.savetxt(f"dataset/pose/{i:04d}.txt", pose)

    # Save the camera intrinsics
    camera_intrinsics = {
        "fx": 1000,
        "fy": 1000,
        "cx": 256,
        "cy": 256
    }
    with open("dataset/camera_intrinsics.json", "w") as f:
        json.dump(camera_intrinsics, f)

