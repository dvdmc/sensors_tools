from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Literal

from PIL import Image
import numpy as np
import cv2

from sensors_tools.base.cameras import CameraInfo

from .base_bridge import BaseBridge, BaseBridgeConfig

ScanNetSensorDataTypes = Literal["rgb", "depth", "semantic", "pose"]
"""
    List of sensor data to query.
    - "pose": query poses.
    - "rgb": query rgb images.
    - "depth": query depth images.
    - "semantic": query semantic images.
"""

@dataclass
class ScanNetBridgeConfig(BaseBridgeConfig):
    """
        Configuration class for ScanNet
    """
    data_types: List[ScanNetSensorDataTypes] = field(default_factory=list, metadata={"default": ["rgb", "poses"]})
    """ Data types to query """

    dataset_path: Path = Path("/home/david/datasets/ScanNet")
    """ Path to the dataset """

    downsampling_factor_dataset: int = 20
    """ Downsampling factor for the dataset. Or how many images are there in between images. """
    
    width: float = 512
    """ Image width """

    height: float = 512
    """ Image height """


class ScanNetBridge(BaseBridge):
    """
        Bridge for Airsim
    """
    def __init__(self, cfg: ScanNetBridgeConfig):
        """
            Constructor
        """
        super().__init__(cfg)
        self.cfg = cfg

    def setup(self):
        """
            Setup the bridge
        """
        # Data acquisition configuration
        self.static_tf = []
        print("Dataset path: ", self.cfg.dataset_path)
        self.seq_n = 1
        self.each_n_frame = self.cfg.downsampling_factor_dataset
        self.data_length = len(os.listdir(self.cfg.dataset_path / "color"))
        print("Sequence length: ", self.data_length)
        print("Each n frame: ", self.each_n_frame)
        
        # Sim config data TODO: Move to config file
        #######################################################
        # RELEVANT CAMERA DATA

        # Get camera from the textfile "sequence_name.txt" inside the dataset folder
        sequence_data_file = open(self.cfg.dataset_path / "sequence_name.txt", "r")
        sequence_data = sequence_data_file.readlines()
        sequence_data_file.close()

        # Match variables with the textfile
        for line in sequence_data:
            if "colorHeight" in line:
                self.height_color = int(line.split(" ")[2])
            elif "colorWidth" in line:
                self.width_color = int(line.split(" ")[2])
            elif "depthHeight" in line:
                self.height_depth = int(line.split(" ")[2])
            elif "depthWidth" in line:
                self.width_depth = int(line.split(" ")[2])
            elif "fx_color" in line:
                self.fx_color = float(line.split(" ")[2])
            elif "fy_color" in line:
                self.fy_color = float(line.split(" ")[2])
            elif "mx_color" in line:
                self.cx_color = float(line.split(" ")[2])
            elif "my_color" in line:
                self.cy_color = float(line.split(" ")[2])
            elif "fx_depth" in line:
                self.fx_depth = float(line.split(" ")[2])
            elif "fy_depth" in line:
                self.fy_depth = float(line.split(" ")[2])
            elif "mx_depth" in line:
                self.cx_depth = float(line.split(" ")[2])
            elif "my_depth" in line:
                self.cy_depth = float(line.split(" ")[2])
        self.camera_info = CameraInfo(cx=self.cx_color, cy=self.cy_color, fx=self.fx_color, fy=self.fy_color, width=self.cfg.width, height=self.cfg.height)
        self.camera_info_depth = CameraInfo(cx=self.cx_depth, cy=self.cy_depth, fx=self.fx_depth, fy=self.fy_depth, width=self.cfg.width, height=self.cfg.height)
        #######################################################

    def remap_NYU_classes(self, label_40):
        label_13 = np.zeros_like(label_40)
        for key, value in self.remapping_40_to_13_classes.items():
            label_13[np.where(label_40 == key)] = value
        return label_13

    def remap_ScanNet_to_13_classes(self, ScanNet_label):
        label_40 = np.zeros_like(ScanNet_label)
        for key, value in self.remap_to_NYU_classes.items():
            label_40[np.where(ScanNet_label == key)] = value
        label_13 = self.remap_NYU_classes(label_40)
        return label_13
    
    def open_images(self):
        """
        Open the images from the dataset folder
        Stores the data in a dictionary
        """
        data = {}

        if "rgb" in self.cfg.data_types:
            # Load RGB image
            img_path = self.cfg.dataset_path / "color" / f"{self.seq_n}.jpg"
            # Open image as a np array
            img = (Image.open(img_path)).convert('RGB')
            img = img.resize((640, 480)) #Resize to match the depth image
            img = img.crop((80, 0, 560, 480)) #Crop the image to match the depth image
            data["rgb"] = np.array(img)
        
        if "semantic" in self.cfg.data_types:
            # Load GT label
            label_path = self.cfg.dataset_path / "label" / f"{self.seq_n}.png"
            label = np.array(Image.open(label_path))
            label = self.remap_ScanNet_to_13_classes(label)
            label[np.where(label == 255)] = 0 #Remove the white contour
            label = cv2.resize(label, (640, 480), interpolation = cv2.INTER_NEAREST)
            label = label[:, 80:560]
            data["semantic_gt"] = label

        if "depth" in self.cfg.data_types:
            # Load depth image (depth frames as 16-bit pngs (depth shift 1000))
            depth_path = self.cfg.dataset_path / "depth" / f"{self.seq_n}.png"
            depth = np.array(Image.open(depth_path))
            depth = (depth/1000).astype(np.float32)
            depth = depth[:, 80:560]
            data["depth"] = depth

        return data

    
    def get_data(self):
        """
            Get data from the bridge
        """
        data = {}
        if "pose" in self.cfg.data_types:
            pose_path = self.cfg.dataset_path / "pose" / f"{self.seq_n}.txt"
            data["pose"] = np.loadtxt(pose_path)

        img_data = self.open_images()

        data.update(img_data)

        return data
