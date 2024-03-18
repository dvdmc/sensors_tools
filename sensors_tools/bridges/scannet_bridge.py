from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Literal

from PIL import Image
import numpy as np
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation

from sensors_tools.base.cameras import CameraData

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

    dataset_path: Path = Path("/media/david/dataset/ScanNet")
    """ Path to the dataset """

    tsv_path: Path = Path("/media/david/dataset/ScanNet/scannetv2-labels.combined.tsv")
    """ Path to the tsv file """

    downsampling_factor_dataset: int = 20
    """ Downsampling factor for the dataset. Or how many images are there in between images. """


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
        self.seq_n = 0
        self.each_n_frame = self.cfg.downsampling_factor_dataset
        self.data_length = len(os.listdir(self.cfg.dataset_path / "color"))
        print("Sequence length: ", self.data_length)
        print("Each n frame: ", self.each_n_frame)
        
        # Sim config data TODO: Move to config file
        #######################################################
        # RELEVANT CAMERA DATA

        # Get camera from the textfile "sequence_name.txt" inside the dataset folder
        # Sequence name is the name of the folder where the data is stored
        self.sequence_name = f"{self.cfg.dataset_path.name}.txt"
        sequence_data_file = open(self.cfg.dataset_path / self.sequence_name, "r")
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
        # We adjust to the depth image size
        ratio_width = self.width_depth/self.width_color
        ratio_height = self.height_depth/self.height_color
        self.width_color = self.width_depth
        self.height_color = self.height_depth
        # Adapt cx, cy, fx and fy
        self.cx_color = self.cx_color*ratio_width
        self.cy_color = self.cy_color*ratio_height
        self.fx_color = self.fx_color*ratio_width
        self.fy_color = self.fy_color*ratio_height
        self.camera_info = CameraData(cx=self.cx_color, cy=self.cy_color, fx=self.fx_color, fy=self.fy_color, width=self.width_color, height=self.height_color) 
        print("CAMERA INFO: ", self.camera_info)
        self.camera_info_depth = CameraData(cx=self.cx_depth, cy=self.cy_depth, fx=self.fx_depth, fy=self.fy_depth, width=self.width_depth, height=self.height_depth)
        #######################################################
        label_mapping = pd.read_csv(self.cfg.tsv_path, sep='\t')
        NYU_classes = label_mapping['nyu40id'].values
        id_classes = label_mapping['id'].values
        self.remap_to_NYU_classes = {id_classes[i]: NYU_classes[i].astype(int) for i in range(len(NYU_classes))}
        self.remapping_40_to_13_classes = {0: 0, 1:12, 2:5, 3:6, 4:1, 5:4, 6:9, 7:10, 8:12, 9:13, 10:6, 11:8, 12:6, 
                                    13:13, 14:10, 15:6, 16:13, 17:6, 18:7, 19:7, 20:5, 21:7, 22:3, 23:2, 24:6,
                                    25:11, 26:7, 27:7, 28:7, 29:7, 30:7, 31:7, 32:6, 33:7, 34:7, 35:7, 36:7,
                                    37:7, 38:7, 39:6, 40:7}
        


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
            img = img.resize((self.camera_info_depth.width, self.camera_info_depth.height)) #Resize to match the depth image
            # img = img.crop((80, 0, 560, 480)) #Crop the image to match the depth image
            data["rgb"] = np.array(img)
        
        if "semantic" in self.cfg.data_types:
            # Load GT label
            label_path = self.cfg.dataset_path / "label" / f"{self.seq_n}.png"
            label = np.array(Image.open(label_path))
            label = self.remap_ScanNet_to_13_classes(label)
            label[np.where(label == 255)] = 0 #Remove the white contour
            label = cv2.resize(label, (self.camera_info_depth.width, self.camera_info_depth.height), interpolation = cv2.INTER_NEAREST)
            # label = label[:, 80:560]
            data["semantic_gt"] = label

        if "depth" in self.cfg.data_types:
            # Load depth image (depth frames as 16-bit pngs (depth shift 1000))
            depth_path = self.cfg.dataset_path / "depth" / f"{self.seq_n}.png"
            depth = np.array(Image.open(depth_path))
            depth = (depth/1000).astype(np.float32)
            # depth = depth[:, 80:560]
            data["depth"] = depth

        return data

    
    def get_data(self):
        """
            Get data from the bridge
        """
        data = {}
        if "pose" in self.cfg.data_types:
            pose_path = self.cfg.dataset_path / "pose" / f"{self.seq_n}.txt"
            translation_matrix = np.loadtxt(pose_path)
            translation = translation_matrix[:3, 3]
            rotation = Rotation.from_matrix(translation_matrix[:3, :3])
            data["pose"] = (translation, rotation)

        img_data = self.open_images()

        data.update(img_data)

        return data

    def move(self):
        """
            Apply increment seq as moving the sensor
        """
        self.seq_n += self.each_n_frame