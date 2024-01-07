###
#
# You can use this file in combination with the test_data and generator.py to test certain loading functionalities.
#
###
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import List, Literal

from PIL import Image
import numpy as np

from base_bridge import BaseBridge, BridgeConfig

TestSensorDataTypes = Literal["rgb", "depth", "semantic", "poses"]
"""
    List of sensor data to query.
    - "poses": query poses.
    - "rgb": query rgb images.
    - "depth": query depth images.
    - "semantic": query semantic images.
"""

@dataclass
class TestBridgeConfig(BridgeConfig):
    """
        Configuration class for Test
    """
    data_types: List[TestSensorDataTypes] = field(default_factory=list, metadata={"default": ["rgb", "poses"]})
    """ Data types to query """

    dataset_path: Path = Path("./test_data/")
    """ Path to the dataset """
    
    width: float = 512
    """ Image width """

    height: float = 512
    """ Image height """


class TestBridge(BaseBridge):
    """
        Bridge for Airsim
    """
    def __init__(self, cfg: TestBridgeConfig):
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
        self.data_length = len(os.listdir(self.cfg.dataset_path / "color"))
        print("Sequence length: ", self.data_length)
        
        # Sim config data TODO: Move to config file
        #######################################################
        # RELEVANT CAMERA DATA

        # Get camera from the json "camera_intrinsics.json" inside the dataset folder
        with open(self.cfg.dataset_path / "camera_intrinsics.json") as f:
            camera_data = json.load(f)
        
        print("Camera data: ", camera_data)
        #######################################################
    
    def open_images(self):
        """
        Open the images from the dataset folder
        Stores the data in a dictionary
        """
        data = {}

        if "rgb" in self.cfg.data_types:
            # Load RGB image
            img_path = self.cfg.dataset_path / "color" / f"{self.seq_n:04d}.png"
            # Open image as a np array
            img = (Image.open(img_path)).convert('RGB')
            data["rgb"] = np.array(img)
        
        if "semantic" in self.cfg.data_types:
            # Load GT label
            label_path = self.cfg.dataset_path / "label" / f"{self.seq_n:04d}.png"
            label = np.array(Image.open(label_path))
            data["semantic_gt"] = label

        if "depth" in self.cfg.data_types:
            # Load depth image (depth frames as 16-bit pngs (depth shift 1000))
            depth_path = self.cfg.dataset_path / "depth" / f"{self.seq_n:04d}.png"
            depth = np.array(Image.open(depth_path))
            depth = (depth/1000).astype(np.float32)
            data["depth"] = depth

        return data

    
    def get_data(self):
        """
            Get data from the bridge
        """
        data = {}
        if "poses" in self.cfg.data_types:
            pose_path = self.cfg.dataset_path / "pose" / f"{self.seq_n:04d}.txt"
            data["pose"] = np.loadtxt(pose_path)

        img_data = self.open_images()

        data.update(img_data)

        return data
    
    def increment_seq(self):
        """
            Increment the sequence number
        """
        self.seq_n += 1
        if self.seq_n > self.data_length:
            self.seq_n = 1
        print("Sequence number: ", self.seq_n)