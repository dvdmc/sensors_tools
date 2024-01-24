from dataclasses import dataclass, field
from typing import List, Literal

import rospy
from sensors_tools.base.cameras import CameraInfo
import tf2_ros
import cv_bridge
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

from sensors_tools.bridges.base_bridge import BaseBridge, BaseBridgeConfig

ROSSensorDataTypes = Literal["rgb", "pose"]
"""
    List of sensor data to query.
    - "poses": query poses.
    - "rgb": query rgb images.
    - "depth": query depth images.
    - "semantic": query semantic images.
"""

@dataclass
class ROSBridgeConfig(BaseBridgeConfig):
    """
        Configuration class for AirsimBridge
    """
    data_types: List[ROSSensorDataTypes] = field(default_factory=list, metadata={"default": ["rgb", "pose"]})
    """ Data types to query """

    rgb_topic: str = "/camera/rgb/image_raw"
    """ RGB topic """

    origin_tf: str = "map"
    """ Origin frame to query """

    poses_tf: str = "/camera/base_link"
    """ Poses frame to query """

    width: int = 512
    """ Image width """

    height: int = 512
    """ Image height """

    fov_h: float = 54.4
    """ Horizontal field of view """

class ROSBridge(BaseBridge):
    """
        Bridge for ROS
    """
    def __init__(self, cfg: ROSBridgeConfig):
        """
            Constructor
        """
        super().__init__(cfg)
        self.cfg = cfg

    def setup(self):
        """
            Setup the bridge
        """
        # Members
        self.rgb = None
        self.pose = None

        # Init ros subscribers
        self.rgb_subscriber = rospy.Subscriber(self.cfg.rgb_topic, Image, self.rgb_callback)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer)

        # Sim config data TODO: Obtain from camera_data topic in the future
        #######################################################
        # RELEVANT CAMERA DATA
        self.width = self.cfg.width
        self.height = self.cfg.height
        self.fov_h = self.cfg.fov_h
        self.cx = float(self.width) / 2
        self.cy = float(self.height) / 2
        fov_h_rad = self.fov_h * np.pi / 180.0
        self.fx = self.cx / (np.tan(fov_h_rad / 2))
        self.fy = self.fx * self.height / self.width
        self.camera_info = CameraInfo(cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy, width=self.cfg.width, height=self.cfg.height)
        #######################################################

    def rgb_callback(self, data):
        """
            Callback for the rgb image
        """
        self.rgb = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")

        # Update pose using the tf listener
        self.pose = self.tf_buffer.lookup_transform(self.cfg.origin_tf, self.cfg.poses_tf, rospy.Time(0))

    def get_data(self):
        """
            Get data from the bridge
        """
        data = {}
        if "rgb" in self.cfg.data_types and self.rgb is not None:
            data["rgb"] = self.rgb

        if "pose" in self.cfg.data_types and self.pose is not None:
            data["pose"] = self.pose
            
        return data
