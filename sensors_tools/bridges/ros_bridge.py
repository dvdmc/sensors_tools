from dataclasses import dataclass, field
from typing import List, Literal

import rospy
from sensors_tools.base.cameras import CameraData
import tf2_ros
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo
from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation

from sensors_tools.bridges.base_bridge import BaseBridge, BaseBridgeConfig

ROSSensorDataTypes = Literal["rgb", "pose"]
"""
    List of sensor data to query.
    - "pose": query poses.
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
        if "rgb" in self.cfg.data_types:
            self.rgb_sub = rospy.Subscriber(self.cfg.rgb_topic, Image, self.rgb_callback)
            self.camera_info_sub = rospy.Subscriber("/camera/rgb/camera_info", CameraInfo, self.camera_info_callback)
        # TF listener
        if "pose" in self.cfg.data_types:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer)

        # RELEVANT CAMERA DATA
        # Wait for the camera_info topic to be published
        self.has_camera_info = False
        while not self.has_camera_info:
            rospy.loginfo("Waiting for camera_info topic to be published")
            rospy.sleep(1)
        

    def camera_info_callback(self, data: CameraInfo):
        """
        Callback for the camera info
        """
        self.width = data.width
        self.height = data.height
        self.cx = data.K[2]
        self.cy = data.K[5]
        self.fx = data.K[0]
        self.fy = data.K[4]

        self.camera_info = CameraData(
            cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy, width=self.width, height=self.height
        )
        self.has_camera_info = True

    def rgb_callback(self, data: Image):
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
