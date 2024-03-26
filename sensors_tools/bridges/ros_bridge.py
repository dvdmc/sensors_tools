from dataclasses import dataclass, field
from typing import List, Literal

import cv2

import rospy
from sensors_tools.base.cameras import CameraData
import tf2_ros
import cv_bridge
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber

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

    camera_info_topic: str = "/camera/rgb/camera_info"
    """ Camera info topic """

    depth_topic: str = "/camera/depth/image_raw"
    """ Depth topic """

    origin_tf: str = "map"
    """ Origin frame to query """

    poses_tf: str = "camera/base_link"
    """ Poses frame to query """


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
        self.ready = False

    def setup(self):
        """
        Setup the bridge
        """
        # Members
        self.rgb = None
        self.depth = None
        self.pose = None
        self.semantic_gt = None
        self.has_camera_info = False
        self.has_depth_camera_info = False
        # Init ros subscribers

        # Sync rgb and depth if they are both present
        if "rgb" in self.cfg.data_types and "depth" in self.cfg.data_types:
            self.bridge = cv_bridge.CvBridge()
            self.rgb_sub = Subscriber(self.cfg.rgb_topic, RosImage)
            self.depth_sub = Subscriber(self.cfg.depth_topic, RosImage)
            self.sync = ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub], 10, 0.1)
            self.sync.registerCallback(self.sync_callback)
            self.camera_info_sub = rospy.Subscriber(self.cfg.camera_info_topic, CameraInfo, self.camera_info_callback)
            self.depth_camera_info_sub = rospy.Subscriber(self.cfg.camera_info_topic, CameraInfo, self.depth_camera_info_callback)
        else:
            if "rgb" in self.cfg.data_types:
                self.rgb_sub = rospy.Subscriber(self.cfg.rgb_topic, RosImage, self.rgb_callback)
                self.camera_info_sub = rospy.Subscriber(self.cfg.camera_info_topic, CameraInfo, self.camera_info_callback)
            if "depth" in self.cfg.data_types:
                self.depth_sub = rospy.Subscriber(self.cfg.depth_topic, RosImage, self.depth_callback)
                self.depth_camera_info_sub = rospy.Subscriber(self.cfg.camera_info_topic, CameraInfo, self.depth_camera_info_callback)

        # TF listener
        if "pose" in self.cfg.data_types:
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(buffer=self.tf_buffer)

        # RELEVANT CAMERA DATA
        if "rgb" in self.cfg.data_types:
            # Wait for the camera_info topic to be published
            while not self.has_camera_info:
                rospy.loginfo("Waiting for camera_info topic to be published")
                rospy.sleep(1)

        if "depth" in self.cfg.data_types:
            # Wait for the depth camera_info topic to be published
            while not self.has_depth_camera_info:
                rospy.loginfo("Waiting for depth camera_info topic to be published")
                rospy.sleep(1)

        self.ready = True

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

    def depth_camera_info_callback(self, data: CameraInfo):
        """
        Callback for the depth camera info
        """
        self.depth_width = data.width
        self.depth_height = data.height
        self.depth_fx = data.K[0]
        self.depth_fy = data.K[4]
        self.depth_cx = data.K[2]
        self.depth_cy = data.K[5]
        self.depth_camera_info = CameraData(
            cx=self.depth_cx, cy=self.depth_cy, fx=self.depth_fx, fy=self.depth_fy, width=self.depth_width, height=self.depth_height
        )
        self.has_depth_camera_info = True

    def sync_callback(self, rgb_data: RosImage, depth_data: RosImage):
        """
        Callback for the sync rgb and depth
        """
        self.rgb_callback(rgb_data)
        self.depth_callback(depth_data)

    def rgb_callback(self, data: RosImage):
        """
        Callback for the rgb image
        """
        self.rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")

        # Move to a different callback
        if "pose" in self.cfg.data_types:
            # Update pose using the tf listener
            geometry_msg_pose = self.tf_buffer.lookup_transform(self.cfg.origin_tf, self.cfg.poses_tf, rospy.Time(0), timeout=rospy.Duration(1))
            self.pose = (
                np.array(
                    [
                        geometry_msg_pose.transform.translation.x,
                        geometry_msg_pose.transform.translation.y,
                        geometry_msg_pose.transform.translation.z,
                    ]
                ),
                Rotation.from_quat(
                    [
                        geometry_msg_pose.transform.rotation.x,
                        geometry_msg_pose.transform.rotation.y,
                        geometry_msg_pose.transform.rotation.z,
                        geometry_msg_pose.transform.rotation.w,
                    ]
                ),
            )

    def depth_callback(self, data: RosImage):
        """
        Callback for the depth image
        """
        self.depth = self.bridge.imgmsg_to_cv2(data, "passthrough") / 1000.0 # Convert to meters

    def get_data(self):
        """
        Get data from the bridge
        """
        data = {}
        if "rgb" in self.cfg.data_types:
            if self.rgb is not None:
                data["rgb"] = self.rgb
            else:
                print("RGB data not available")

        if "depth" in self.cfg.data_types:
            if self.depth is not None:
                data["depth"] = self.depth
            else:
                print("Depth data not available")
        
        # If rgb and depth are requested, resize the depth to match the rgb
        if "rgb" in self.cfg.data_types and "depth" in self.cfg.data_types:
            data["depth"] = cv2.resize(data["depth"], (self.width, self.height))

        if "pose" in self.cfg.data_types:
            if self.pose is not None:
                data["pose"] = self.pose
            else:
                print("Pose data not available")

        if "semantic" in self.cfg.data_types:
            if self.semantic_gt is not None:
                data["semantic_gt"] = self.semantic_gt
            else:
                # Fill in fake data
                data["semantic_gt"] = np.zeros((self.height, self.width))

        # If any of the data is not available, return None
        if any([v is None for v in data.values()]):
            return None
        
        return data

    def get_pose(self):
        """
        Get the pose from the bridge
        """
        return self.pose