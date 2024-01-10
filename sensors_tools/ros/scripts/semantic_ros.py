import numpy as np

import rospy
from sensor_msgs.msg import Image as ImageMsg
from std_srvs.srv import SetBoolResponse, SetBool
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

from sensors_tools.sensor import SemanticInferenceSensor, SensorConfig
class SemanticNode:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self.setup()

    def setup(self):

        # Get params: frequency and inference type
        self.frequency = rospy.get_param('~frequency', 10)
        self.inference_type = rospy.get_param("~inference_type", "deterministic")

        self.offset_init = rospy.get_param('~offset_init', [0,0,0])
        self.cfg.inference_type = self.inference_type
        self.sensor = SemanticInferenceSensor(self.cfg)

        self.static_tf = []
        self.last_tf_msg = None
        self.last_odom_msg = None

        #######################################################
        # RELEVANT CAMERA DATA
        self.camera_info = self.sensor.bridge.camera_info
        #######################################################
        # ROS CAMERA DATA
        self.camera_name = "0"
        self.camera_odom_frame_id = "cam" + self.camera_name + "_odom"

        #TODO: Offset init should be provided by the bridge
        static_start_transform = TransformStamped()
        static_start_transform.header.stamp = rospy.Time.now()
        static_start_transform.header.frame_id = "map"
        static_start_transform.child_frame_id = self.camera_odom_frame_id
        static_start_transform.transform.translation.x = self.offset_init[0]
        static_start_transform.transform.translation.y = self.offset_init[1]
        static_start_transform.transform.translation.z = self.offset_init[2]
        static_start_transform.transform.rotation.x = 0
        static_start_transform.transform.rotation.y = 0
        static_start_transform.transform.rotation.z = 0
        static_start_transform.transform.rotation.w = 1
        self.static_tf.append(static_start_transform)
        #######################################################

        # ROS setup
        self.pub_camera_odometry = rospy.Publisher("/camera_bay/odom", Odometry, queue_size=10)
        self.pub_rgb = rospy.Publisher('/camera_bay/rgb/image_raw', ImageMsg, queue_size=10)
        self.pub_out = rospy.Publisher('/camera_bay/rgb/image_raw_out', ImageMsg, queue_size=10)
 
        self.pub_aleatoric_uncertainty = rospy.Publisher('/camera_bay/aleatoric_uncertainty', ImageMsg, queue_size=10)
        self.pub_epistemic_uncertainty = rospy.Publisher('/camera_bay/epistemic_uncertainty', ImageMsg, queue_size=10)
        self.pub_total_uncertainty = rospy.Publisher('/camera_bay/total_uncertainty', ImageMsg, queue_size=10)
        self.pub_point_cloud = rospy.Publisher('/camera_bay/point_cloud', PointCloud2, queue_size=10)

        if self.frequency == -1:
            self.srv_capture_data = rospy.Service('/sensor/capture_bay', SetBool, self.loop_srv)
        else:
            self.timer = rospy.Timer(rospy.Duration(1.0/self.frequency), self.loop)

        def to_pose_msg(self, pose: np.ndarray, timestamp: rospy.Time) -> Odometry:
            """
                Convert a 4x4 pose matrix to a ROS Odometry message
            """
            odom_msg = Odometry()
            odom_msg.header.frame_id = self.camera_odom_frame_id
            odom_msg.header.stamp = timestamp
            odom_msg.child_frame_id = "base_link"
            odom_msg.pose.pose.position.x = pose[0,3]
            odom_msg.pose.pose.position.y = pose[1,3]
            odom_msg.pose.pose.position.z = pose[2,3]
            odom_msg.pose.pose.orientation.x = pose[0,0]
            odom_msg.pose.pose.orientation.y = pose[1,0]
            odom_msg.pose.pose.orientation.z = pose[2,0]
            odom_msg.pose.pose.orientation.w = pose[3,0]
            return odom_msg