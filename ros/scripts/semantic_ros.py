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
        self.inference_type = rospy.get_param('~inference_type', 'deterministic')
        self.cfg.inference_type = self.inference_type
        self.sensor = SemanticInferenceSensor(self.cfg)

        self.static_tf = []
        self.last_tf_msg = None
        self.last_odom_msg = None

        #######################################################
        # RELEVANT CAMERA DATA TODO: Get from bridge
        self.width = 512
        self.height = 512
        self.fov_h = 54.4
        self.cx = float(self.width)/2
        self.cy = float(self.height)/2
        fov_h_rad = self.fov_h * np.pi / 180.0
        self.fx = self.cx / (np.tan(fov_h_rad/2))
        self.fy = self.fx * self.height / self.width
        self.client.simSetFocusAperture(7.0,"0") # Avoids depth of field blur
        self.client.simSetFocusDistance(100.0,"0") # Avoids depth of field blur
        #######################################################
        # ROS CAMERA DATA
        self.camera_name = "0"
        self.camera_odometry = self.pose_airsim_to_ros(self.stateClient.simGetCameraInfo("0").pose)
        self.camera_odom_frame_id = "cam" + self.camera_name + "_odom"
        self.player_start = [0,0,0] # In Unreal is [28.63, -133.27, 156.77] but we have cm instead of m and x -y z and an offset of +height/2 in z [0.386, 1.5, 1.0677]
        static_start_transform = TransformStamped()
        static_start_transform.header.stamp = rospy.Time.now()
        static_start_transform.header.frame_id = "map"
        static_start_transform.child_frame_id = self.camera_odom_frame_id
        static_start_transform.transform.translation.x = self.player_start[0]
        static_start_transform.transform.translation.y = self.player_start[1]
        static_start_transform.transform.translation.z = self.player_start[2]
        static_start_transform.transform.rotation.x = 0
        static_start_transform.transform.rotation.y = 0
        static_start_transform.transform.rotation.z = 0
        static_start_transform.transform.rotation.w = 1
        self.static_tf.append(static_start_transform)
        #######################################################

        # ROS setup
        self.pub_camera_odometry = rospy.Publisher("/camera_bay/odom", Odometry, queue_size=10)
        self.pub = rospy.Publisher('/camera_bay/rgb/image_raw', ImageMsg, queue_size=10)

        self.pub_out = rospy.Publisher('/camera_bay/rgb/image_raw_out', ImageMsg, queue_size=10) 
        self.pub_aleatoric_uncertainty = rospy.Publisher('/camera_bay/aleatoric_uncertainty', ImageMsg, queue_size=10)
        self.pub_epistemic_uncertainty = rospy.Publisher('/camera_bay/epistemic_uncertainty', ImageMsg, queue_size=10)
        self.pub_total_uncertainty = rospy.Publisher('/camera_bay/total_uncertainty', ImageMsg, queue_size=10)
        self.pub_point_cloud = rospy.Publisher('/camera_bay/point_cloud', PointCloud2, queue_size=10)

        if self.frequency == -1:
            self.srv_capture_data = rospy.Service('/sensor/capture_bay', SetBool, self.loop_srv)
        else:
            self.timer = rospy.Timer(rospy.Duration(1.0/self.frequency), self.loop)
