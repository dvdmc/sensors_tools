from dataclasses import fields
import dataclasses
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import rospy
from sensor_msgs.msg import Image as ImageMsg
from std_srvs.srv import SetBoolResponse, SetBool
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from scipy.spatial.transform import Rotation

from poses_tools.frame_converter import FrameConverter

from sensors_tools.sensor import SemanticInferenceSensor, SensorConfig
from sensors_tools.bridges import ControllableBridges, get_bridge, get_bridge_config
from sensors_tools.inference import get_inference, get_inference_config
from sensors_tools.utils.semantics_utils import label2rgb

class SemanticNode:
    def __init__(self):
        print("Initializing Semantic Node")
        self.setup()

    def setup(self):
        # Get params
        self.load_config()

        # Asserts are placed here for typing purposes
        assert isinstance(self.frequency, float), "Frequency must be a float"
        assert isinstance(self.camera_name, str), "Camera name must be a string"
        assert isinstance(self.offset_init, list) and len(self.offset_init) == 3, "Offset init must be a list of 3 floats"

        self.sensor = SemanticInferenceSensor(self.cfg)
        self.sensor.setup()

        self.controllable_bridge = self.sensor.cfg.bridge_type in ControllableBridges # enables the move service OR move with timer

        self.pose_transformer = FrameConverter()
        self.pose_transformer.setup_transform_function(self.sensor.cfg.bridge_type, "ros")

        #######################################################
        # RELEVANT CAMERA DATA
        self.camera_info = self.sensor.bridge.camera_info
        #######################################################
        # ROS CAMERA DATA
        self.camera_odom_frame_id = self.camera_name + "_odom"
        
        self.tf_broadcaster = TransformBroadcaster()
        self.static_tf_broadcaster = StaticTransformBroadcaster()

        #TODO: Offset init should be provided by the bridge
        self.static_tf = []
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

        # ROS setup depending on the queried sensors
        if "pose" in self.sensor.cfg.bridge_cfg.data_types:
            self.pub_camera_odometry = rospy.Publisher(f"/{self.camera_name}/odom", Odometry, queue_size=10)

        if "rgb" in self.sensor.cfg.bridge_cfg.data_types:
            self.pub_rgb = rospy.Publisher(f'/{self.camera_name}/rgb/image_raw', ImageMsg, queue_size=10)

        if "depth" in self.sensor.cfg.bridge_cfg.data_types:
            self.pub_depth = rospy.Publisher(f'/{self.camera_name}/depth/image_raw', ImageMsg, queue_size=10)
            self.pub_point_cloud = rospy.Publisher(f'/{self.camera_name}/point_cloud', PointCloud2, queue_size=10)

        if "semantic" in self.sensor.cfg.bridge_cfg.data_types:
            self.pub_semantic_gt = rospy.Publisher(f'/{self.camera_name}/semantic_gt/image_raw', ImageMsg, queue_size=10)
            self.pub_semantic = rospy.Publisher(f'/{self.camera_name}/semantic/image_raw', ImageMsg, queue_size=10)

        if "semantic" in self.sensor.cfg.bridge_cfg.data_types and self.sensor.cfg.inference_type in ["mcd"]:
            self.pub_uncertainty = rospy.Publisher(f'/{self.camera_name}/uncertainty/image_raw', ImageMsg, queue_size=10)

        if self.frequency == -1:
            self.srv_capture_data = rospy.Service(f'/{self.camera_name}/capture_data', SetBool, self.loop_srv)
            if self.controllable_bridge:
                self.srv_move = rospy.Service(f'/{self.camera_name}/move', SetBool, self.move_srv)
        else:
            interval = int(1.0/self.frequency)
            self.timer = rospy.Timer(rospy.Duration(nsecs=interval), self.loop)


    def load_config(self):
        """
            Load config from rosparams
        """
        print("Loading ROSPARAMS")
        bridge_type = rospy.get_param("~bridge_type", "test")
        print(f"Bridge type: {bridge_type}")
        bridge_config_class = get_bridge_config(bridge_type) # type: ignore TODO: Solve
        bridge_parameters = self.load_rosparams(bridge_config_class, "bridge")
        bridge_cfg = bridge_config_class(**bridge_parameters)

        inference_type = rospy.get_param("~inference_type", "deterministic")
        print(f"Inference type: {inference_type}")
        inference_config_class = get_inference_config(inference_type) # type: ignore TODO: Solve
        inference_parameters = self.load_rosparams(inference_config_class, "inference")
        inference_cfg = inference_config_class(**inference_parameters)

        self.cfg = SensorConfig(bridge_type=bridge_type, bridge_cfg=bridge_cfg, inference_type=inference_type, inference_cfg=inference_cfg) # type: ignore TODO: Solve
        print(f"Loaded Sensor")

        if "depth" in self.cfg.bridge_cfg.data_types:
            self.stride = rospy.get_param("~stride", 1)

        self.frequency = rospy.get_param("~frequency", 10.0)
        self.camera_name = rospy.get_param("~camera_name", "cam0")
        self.offset_init = rospy.get_param("~offset_init", [0.,0.,0.])


    def load_rosparams(self, config_class, namespace: str = "") -> dict:
        """
            Load rosparams from the bridge config template
            and use it to initialize the bridge config class
        """
        parameters = {}

        # Get bridge config class fields
        config_fields = fields(config_class)
        # Get bridge config rosparams
        for field in config_fields:
            field_name = field.name
            field_type = field.type
            field_default = field.default
            # In case of factory default, get the default from metadata
            if isinstance(field_default, dataclasses._MISSING_TYPE):
                field_default = field.metadata["default"]
            field_value = rospy.get_param(f"~{namespace}/{field_name}", field_default)
            # Check for path in name to get a Path object
            if field_value is not None and "path" in field_name:
                field_value = Path(field_value) # type: ignore
            print(f"Loaded: {field_name} = {field_value}")
            parameters[field_name] = field_value

        return parameters    

    def to_odom_msg(self, translation: np.ndarray, rotation: Rotation, timestamp: rospy.Time) -> Odometry:
        """
            Convert a 4x4 pose matrix to a ROS Odometry message
        """
        odom_msg = Odometry()
        odom_msg.header.frame_id = self.camera_odom_frame_id
        odom_msg.header.stamp = timestamp
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = translation[0]
        odom_msg.pose.pose.position.y = translation[1]
        odom_msg.pose.pose.position.z = translation[2]
        quat = rotation.as_quat()
        odom_msg.pose.pose.orientation.x = quat[0]
        odom_msg.pose.pose.orientation.y = quat[1]
        odom_msg.pose.pose.orientation.z = quat[2]
        odom_msg.pose.pose.orientation.w = quat[3]
        return odom_msg
    
    def odom_msg_to_tf_msg(self, odom_msg: Odometry) -> TransformStamped:
        """
            Convert a ROS Odometry message to a ROS TransformStamped message
        """
        tf_msg = TransformStamped()
        tf_msg.header.stamp = odom_msg.header.stamp
        tf_msg.header.frame_id = odom_msg.header.frame_id
        tf_msg.child_frame_id = odom_msg.child_frame_id
        tf_msg.transform.translation.x = odom_msg.pose.pose.position.x
        tf_msg.transform.translation.y = odom_msg.pose.pose.position.y
        tf_msg.transform.translation.z = odom_msg.pose.pose.position.z
        tf_msg.transform.rotation.x = odom_msg.pose.pose.orientation.x
        tf_msg.transform.rotation.y = odom_msg.pose.pose.orientation.y
        tf_msg.transform.rotation.z = odom_msg.pose.pose.orientation.z
        tf_msg.transform.rotation.w = odom_msg.pose.pose.orientation.w
        return tf_msg
    
    def update_and_publish_static_transforms(self, timestamp):
        # Update the static transforms
        for static_tf in self.static_tf:
            static_tf.header.stamp = timestamp
            self.static_tf_broadcaster.sendTransform(static_tf)
    
    def pcd_from_rgb_depth(self, rgb: Image.Image, depth: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """A function that converts an rgb image and a depth image to a point cloud
        Depth image is measured with respect to image plane. 
        Depth is along X axis

        Args:
            rgb: PIL Image
            depth: PIL Image

        Returns:
            pcd: a point cloud as a numpy array with shape (H*W, 4)
        """
        rgb_np = np.array(rgb).astype(np.float32) / 255
        depth_np = np.array(depth)
        H = depth_np.shape[0]
        W = depth_np.shape[1]
        columns, rows = np.meshgrid(np.linspace(0, W-1, num=int(W/self.stride)), np.linspace(0, H-1, num=int(H/self.stride))) # type: ignore
        point_depth = depth_np[::self.stride, ::self.stride]
        y = -(columns - self.camera_info.cx) * point_depth / self.camera_info.fx # Originally x : Now -y
        z = -(rows - self.camera_info.cy) * point_depth / self.camera_info.fy # Originally y : Now -z
        x = point_depth # Originally z : Now x
        pcd = np.dstack((x, y, z)).astype(np.float32)
        colors = rgb_np[::self.stride, ::self.stride, :] * 255
        colors = colors.astype(np.uint8)
        # Add alpha channel
        colors = np.dstack((colors, np.ones((colors.shape[0], colors.shape[1], 1), dtype=np.uint8) * 255))
        return pcd, colors

    def generate_point_cloud_msg(self, points_pcd: np.ndarray, points_RGB: np.ndarray, semantic: np.ndarray, semantic_gt: np.ndarray, time_stamp: rospy.Time):
        # Generate the point cloud message from a point cloud of size
        point_cloud_msg = PointCloud2()
        point_cloud_msg.header.stamp = time_stamp
        point_cloud_msg.header.frame_id = "base_link"
        point_cloud_msg.height = 1
        point_cloud_msg.width = points_pcd.shape[0] * points_pcd.shape[1]
        
        sent_n_classes = self.cfg.inference_cfg.num_classes #type: ignore
        
        # We will leave the n_forward_passes for now. Before it was used to send al the MC samples
        self.n_forward_passes = 1
        
        # The PointField is defined with a name, the starting byte offset, the data type and number of elements.
        # rgb is encoded in the standard way so RViz can visualize it. It will be transformed to float32 with view
        # gt_class includes a single value with the ground truth class. It is of type float because it is easier to modify in the dstack
        # prob includes the probabilities of the classes. It has as many elements as classes
        point_cloud_msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                                    PointField('y', 4, PointField.FLOAT32, 1),
                                    PointField('z', 8, PointField.FLOAT32, 1),
                                    PointField('rgb', 12, PointField.UINT32, 1),
                                    PointField('gt_class', 16, PointField.FLOAT32, 1),
                                    PointField('prob', 20, PointField.FLOAT32, sent_n_classes * self.n_forward_passes)]
        point_cloud_msg.is_bigendian = False
        point_cloud_msg.point_step = 20 + 4 * sent_n_classes * self.n_forward_passes
        point_cloud_msg.is_dense = True
        # Turn accumulated_pred into a numpy array with correct order 512 512 n_forward_passes n_classes
        # ADD ONE DIMENSION TO ACCUMULATED PRED TO BE CONSISTENT WITH FORWARDS PASES
        accumulated_pred_out = np.expand_dims(semantic, axis=0)
        accumulated_pred_out = np.transpose(accumulated_pred_out, (2, 3, 0, 1))
        accumulated_pred_out_reshaped = accumulated_pred_out.reshape((self.camera_info.height, self.camera_info.width, self.n_forward_passes * sent_n_classes), order='C')

        # Turn numpy matrix points_RGB of 128 128 4 of type uint8 into a numpy array of size 128 128 1 of type float32
        colors_converted = points_RGB.view(np.float32)
        gt_class_converted = semantic_gt.astype(np.float32)

        points_data = np.dstack((points_pcd, colors_converted, gt_class_converted[::self.stride, ::self.stride], accumulated_pred_out_reshaped[::self.stride, ::self.stride, :]))
        point_cloud_msg.data += points_data.tobytes()

        return point_cloud_msg

    def loop(self, event):
        """
            Loop that captures data and publishes it
        """
        # Get data
        data = self.sensor.get_data()
        curr_timestamp = rospy.Time.now()

        # Publish data
        if "pose" in self.sensor.cfg.bridge_cfg.data_types:
            translation, rotation = data["pose"]
            translation_ros, rotation_ros = self.pose_transformer.transform_function(translation, rotation)
            timestamp = curr_timestamp
            odom_msg = self.to_odom_msg(translation_ros, rotation_ros, timestamp)
            tf_msg = self.odom_msg_to_tf_msg(odom_msg)
            self.tf_broadcaster.sendTransform(tf_msg)
            self.pub_camera_odometry.publish(odom_msg)

        if "rgb" in self.sensor.cfg.bridge_cfg.data_types:
            try:
                rgb_msg = CvBridge().cv2_to_imgmsg(data["rgb"], "rgb8")
                self.pub_rgb.publish(rgb_msg)
            except CvBridgeError as e:
                print(e)

        if "semantic" in self.sensor.cfg.bridge_cfg.data_types:
            try:
                semantic_gt_img = label2rgb(data["semantic_gt"], self.sensor.inference_model.color_map)
                semantic_gt_msg = CvBridge().cv2_to_imgmsg(semantic_gt_img, "rgb8")
                self.pub_semantic_gt.publish(semantic_gt_msg)
            except CvBridgeError as e:
                print(e)

            semantic_msg = CvBridge().cv2_to_imgmsg(data["semantic_rgb"], "rgb8")
            self.pub_semantic.publish(semantic_msg)

        if "depth" in self.sensor.cfg.bridge_cfg.data_types:
            try:
                depth_msg = CvBridge().cv2_to_imgmsg(data["depth"], "passthrough")
                self.pub_depth.publish(depth_msg)
            except CvBridgeError as e:
                print(e)
            # Publish point cloud
            points_pcd, points_RGB = self.pcd_from_rgb_depth(data["rgb"], data["depth"])
            if "semantic" in self.sensor.cfg.bridge_cfg.data_types:
                # TODO: In the future, we can generate a pointcloud message with uncertainty / uncertainty per-class
                pcd_msg = self.generate_point_cloud_msg(points_pcd, points_RGB, data["semantic"], data["semantic_gt"], curr_timestamp)
                self.pub_point_cloud.publish(pcd_msg)

        pass

    def move_srv(self, req: SetBool) -> SetBoolResponse:
        """
            Move the camera
        """
        self.sensor.bridge.move()
        return SetBoolResponse(success=True, message="Moved")

    def loop_srv(self, req: SetBool) -> SetBoolResponse:
        """
            Loop that captures data and publishes it
        """
        self.loop(None)
        return SetBoolResponse(success=True, message="Captured")