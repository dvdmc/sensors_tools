from dataclasses import dataclass, field
from typing import List, Literal

from PIL import Image
import numpy as np

import airsim #type: ignore

from airsim_tools.depth_conversion import depth_conversion
from airsim_tools.semantics import airsim2class_id

from bridges.base_bridge import BaseBridge, BridgeConfig

AirsimSensorDataTypes = Literal["rgb", "depth", "semantic", "poses"]
"""
    List of sensor data to query.
    - "poses": query poses.
    - "rgb": query rgb images.
    - "depth": query depth images.
    - "semantic": query semantic images.
"""

@dataclass
class AirsimBridgeConfig(BridgeConfig):
    """
        Configuration class for AirsimBridge
    """
    data_types: List[AirsimSensorDataTypes] = field(default_factory=list, metadata={"default": ["rgb", "poses"]})
    """ Data types to query """

    semantic_config: List[tuple] = field(default_factory=list, metadata={"default": []})
    """ Semantic configuration """

    width: float = 512
    """ Image width """

    height: float = 512
    """ Image height """

    fov_h: float = 54.4
    """ Horizontal field of view """

class AirsimBridge(BaseBridge):
    """
        Bridge for Airsim
    """
    def __init__(self, cfg: AirsimBridgeConfig):
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
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        
        # Sim config data TODO: Move to config file
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
        self.client.simSetFocusAperture(7.0, "0")  # Avoids depth of field blur
        self.client.simSetFocusDistance(100.0, "0")  # Avoids depth of field blur
        #######################################################

        # Set the data to query
        self.query_data = []
        if "rgb" in self.cfg.data_types:
            self.query_data.append(airsim.ImageRequest("0", airsim.ImageType.Scene, False, False))
        if "depth" in self.cfg.data_types:
            self.query_data.append(airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True, False))
        if "semantic" in self.cfg.data_types:
            self.query_data.append(airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False))

        if "semantic" in self.cfg.data_types:
            self.setup_semantic_config()
    
    def setup_semantic_config(self):
        # Set all objects in the scene to label 0 in the beggining
        if self.cfg.semantic_config is not None:
            for object_id in self.client.simListSceneObjects(): #type: ignore
                changed = False
                for object_str, label in self.cfg.semantic_config:
                    if object_str in object_id:
                        changed = True
                        success = self.client.simSetSegmentationObjectID(object_id, label)
                        if not success:
                            print("Could not set segmentation object ID for {}".format(object_id))
                        else:
                            print("Changed object ID to {} for {}".format(label, object_id))
                        break
                if not changed:  # TODO: Check if this is faster than just setting all objects to 0
                    self.client.simSetSegmentationObjectID(object_id, 0)
            print("Finished setting object IDs")

    def process_img_responses(
        self, responses: List[airsim.ImageResponse]
    ) -> dict:
        """
        Process the data from AirSim.
        """
        img_data = {}
      
        for response in responses:
            if response.image_type == airsim.ImageType.Scene:
                np_image = (
                    np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    .reshape(response.height, response.width, 3)
                    .copy()
                )
                correct_image = np_image[:, :, ::-1]
                image = Image.fromarray(correct_image)
                img_data["rgb"] = image
            elif response.image_type == airsim.ImageType.DepthPerspective:
                img_depth_meters = airsim.list_to_2d_float_array(
                    response.image_data_float, response.width, response.height
                )
                img_depth_meters_corrected = depth_conversion(img_depth_meters, self.fx)
                img_data["depth"] = img_depth_meters_corrected

            elif response.image_type == airsim.ImageType.Segmentation:
                # Transform Airsim segmentation image to a different color system
                img_buffer = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                img_rgb_airsim = img_buffer.reshape(response.height, response.width, 3)
                np_rgb_airsim = img_rgb_airsim[:,:,::-1]
                # Get the semantic image
                semantic = airsim2class_id(np_rgb_airsim)
                img_data["semantic"] = semantic

        return img_data
    
    def get_data(self):
        """
            Get data from the bridge
        """
        data = {}
        if "poses" in self.cfg.data_types:
            poses = self.client.simGetVehiclePose()
            data["poses"] = poses

        responses = self.client.simGetImages(self.query_data)
        img_data = self.process_img_responses(responses)

        data.update(img_data)

        return data
