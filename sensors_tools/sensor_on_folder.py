#
#
# This file instantiates a sensor and prints the results (mainly for testing semantic inference)
#
#

from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from sensor import SensorConfig, SemanticSegmentationSensor
from sensors_tools.bridges.folder_bridge import FolderBridgeConfig
from sensors_tools.inference.semantic_segmentation import SemanticSegmentationTridentConfig

if __name__ == '__main__':
    # Setup the sensor
    bridge_cfg = FolderBridgeConfig(data_types=["rgb"], dataset_path=Path("/home/david/datasets/folder/"))
    sem_cfg = SemanticSegmentationTridentConfig(semantic_feature_type="probability_vector", semantic_dataset_type="custom_set", custom_set_labels=["grass","path"])
    cfg = SensorConfig(
        bridge_cfg = bridge_cfg,
        bridge_type = "folder",
        inference_cfg = sem_cfg,
        save_inference=True,
        save_inference_path=Path("/home/david/datasets/folder/out/"),
    )

    sensor = SemanticSegmentationSensor(cfg)
    sensor.setup()

    # Process dataset
    sensor.get_data()
    while sensor.move():
        sensor.get_data()