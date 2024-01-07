###
#
# Use this file to launch the test_bridge.py script
#
###
from pathlib import Path

import cv2
from matplotlib import pyplot as plt
import numpy as np
from test_bridge import TestBridge, TestBridgeConfig

if __name__ == '__main__':
    config = TestBridgeConfig()
    config.data_types = ["rgb", "depth", "semantic", "poses"]
    config.dataset_path = Path("./test_data/")
    config.width = 512
    config.height = 512

    bridge = TestBridge(config)
    bridge.setup()


    fig, ax = plt.subplots(1,3)

    # Show the images
    for i in range(bridge.data_length):
        data = bridge.get_data()
        ax[0].imshow(data["rgb"])
        # Titles
        ax[0].set_title("RGB")

        print(f"Depth min: {np.min(data['depth'])}, max: {np.max(data['depth'])}")
        ax[1].imshow(data["depth"], cmap="jet")
        ax[1].set_title("Depth")

        ax[2].imshow(data["semantic_gt"])
        ax[2].set_title("Semantic")

        print("Pose: ", data["pose"])
        
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

        bridge.increment_seq()