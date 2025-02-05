from dataclasses import dataclass, field
from pathlib import Path
import time
import random  # Just for setting the random seed
from typing import Literal, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
import torch  # Just for setting the random seed

from sensors_tools.bridges import BridgeConfig, BridgeType, get_bridge

from sensors_tools.inference.semantic import SemanticInferenceConfig
from sensors_tools.inference.semantic_mcd import SemanticMCDInferenceConfig
from sensors_tools.inference.open_clip_semantic import OpenClipSemanticInferenceConfig
from sensors_tools.inference.open_trident_semantic import OpenTridentSemanticInferenceConfig
from sensors_tools.inference import get_inference
from sensors_tools.utils.semantics_utils import apply_label_map, get_label_mapper
from sensors_tools.utils.random_utils import set_seed


@dataclass
class SensorConfig:
    """
    Configuration class for DeterministicSensor
    """

    bridge_cfg: BridgeConfig
    """ Bridge configuration """

    bridge_type: BridgeType
    """ Type of bridge to be used """

    gt_labels_mapper: Optional[str] = None
    """ Name reference the label map for derived datasets. Example: coco_voc_2_pascal_8 """

    inference_cfg: Optional[Union[SemanticInferenceConfig, SemanticMCDInferenceConfig, OpenClipSemanticInferenceConfig, OpenTridentSemanticInferenceConfig]] = field(
        default_factory=SemanticInferenceConfig, metadata={"default": SemanticInferenceConfig()}
    )
    """ Inference configuration """

    save_inference: bool = False
    """ Whether to save the inference results """

    save_inference_path: Optional[Path] = None
    """ Path to save the inference results """

class SemanticInferenceSensor:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg
        self.seq = 0
        set_seed(42)

    def setup(self):
        # Setup the inference models. 
        # This is done before the bridge to allow for loading the model and weights before starting the robot.
        print("Setting up the inference model")
        if "semantic" in self.cfg.bridge_cfg.data_types:
            assert self.cfg.inference_cfg is not None, "Inference cfg must be specified if semantic data is requested"
            self.inference_model = get_inference(self.cfg.inference_cfg)
            self.inference_model.setup()
            if self.cfg.save_inference:
                assert (
                    self.cfg.save_inference_path is not None
                ), "save_inference_path must be specified if save_inference is True"
                self.pred_path = self.cfg.save_inference_path / "pred"
                self.pred_rgb_path = self.cfg.save_inference_path / "pred_rgb"
                self.pred_path.mkdir(parents=True, exist_ok=True)
                self.pred_rgb_path.mkdir(parents=True, exist_ok=True)

        # Setup the bridge
        print("Setting up the bridge")
        self.bridge = get_bridge(self.cfg.bridge_type, self.cfg.bridge_cfg)
        self.bridge.setup()

        # If there is a GT label mapper, load it
        print(self.cfg.gt_labels_mapper)
        self.gt_labels_mapper = None
        if self.cfg.gt_labels_mapper is not None:
            print("Found labels mapper")
            self.gt_labels_mapper = get_label_mapper(self.cfg.gt_labels_mapper)


    def get_data(self) -> Optional[dict]:
        if not self.bridge.ready:
            return None

        start = time.time()
        data = self.bridge.get_data()
        print(f"Time to get data: {time.time() - start}")
        img = data["rgb"]
        if "semantic" in self.cfg.bridge_cfg.data_types:
            assert self.cfg.inference_cfg is not None, "Inference cfg must be specified if semantic data is requested"

            start = time.time()
            out = self.inference_model.get_prediction(img)
            print(f"Time to get prediction: {time.time() - start}")

            if self.gt_labels_mapper is not None and "semantic_gt" in data:
                data["semantic_gt"] = apply_label_map(data["semantic_gt"], self.gt_labels_mapper)

            data["semantic"] = out["probs"]
            data["semantic_rgb"] = out["img_out"]

            # TODO: Populate inside the inference_model by providing the "data" to make it more generic
            if "mcd" in self.cfg.inference_cfg.model_name:
                data["epistemic_var"] = out["epistemic_var"]
                data["acc_probs"] = out["acc_probs"]

            if self.cfg.save_inference:
                np.save(self.pred_path / f"{self.seq}.npy", out["probs"])
                semantic_rgb = Image.fromarray(out["img_out"])
                semantic_rgb.save(self.pred_rgb_path / f"{self.seq}.png")
                self.seq += 1

        return data

    def move(self) -> bool:
        return self.bridge.move()