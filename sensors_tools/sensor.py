from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from sensors_tools.bridges import BridgeConfig, BridgeType, get_bridge

from sensors_tools.inference.semantic import SemanticInferenceConfig
from sensors_tools.inference.semantic_mcd import SemanticMCDInferenceConfig
from sensors_tools.inference import get_inference

@dataclass
class SensorConfig:
    """
        Configuration class for DeterministicSensor
    """
    bridge_cfg: BridgeConfig
    """ Bridge configuration """

    bridge_type: BridgeType
    """ Type of bridge to be used """

    inference_cfg: Optional[Union[SemanticInferenceConfig, SemanticMCDInferenceConfig]] = field(default_factory=SemanticInferenceConfig, metadata={"default": SemanticInferenceConfig()})
    """ Inference configuration """
    
    inference_type: Literal["deterministic", "mcd"] = "deterministic"
    """ Type of inference to be used """
    
class SemanticInferenceSensor:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg

    def setup(self):
        # Setup the bridge
        self.bridge = get_bridge(self.cfg.bridge_type, self.cfg.bridge_cfg)
        self.bridge.setup()

        # Setup the inference models
        if "semantic" in self.cfg.bridge_cfg.data_types:
            assert self.cfg.inference_type is not None, "Inference type must be specified if semantic data is requested"
            assert self.cfg.inference_cfg is not None, "Inference cfg must be specified if semantic data is requested"
            self.inference_model = get_inference(self.cfg.inference_type, self.cfg.inference_cfg)
            self.inference_model.setup()

    def get_data(self):
        data = self.bridge.get_data()
        img = data["rgb"]
        if "semantic" in self.cfg.bridge_cfg.data_types:
            out = self.inference_model.get_prediction(img)
            data["semantic"] = out["probs"]
            data["semantic_rgb"] = out["img_out"]
        
            if self.cfg.inference_type == "mcd":
                data["epistemic_var"] = out["epistemic_var"]
                data["acc_probs"] = out["acc_probs"]
        return data