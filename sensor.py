from dataclasses import dataclass, field
from typing import Literal, Optional, Union

from bridges import BridgeConfig, BridgeType, get_bridge

from inference.semantic import SemanticInferenceConfig, SemanticInference
from inference.semantic_mcd import SemanticMCDInference, SemanticMCDInferenceConfig

@dataclass
class SensorConfig:
    """
        Configuration class for DeterministicSensor
    """
    bridge_cfg: BridgeConfig
    """ Bridge configuration """

    bridge_type: BridgeType
    """ Type of bridge to be used """

    inference_cfg: Optional[SemanticInferenceConfig | SemanticMCDInferenceConfig] = field(default_factory=SemanticInferenceConfig, metadata={"default": SemanticInferenceConfig()})
    """ Inference configuration """
    
    inference_type: Literal["deterministic", "mcd"] = "deterministic"
    """ Type of inference to be used """

def get_semantic_inference_model(cfg: SensorConfig) -> SemanticInference:
    assert cfg.inference_cfg is not None, "Inference cfg must be specified if semantic data is requested"
    assert cfg.inference_type is not None, "Inference type must be specified if semantic data is requested"
    if cfg.inference_type == "deterministic":
        assert isinstance(cfg.inference_cfg, SemanticInferenceConfig), "Inference cfg must be of type SemanticInferenceConfig"
        return SemanticInference(cfg.inference_cfg)
    elif cfg.inference_type == "mcd":
        assert isinstance(cfg.inference_cfg, SemanticMCDInferenceConfig), "Inference cfg must be of type SemanticMCDInferenceConfig"
        return SemanticMCDInference(cfg.inference_cfg)
    else:
        raise NotImplementedError("Inference type not implemented")
    
class SemanticInferenceSensor:
    def __init__(self, cfg: SensorConfig):
        self.cfg = cfg

    def setup(self):
        # Setup the bridge
        self.bridge = get_bridge(self.cfg.bridge_type, self.cfg.bridge_cfg)
        self.bridge.setup()

        # Setup the inference models
        if "semantic" in self.cfg.bridge_cfg.data_types:
            self.inference_model = get_semantic_inference_model(self.cfg)
            self.inference_model.setup()

    def get_data(self):
        data = self.bridge.get_data()
        img = data["rgb"]
        if "semantic" in self.cfg.bridge_cfg.data_types:
            probs, img_out = self.inference_model.get_prediction(img)
            data["semantic"] = img_out

        return data