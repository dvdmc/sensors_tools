###
#
# This file manages the bridges module.
#
###
from dataclasses import fields
from typing import Literal, Union, TYPE_CHECKING

# We use dynamic imports to avoid not used requirements.
# For this to work, used types must be "forward declared" in quotes 
# (see https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING)
# Then, if bridge is selected, we can import the required module
if TYPE_CHECKING:
    from .semantic import SemanticInference, SemanticInferenceConfig
    from .semantic_mcd import SemanticMCDInference, SemanticMCDInferenceConfig

InferenceConfig = Union['SemanticInferenceConfig', 'SemanticMCDInferenceConfig']

InferenceType = Literal["deterministic", "mcd"]

Inference = Union['SemanticInference', 'SemanticMCDInference']

def get_inference_config(inference_type: InferenceType):
    assert inference_type is not None, "Inference type must be specified"

    if inference_type == "deterministic":
        from .semantic import SemanticInferenceConfig
        return SemanticInferenceConfig
    elif inference_type == "mcd":
        from .semantic_mcd import SemanticMCDInferenceConfig
        return SemanticMCDInferenceConfig
    else:
        raise NotImplementedError("Inference type not implemented")

def get_inference(inference_type: InferenceType, inference_cfg: InferenceConfig) -> Inference:

    if inference_type == "deterministic":
        from .semantic import SemanticInference, SemanticInferenceConfig
        assert isinstance(inference_cfg, SemanticInferenceConfig), "Inference cfg must be of type SemanticInferenceConfig"
        return SemanticInference(inference_cfg)
    elif inference_type == "mcd":
        from .semantic_mcd import SemanticMCDInference, SemanticMCDInferenceConfig
        assert isinstance(inference_cfg, SemanticMCDInferenceConfig), "Inference cfg must be of type SemanticMCDInferenceConfig"
        return SemanticMCDInference(inference_cfg)
    else:
        raise NotImplementedError("Inference type not implemented")